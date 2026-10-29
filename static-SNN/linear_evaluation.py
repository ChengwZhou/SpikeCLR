import os
import math
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import datetime
import gc

from BartonTwins import BartonTwins, BartonTwins_imagenet
from BartonTwins_spiking import BartonTwinsSpiking, BartonTwinsSpiking_imagenet
from model import load_optimizer, save_model
from utils import yaml_config_hook

from modules.transformations import DataTransforms, DataTransforms_imagenet
from modules import get_resnet, get_resnet_spiking, modify_resnet_model, LogisticRegression
from modules.spike_layer import MixedLIF, LIFt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import TinyImageNetDataset


def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    if world_size > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt
    else:
        return tensor

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    warmup_epochs = 20
    if epoch < warmup_epochs:
        # lr = args.lr * epoch / warmup_epochs
        lr = args.lr
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inference(args, loader, bt_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, _, _, = bt_model(x, x)

        if args.spiking:
            # representations
            h = h.mean(0)  # [B, C]
            h = h.detach()
            feature_vector.extend(h.cpu().numpy())

            # labels
            labels_vector.extend(y)
        else:
            h = h.detach()
            feature_vector.extend(h.cpu().numpy())
            labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(args, bt_model, train_loader, test_loader, device):
    train_X, train_y = inference(args, train_loader, bt_model, device)
    test_X, test_y = inference(args, test_loader, bt_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, model, criterion, optimizer, epoch, world_size):
    loss_epoch = 0
    accuracy_epoch_top1 = 0
    accuracy_epoch_top5 = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch + step / len(loader), args)

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y.long())

        # top-1
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch_top1 += acc
        # top-5
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        acc_5 = pred.eq(y[None])
        acc_5 = acc_5.flatten().sum(dtype=torch.float32) / y.size(0)
        accuracy_epoch_top5 += acc_5

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    loss_epoch = reduce_tensor(torch.tensor(loss_epoch, device=args.device), world_size)
    accuracy_epoch_top1 = reduce_tensor(torch.tensor(accuracy_epoch_top1, device=args.device), world_size)
    accuracy_epoch_top5 = reduce_tensor(torch.tensor(accuracy_epoch_top5, device=args.device), world_size)

    return loss_epoch.item(), accuracy_epoch_top1.item(), accuracy_epoch_top5.item()


def test(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch_top1 = 0
    accuracy_epoch_top5 = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)  # [B, num_classes]
        loss = criterion(output, y.long())

        # top-1
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch_top1 += acc
        # top-5
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        acc_5 = pred.eq(y[None])
        acc_5 = acc_5.flatten().sum(dtype=torch.float32) / y.size(0)
        accuracy_epoch_top5 += acc_5

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch_top1, accuracy_epoch_top5


def main(rank, world_size, args):
    # init DDP
    args.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=120))
        torch.cuda.set_device(rank)

    if rank == 0:
        print("Now start loading dataset")
    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=DataTransforms(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=DataTransforms(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=DataTransforms(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=False,
            download=True,
            transform=DataTransforms(size=args.image_size).test_transform,
        )
    elif args.dataset in ("Oxford102", "Flowers102"):
        args.n_classes = 102
        train_split = ("train", "val")
        train_dataset = torchvision.datasets.Flowers102(
            args.dataset_dir,
            split="test",
            download=True,
            transform=DataTransforms(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.Flowers102(
            args.dataset_dir,
            split="train",
            download=True,
            transform=DataTransforms(size=args.image_size).test_transform,
        )
    elif args.dataset == "Tiny-ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms_imagenet(size=args.image_size).train_evaluation_transform,
        )
        # test_dataset = torchvision.datasets.ImageFolder(
        #     os.path.join(args.dataset_dir, 'val'),
        #     transform=DataTransforms(size=args.image_size).test_transform,
        # )
        test_dataset = TinyImageNetDataset(args.dataset_dir,
                                           'val',
                                           DataTransforms_imagenet(size=args.image_size).test_transform)
    elif args.dataset == "ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms_imagenet(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'val'),
            transform=DataTransforms_imagenet(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    # Create a 1%/10%/100% data subset using the random indices
    num_samples = int(len(train_dataset) * args.eval_proportion)
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    if world_size > 1:
        sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.logistic_batch_size,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=args.num_workers,
        sampler=sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # initialize active function of lif
    if args.act_func == 'MixedLIF':
        Act_func = MixedLIF
    else:
        Act_func = LIFt

    # initialize ResNet, encoder is resnet/resnet_snn
    # load pre-trained model from checkpoint
    if args.spiking:
        if "resnet" in args.model:
            backbone = get_resnet_spiking(args.model, args.timestep, args.sync_norm, Act_func, args.n_classes)
            n_features = backbone.fc.in_features  # get dimensions of fc layer
            backbone.fc = nn.Identity()
            # convert to cifar-fitted structure
            if "CIFAR" in args.dataset and args.image_size==32:
                backbone = modify_resnet_model(backbone)

        # if "ImageNet" in args.dataset:   # Notice
        bt_model = BartonTwinsSpiking_imagenet(backbone, in_dim=n_features, out_dim=args.projection_dim, act_func=Act_func,
                                       timestep=args.timestep)
        # else:
        #     bt_model = BartonTwinsSpiking(backbone, in_dim=n_features, out_dim=args.projection_dim, act_func=Act_func,
        #                                timestep=args.timestep)

    else:
        if "resnet" in args.model:
            backbone = get_resnet(args.model, args.n_classes)
            n_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            # convert to cifar-fitted structure
            if "CIFAR" in args.dataset and args.image_size==32:
                backbone = modify_resnet_model(backbone)

        # if "ImageNet" in args.dataset:
        bt_model = BartonTwins_imagenet(backbone, in_dim=n_features, out_dim=args.projection_dim)
        # else:
        #     bt_model = BartonTwins(backbone, in_dim=n_features, out_dim=args.projection_dim)

    # reload from checkpoint
    model_fp = os.path.join(
        args.model_path, "checkpoint_epoch_{}.tar".format(args.epoch_num)
    )
    checkpoint = torch.load(model_fp, map_location=args.device)
    bt_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Model and optimizer loaded from checkpoint '{model_fp}' at epoch {checkpoint['epoch']}.")

    bt_model = bt_model.to(args.device)
    bt_model.eval()

    ## Logistic Regression
    n_classes = args.n_classes  # CIFAR-10 / STL-10 / CIFAR-100
    model = LogisticRegression(n_features, n_classes)
    model = model.to(args.device)
    if world_size > 1:  # DDP
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1*1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        args, bt_model, train_loader, test_loader, args.device
    )

    if args.spiking:
        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, args.logistic_batch_size
        )
    else:
        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, args.logistic_batch_size
        )

    for epoch in range(args.logistic_epochs):
        if world_size > 1:  # shuffle
            sampler.set_epoch(epoch)
        loss_epoch, accuracy_epoch_top1, accuracy_epoch_top5= train(
            args, arr_train_loader, model, criterion, optimizer, epoch, world_size
        )
        if rank == 0 and epoch % 20 == 0:
            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy top-1: {accuracy_epoch_top1 / len(arr_train_loader)}\t Accuracy top-5: {accuracy_epoch_top5 / len(arr_train_loader)}"
            )

    # final testing
    if rank == 0:
        loss_epoch, accuracy_top1, accuracy_top5 = test(
            args, arr_test_loader, model, criterion
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy top-1: {accuracy_top1 / len(arr_test_loader)}\t Accuracy top-5: {accuracy_top5 / len(arr_test_loader)}"
        )
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.lr = float(args.lr)
    print(vars(args))

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"

    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(0, 1, args)