import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import datetime

from BartonTwins import BartonTwins
from BartonTwins_spiking import BartonTwinsSpiking
from model import load_optimizer, save_model

from utils import yaml_config_hook

from modules.transformations import DataTransforms, DataTransforms_imagenet
from modules import get_resnet, get_resnet_spiking, modify_resnet_model, LogisticRegression
from modules.spike_layer import LIF
from dataset import TinyImageNetDataset

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


def train(args, loader, model, criterion, optimizer, world_size):
    loss_epoch = 0
    accuracy_epoch_top1 = 0
    accuracy_epoch_top5 = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        if args.spiking:
            output, _ = model(x)
        else:
            output = model(x)

        loss = criterion(output, y)

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

        with torch.no_grad():
            if args.spiking:
                output, _ = model(x)
            else:
                output = model(x)

        loss = criterion(output, y)

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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     config = yaml_config_hook("./config/config.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))
#
#     args = parser.parse_args()
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(vars(args))


def main(rank, world_size, args):
    # init DDP
    args.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size,
                                timeout=datetime.timedelta(seconds=120))
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
            split="train",
            download=True,
            transform=DataTransforms(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.Flowers102(
            args.dataset_dir,
            split="test",
            download=True,
            transform=DataTransforms(size=args.image_size).test_transform,
        )
    elif args.dataset == "Tiny-ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms_imagenet(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = TinyImageNetDataset(args.dataset_dir,
                                           'val',
                                           DataTransforms_imagenet(size=args.image_size).test_transform)
    elif args.dataset == "ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms(size=args.image_size).train_evaluation_transform,
        )
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'val'),
            transform=DataTransforms(size=args.image_size).test_transform,
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

    if args.spiking:
        if "resnet" in args.model:
            model = get_resnet_spiking(args.model, args.timestep, True, LIF, args.n_classes)
            if "CIFAR" in args.dataset and args.image_size==32:
                model = modify_resnet_model(model)

    else:
        if "resnet" in args.model:
            model = get_resnet(args.model, args.n_classes)
            if "CIFAR" in args.dataset and args.image_size==32:
                model = modify_resnet_model(model)

    # Load weights from pre-trained file
    model_fp = os.path.join(
        args.model_path, "checkpoint_epoch_{}.tar".format(args.epoch_num)
    )
    checkpoint = torch.load(model_fp, map_location=args.device)

    # Map the keys from the source model to the target model
    pretrained_dict = checkpoint['model_state_dict']
    print(f"Model and optimizer loaded from checkpoint '{model_fp}' at epoch {checkpoint['epoch']}.")
    new_dict = {}
    for k, v in pretrained_dict.items():
        if "backbone." in k:
            # Remove the 'encoder.' prefix and use the rest as the key for the target model
            key = k.replace("backbone.", "")
            new_dict[key] = v

    model.load_state_dict(new_dict, strict=False)
    model = model.to(args.device)
    if world_size > 1:  # DDP
        model = DDP(model, device_ids=[rank])

    # Freeze param. besides fc
    # for param in model.parameters():
    #     param.requires_grad = False
    # # Only the fc layer has requires_grad = True
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1*1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.logistic_epochs):
        if world_size > 1:  # shuffle
            sampler.set_epoch(epoch)
        loss_epoch, accuracy_epoch_top1, accuracy_epoch_top5 = train(
            args, train_loader, model, criterion, optimizer, world_size
        )
        if rank == 0 and epoch % 20 == 0:
            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy top-1: {accuracy_epoch_top1 / len(train_loader)}\t Accuracy top-5: {accuracy_epoch_top5 / len(train_loader)}"
            )

    # final testing
    if rank == 0:
        loss_epoch, accuracy_top1, accuracy_top5 = test(
            args, test_loader, model, criterion
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy top-1: {accuracy_top1 / len(test_loader)}\t Accuracy top-5: {accuracy_top5 / len(test_loader)}"
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