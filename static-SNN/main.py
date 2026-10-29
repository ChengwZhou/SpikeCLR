import math
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import os

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from modules import get_resnet, get_resnet_spiking, SynctdBatchNorm, modify_resnet_model
from loss import (BarlowTwinsLoss, BarlowTwinsTemporalLoss,
                  SimCLRLoss, SimCLRTemporalLoss,
                  VICRegLoss, VICRegTemporalLoss)
from modules.spike_layer import MixedLIF, LIFt

from BartonTwins import BartonTwins, BartonTwins_imagenet
from BartonTwins_spiking import BartonTwinsSpiking, BartonTwinsSpiking_imagenet
from utils import yaml_config_hook
from model import load_optimizer, save_model, save_model_top_k
from modules.transformations import DataTransforms, DataTransforms_imagenet

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
# amp
from torch.cuda.amp import GradScaler, autocast
# spikingjelly
# from spikingjelly.activation_based import functional


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    warmup_epochs = 0
    if epoch < warmup_epochs:
        lr = args.lr * epoch / warmup_epochs
        # lr = args.lr
    else:
        lr = 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (1000 - warmup_epochs)))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * args.lr
    optimizer.param_groups[0]['lr'] = lr * args.lr
    optimizer.param_groups[1]['lr'] = lr * args.lr_bias


def train(args, train_loader, model, criterion, optimizer, writer, scaler):
    loss_epoch = 0

    forward_time_total = 0.0
    loss_time_total = 0.0
    backward_time_total = 0.0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        # reset memory tracker
        torch.cuda.reset_peak_memory_stats(args.device)

        optimizer.zero_grad()
        x_i = x_i.cuda(args.device, non_blocking=True)
        x_j = x_j.cuda(args.device, non_blocking=True)

        adjust_learning_rate(optimizer, args.current_epoch + step / len(train_loader), args)

        # x_i = x_i.contiguous(memory_format=torch.channels_last)
        # x_j = x_j.contiguous(memory_format=torch.channels_last)


        start_forward = time.time()
        with autocast():
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            # spiking but not temporal loss
            forward_time = time.time() - start_forward
            forward_time_total += forward_time

            if args.spiking:
                if not args.temporal_loss:
                    z_i = z_i.mean(0)
                    z_j = z_j.mean(0)
            start_loss = time.time()
            loss = criterion(z_i, z_j)
            loss_time = time.time() - start_loss
            loss_time_total += loss_time

        # use amp loss scaler
        start_backward = time.time()
        scaler.scale(loss).backward()  # gradient scale

        # gradient clipping
        if not args.clip_grad_norm=="None":
            scaler.unscale_(optimizer)  # unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))

        scaler.step(optimizer)
        scaler.update()
        backward_time = time.time() - start_backward
        backward_time_total += backward_time

        # functional.reset_net(model)

        # use all_reduce to store loss correctly
        # if dist.is_available() and dist.is_initialized():
        #     loss = loss.data.clone()
        #     dist.all_reduce(loss.div_(dist.get_world_size()))

        # if args.rank == 0 and step % 10 == 0:
        #     print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        # Peak memory
        peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024  # in MB

        if args.rank == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        if args.rank == 0 and step % 10 == 0:
            print(f"Step [{step}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Fwd: {forward_time:.3f}s | "
                  f"LossCalc: {loss_time:.3f}s | "
                  f"BP: {backward_time:.3f}s | "
                  f"Peak Mem: {peak_memory_mb:.2f} MB")

        loss_epoch += loss.item()
    return loss_epoch

def main(gpu, args):
    args.rank = args.nr * args.gpus + gpu
    args.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    if args.world_size > 1:
        dist.init_process_group("nccl", init_method="env://", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Start Dataset Verification")
    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=DataTransforms(size=32),
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=DataTransforms(size=32),
        )
    elif args.dataset == "Tiny-ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms_imagenet(size=args.image_size),
        )
    elif args.dataset == "ImageNet":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=DataTransforms_imagenet(size=args.image_size),
        )
    else:
        raise NotImplementedError
    print ("Dataset Verification Finished")

    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    else:
        train_sampler = None

    dataloader_train_ssl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=args.pin_memory
    )

    # initialize active function of lif
    if args.act_func == 'MixedLIF':
        Act_func = MixedLIF
    else:
        Act_func = LIFt

    # initialize ResNet, encoder is resnet/resnet_snn
    if args.spiking:
        if "resnet" in args.model:
            backbone = get_resnet_spiking(args.model, args.timestep, args.sync_norm, Act_func, args.n_classes)
            n_features = backbone.fc.in_features  # get dimensions of fc layer
            backbone.fc = nn.Identity()
            # convert to cifar-fitted structure
            if "CIFAR" in args.dataset:
                backbone = modify_resnet_model(backbone)

        if "ImageNet" in args.dataset:
            model = BartonTwinsSpiking_imagenet(backbone, in_dim=n_features, out_dim=args.projection_dim, act_func=Act_func,
                                       timestep=args.timestep)
        else:
            model = BartonTwinsSpiking(backbone, in_dim=n_features, out_dim=args.projection_dim, act_func=Act_func,
                                       timestep=args.timestep)
    else:
        if "resnet" in args.model:
            backbone = get_resnet(args.model, args.n_classes)
            n_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            # convert to cifar-fitted structure
            if "CIFAR" in args.dataset:
                backbone = modify_resnet_model(backbone)

        if "ImageNet" in args.dataset:
            model = BartonTwins_imagenet(backbone, in_dim=n_features, out_dim=args.projection_dim)
        else:
            model = BartonTwins(backbone, in_dim=n_features, out_dim=args.projection_dim)
        # model = BartonTwins(backbone, in_dim=n_features, out_dim=args.projection_dim)

    model.to(args.device)
    # amp scaler
    scaler = GradScaler()
    if args.rank == 0:
        print('Using native Torch AMP. Training in mixed precision.')

    param_weights = []
    param_biases = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            param_biases.append(param)
        else:
            param_weights.append(param)

    parameters = [
        {'params': param_weights, 'lr': args.lr},
        {'params': param_biases, 'lr': args.lr_bias}
    ]

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, parameters, len(dataloader_train_ssl))
    _temporal = args.temporal_loss and args.spiking
    if args.loss_type == "simclr":
        if _temporal:
            criterion = SimCLRTemporalLoss(device=args.device, world_size=args.world_size,
                                           temperature=args.temperature,
                                           cross_temporal=args.cross_temporal,
                                           simplified_loss=args.simplified_loss)
        else:
            criterion = SimCLRLoss(device=args.device, world_size=args.world_size,
                                   temperature=args.temperature)
    elif args.loss_type == "vicreg":
        if _temporal:
            criterion = VICRegTemporalLoss(device=args.device, world_size=args.world_size,
                                           sim_coeff=args.vicreg_sim_coeff,
                                           std_coeff=args.vicreg_std_coeff,
                                           cov_coeff=args.vicreg_cov_coeff,
                                           cross_temporal=args.cross_temporal,
                                           simplified_loss=args.simplified_loss)
        else:
            criterion = VICRegLoss(device=args.device, world_size=args.world_size,
                                   sim_coeff=args.vicreg_sim_coeff,
                                   std_coeff=args.vicreg_std_coeff,
                                   cov_coeff=args.vicreg_cov_coeff)
    elif _temporal:
        criterion = BarlowTwinsTemporalLoss(device=args.device, world_size=args.world_size,
                                            cross_temporal=args.cross_temporal,
                                            simplified_loss=args.simplified_loss,
                                            proj_dim=args.projection_dim)
    else:
        criterion = BarlowTwinsLoss(device=args.device, world_size=args.world_size,
                                    proj_dim=args.projection_dim)
    criterion = criterion.to(args.device)

    # resume
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_epoch_{}.tar".format(args.epoch_num)
        )
        checkpoint = torch.load(model_fp, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Model and optimizer loaded from checkpoint '{model_fp}' at epoch {checkpoint['epoch']}.")

    # channels_last
    # model = model.to(memory_format=torch.channels_last)

    # DDP
    if args.world_size > 1:
        if not args.spiking:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            # model = SynctdBatchNorm.convert_sync_tdBatchNorm(model)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            criterion = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criterion)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    writer = None
    if args.rank == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = args.start_epoch
    # ONLY save top k models
    top_models = []
    loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.rank == 0:
            print('===================================')
            start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, dataloader_train_ssl, model, criterion, optimizer, writer, scaler)
        loss_list.append(loss_epoch)

        if scheduler:
            scheduler.step()

        if args.rank == 0:
            epoch_time = time.time() - start_time
            # save_model(args, model, optimizer, scaler)
            save_model_top_k(args, model, optimizer, scaler, loss_epoch, top_models)
            writer.add_scalar("Loss/train", loss_epoch / len(dataloader_train_ssl), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(dataloader_train_ssl)}\t lr: {round(lr, 7)}\t Time: {epoch_time:.2f}s"
            )
            args.current_epoch += 1

    ## end training
    if args.rank == 0:
        print("Training complete. Top 5 models:")
        for _, epoch, model_path in sorted(top_models, reverse=True):
            print(f"Epoch {epoch}: {model_path}")
        loss_array = np.array(loss_list)
        out = os.path.join(args.model_path, "loss.npy")
        np.save(out, loss_array)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(getattr(args, "master_port", "8090"))

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = None
    args.world_size = args.gpus * args.nodes
    args.lr = float(args.lr)
    print(vars(args))

    cudnn.benchmark = True

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} , waiting until all world_size join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
