import datetime
import os
import time
import heapq

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import utils
# from spikingjelly.activation_based import functional
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
# from timm.loss import SoftTargetCrossEntropy
import data_loaders
from barlowtwins import barlow_twins # registers 'barlow_twins' into timm's model registry
# from transforms_factory import contrastive_learning_transforms
from loss import BarlowTwinsLoss

_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
writer = SummaryWriter("./")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (inferred from --dataset-name if not set)')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--timestep', default=16, type=int, help='timestep')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./save', help='path where to save')
    parser.add_argument('--dataset', default='/path/to/dvs128-gesture', help='root path of dataset')
    parser.add_argument('--dataset-name', default='dvs128-gesture',
                        choices=['dvs128-gesture', 'ncaltech101', 'asl-dvs'],
                        help='which neuromorphic dataset to use')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=10, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Act func
    parser.add_argument('--act-func', default='MixedLIF', type=str,
                        choices=['LIF', 'MixedLIF', 'LIFt'],
                        help='spiking activation function')

    # Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        choices=['resnet14', 'resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet50', 'resnet101'],
                        help='ResNet backbone variant for the SNN encoder')

    # Loss
    parser.add_argument('--cross-temporal', default=False, action='store_true',
                        help='Use cross-temporal Barlow Twins loss across timesteps')
    parser.add_argument('--simplified-loss', default=False, action='store_true',
                        help='Use simplified cross-temporal loss (first+last timestep only)')

    args = parser.parse_args()
    args.timestep = args.T  # data frames must match model timesteps
    return args

def get_resume_epoch(args):
    filename = os.path.basename(args.resume)

    if filename.startswith("checkpoint_epoch"):
        epoch_str = filename[len("checkpoint_epoch"):]
        try:
            start_epoch = int(epoch_str)
            print(f"Resuming from epoch: {start_epoch}")
        except ValueError:
            print("Error: Epoch number in filename is not valid.")
            start_epoch = 0
    else:
        print("Error: Filename format is incorrect.")
        start_epoch = 0

    args.start_epoch = start_epoch + 1


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for (image_i, image_j), _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image_i, image_j = image_i.to(device), image_j.to(device)

        if scaler is not None:
            with amp.autocast():
                _, _, output_i, output_j = model(image_i, image_j)
                loss = criterion(output_i, output_j)
        else:
            _, _, output_i, output_j = model(image_i, image_j)
            loss = criterion(output_i, output_j)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        # functional.reset_net(model)

        batch_size = image_i.shape[1]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg


def main(args):

    train_tb_writer = None

    #  initialization for distributed training
    utils.init_distributed_mode(args)
    args.device = f"cuda:{args.gpu}"
    device = torch.device(args.device)
    print(args)

    # Infer num_classes from dataset name if not explicitly set
    if args.num_classes is None:
        args.num_classes = data_loaders.DATASET_NUM_CLASSES[args.dataset_name]
    print(f"Dataset: {args.dataset_name}, num_classes: {args.num_classes}")

    name = args.dataset_name
    output_dir = os.path.join(args.output_dir, f'{name}_T{args.T}')

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    train_dataset, _ = data_loaders.build_dataset(args.dataset_name, args.dataset, T=args.T)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=(train_sampler is None),
                                              num_workers=args.workers,
                                              sampler=train_sampler,
                                              drop_last=True,
                                              pin_memory=True)

    model = barlow_twins(
        act_func=args.act_func,
        name=args.backbone,
        timestep=args.timestep,
        num_classes=args.num_classes,
        sync_norm=True,
    )
    print(f"Backbone: {args.backbone}")
    print(model)
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # criterion_train = LabelSmoothingCrossEntropy()
    criterion_train = BarlowTwinsLoss(args.device, args.world_size,
                                      cross_temporal=args.cross_temporal,
                                      simplified_loss=args.simplified_loss,
                                      proj_dim=1024)
    criterion_train.to(device)

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        get_resume_epoch(args)

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '/logs/train', purge_step=purge_step_train)
        with open(output_dir + '/logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}')

    print("Start training")
    start_time = time.time()
    best_checkpoints = []
    for epoch in range(args.start_epoch, num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)

        lr_scheduler.step(epoch + 1)

        if output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }

            if len(best_checkpoints) < 5:
                heapq.heappush(best_checkpoints, (-train_loss, epoch))
                utils.save_on_master(
                    model_without_ddp.state_dict(),
                    os.path.join(output_dir, f'checkpoint_epoch{epoch}')
                )
            else:
                if -train_loss > best_checkpoints[0][0]:
                    removed_loss, removed_epoch = heapq.heappop(best_checkpoints)
                    removed_file = os.path.join(output_dir,
                                                f'checkpoint_epoch{removed_epoch}')
                    if os.path.exists(removed_file) and utils.is_main_process():
                        os.remove(removed_file)

                    heapq.heappush(best_checkpoints, (-train_loss, epoch))

                    utils.save_on_master(
                        model_without_ddp.state_dict(),
                        os.path.join(output_dir, f'checkpoint_epoch{epoch}')
                    )

        # print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'train_loss', train_loss)


if __name__ == "__main__":
    args = parse_args()
    main(args)
