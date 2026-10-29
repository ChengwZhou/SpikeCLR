import os
import math
import torch
import heapq
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules import SynctdBatchNorm, BatchNormConverter


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super(LARS, self).__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_steps_per_epoch, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.base_lrs = [args.batch_size / 256 for group in optimizer.param_groups]
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs * self.num_steps_per_epoch:
            lr_scale = self.last_epoch / (self.warmup_epochs * self.num_steps_per_epoch)
        else:
            total_steps = self.total_epochs * self.num_steps_per_epoch
            warmup_steps = self.warmup_epochs * self.num_steps_per_epoch
            completed_steps = self.last_epoch - warmup_steps
            T_max = total_steps - warmup_steps
            lr_scale = 0.5 * (1 + math.cos(math.pi * completed_steps / T_max))

        return [base_lr * lr_scale for base_lr in self.base_lrs]


def load_optimizer(args, parameters, len_loader):
    if args.optimizer == "Adam":
        optimizer = torch.optim.AdamW(parameters, args.lr, weight_decay=1*1e-6, eps=1e-8)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    elif args.optimizer == "LARS":
        optimizer = LARS(parameters, lr=args.lr, momentum=0.9, weight_decay=1*1e-6)
        scheduler = None
        # scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=args.epochs,
        #                                     num_steps_per_epoch=len_loader)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1*1e-6)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise NotImplementedError("Only Adam, LARS, and SGD optimizers are implemented.")

    return optimizer, scheduler


def save_model(args, model, optimizer, scaler):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    save_dict = {
        'model_state_dict': None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': args.current_epoch
    }

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        save_dict['model_state_dict'] = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        """
        Convert Sync bn to bn before save model.
        """
        # if args.spiking:
        #     model = SynctdBatchNorm.convert_sync_back_tdBatchNorm(model)
        # else:
        #     model = BatchNormConverter.convert_sync_bn_to_bn(model)
        save_dict['model_state_dict'] = model.module.state_dict()
    else:
        save_dict['model_state_dict'] = model.state_dict()

    torch.save(save_dict, out)


def save_model_top_k(args, model, optimizer, scaler, loss_epoch, top_models):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    out = os.path.join(args.model_path, f"checkpoint_epoch_{args.current_epoch}.tar")

    if len(top_models) < 5:
        heapq.heappush(top_models, (-loss_epoch, args.current_epoch, out))
        save_model_to_disk(model, optimizer, scaler, args, out)
    else:
        current_max_loss = -top_models[0][0]  # loss have to be negative
        if loss_epoch < current_max_loss:
            _, _, old_model_path = heapq.heappop(top_models)  # pop the top
            if os.path.exists(old_model_path):
                os.remove(old_model_path)  # remove old version
            heapq.heappush(top_models, (-loss_epoch, args.current_epoch, out))
            save_model_to_disk(model, optimizer, scaler, args, out)

def save_model_to_disk(model, optimizer, scaler, args, out):
    """
    save model
    """
    save_dict = {
        'model_state_dict': None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': args.current_epoch
    }

    if isinstance(model, torch.nn.DataParallel):
        save_dict['model_state_dict'] = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        save_dict['model_state_dict'] = model.module.state_dict()
    else:
        save_dict['model_state_dict'] = model.state_dict()

    torch.save(save_dict, out)
