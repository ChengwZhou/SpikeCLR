import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist

# from spikingjelly.activation_based.neuron import LIFNode
# from spikingjelly.activation_based.surrogate import PiecewiseLeakyReLU, Sigmoid


a = 1.0
momentum_SGD = 0.9


class LIF(nn.Module):
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,  detach_reset: bool = False,
                 backend='torch'):
        super(LIF, self).__init__()
        self.timestep = timestep
        self.tau = tau
        self.v_threshold = v_threshold

    def forward(self, x):
        T = self.timestep
        B = int(x.shape[0] // self.timestep)
        u = torch.zeros((B,)+x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(T):
            u = (1/self.tau) * u * (1 - spikefunc(u, self.v_threshold).detach()) + x[B*t:B*(t+1), ...]
            o[B*t:B*(t+1), ...] = spikefunc(u, self.v_threshold)  # Equivalent to union of all spikes
        return o


class MixedLIF(nn.Module):
    """
    Activative function is different for two trail of contrastive learning.
    trail-1: original LIf
    trail-2: Relu-like continuous func.
    """
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,  detach_reset: bool = False,
                 backend='torch'):
        super(MixedLIF, self).__init__()
        self.timestep = timestep
        self.tau = tau
        self.v_threshold = v_threshold

    def forward(self, x):
        T = self.timestep
        B = int(x.shape[0] // self.timestep)
        bs = B//2
        u = torch.zeros((bs,)+x.shape[1:], device=x.device)
        u2 = torch.zeros((bs,)+x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(T):
            u = (1/self.tau) * u * (1 - spikefunc(u, self.v_threshold).detach()) + x[B*t:B*(t+1)-bs, ...]
            u2 = (1/self.tau) * u2 * (1 - spikefunc(u2, self.v_threshold).detach()) + x[B*t+bs:B*(t+1), ...]
            o[B*t:B*(t+1)-bs, ...] = spikefunc(u, self.v_threshold)  # Equivalent to union of all spikes
            o[B*t+bs:B*(t+1), ...] = torch.clamp(u2-self.v_threshold+0.5, min=0, max=1.0)
        return o


class LIFt(nn.Module):
    """
    Activative function for two trail of contrastive learning.
    trail-1: LIf
    trail-2: LIf.
    """
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,  detach_reset: bool = False,
                 backend='torch'):
        super(LIFt, self).__init__()
        self.timestep = timestep
        self.tau = tau
        self.v_threshold = v_threshold

    def forward(self, x):
        T = self.timestep
        B = int(x.shape[0] // self.timestep)
        bs = B//2
        u = torch.zeros((bs,)+x.shape[1:], device=x.device)
        u2 = torch.zeros((bs,)+x.shape[1:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)
        for t in range(T):
            u = (1/self.tau) * u * (1 - spikefunc(u, self.v_threshold).detach()) + x[B*t:B*(t+1)-bs, ...]
            u2 = (1/self.tau) * u2 * (1 - spikefunc(u2, self.v_threshold).detach()) + x[B*t+bs:B*(t+1), ...]
            o[B*t:B*(t+1)-bs, ...] = spikefunc(u, self.v_threshold)  # Equivalent to union of all spikes
            o[B*t+bs:B*(t+1), ...] = spikefunc(u2, self.v_threshold)
        return o


# class LIF(nn.Module):
#     def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,  detach_reset: bool = False,
#                  backend='torch'):
#         super(LIF, self).__init__()
#         self.timestep = timestep
#         self.lif = LIFNode(step_mode='m', tau=1 / (1 - 1 / tau), v_threshold=v_threshold,
#                            # surrogate_function=PiecewiseLeakyReLU(w=1.0, c=0.0, spiking=True),
#                            decay_input=False,
#                            detach_reset=detach_reset,
#                            backend=backend)
#
#     def forward(self, x):
#         T = self.timestep
#         B = int(x.shape[0] // self.timestep)
#
#         x = x.reshape(T, B, *x.shape[1:]).contiguous()
#         x = self.lif(x)
#         x = x.flatten(0, 1)
#         return x
#
#
# class LIFt(nn.Module):
#     """
#     Activative function for two trail of contrastive learning.
#     trail-1: LIf
#     trail-2: LIf.
#     """
#
#     def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,
#                  detach_reset: bool = False, backend='torch'):
#         super(LIFt, self).__init__()
#         self.timestep = timestep
#         self.lif_1 = LIFNode(step_mode='m', tau=1 / (1 - 1 / tau), v_threshold=v_threshold,
#                            # surrogate_function=PiecewiseLeakyReLU(w=1.0, c=0.0, spiking=True),
#                            decay_input=False,
#                            detach_reset=detach_reset,
#                            backend=backend)
#         self.lif_2 = LIFNode(step_mode='m', tau=1 / (1 - 1 / tau), v_threshold=v_threshold,
#                                # surrogate_function=PiecewiseLeakyReLU(w=1.0, c=0.0, spiking=True),
#                                decay_input=False,
#                                detach_reset=detach_reset,
#                                backend=backend)
#
#     def forward(self, x):
#         T = self.timestep
#         B = int(x.shape[0] // self.timestep)
#         bs = B // 2
#
#         x = x.reshape(T, B, *x.shape[1:]).contiguous()
#         x = torch.cat((self.lif_1(x[:, :bs, ...]), self.lif_2(x[:, bs:, ...])), dim=1)
#         x = x.flatten(0, 1)
#         return x
#
#
# class MixedLIF(nn.Module):
#     """
#     Activative function of two Paths in the way of contrastive learing.
#     Path-1: original LIf
#     Path-2: Relu-like continuous func.
#     Vth and tau are set same as class spikingjelly.activation_based.neuron.LIFNode
#     """
#     def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, timestep=4, v_reset: float = 0.0,
#                  detach_reset: bool = False, backend='torch'):
#         super(MixedLIF, self).__init__()
#         self.timestep = timestep
#         self.lif = LIFNode(step_mode='m', tau=1/(1-1/tau), v_threshold=v_threshold,
#                            # surrogate_function=PiecewiseLeakyReLU(w=1.0, c=0.0, spiking=True),
#                            decay_input=False,
#                            detach_reset=detach_reset,
#                            backend=backend)
#         self.ClampLu = LIFNode(step_mode='m', tau=1/(1-1/tau), v_threshold=v_threshold,
#                                surrogate_function=Sigmoid(alpha=4.0, spiking=False),
#                                decay_input=False,
#                                detach_reset=detach_reset,
#                                backend=backend)
#
#     def forward(self, x):
#         T = self.timestep
#         B = int(x.shape[0] // self.timestep)
#         bs = B // 2
#
#         x = x.reshape(T, B, *x.shape[1:]).contiguous()
#         x = torch.cat((self.lif(x[:, :bs, ...]), self.ClampLu(x[:, bs:, ...])), dim=1)
#         x = x.flatten(0, 1)
#         return x


class tdBatchNorm(nn.Module):
    """
    Only use the attribution of bn, and hand made the tdBN in the forward function
    However, there are two trails which are independent to each other. And we use
    sync_norm to mark the choice of different norm method.
    """
    def __init__(self, bn, sync_norm=True, alpha=1):
        super(tdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha
        self.sync_norm = sync_norm
        if not sync_norm:
            self.bn2 = bn

    def forward(self, x):
        exponential_average_factor = 0.0

        # calculate exponential average factor
        if self.training and self.bn.track_running_stats:  # track_running_stats default: True
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum

        #Use Synchronous Normalization
        if self.sync_norm:
            if self.training:
                mean = x.mean([0, 2, 3], keepdim=True)  # [batch, channels, height, width]
                var = x.var([0, 2, 3], keepdim=True, unbiased=False)
                n = x.numel() / x.size(1)
                with torch.no_grad():
                    self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0]\
                                           + (1 - exponential_average_factor) * self.bn.running_mean
                    self.bn.running_var = exponential_average_factor * var[0, :, 0, 0] * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.bn.running_var
            else:
                mean = self.bn.running_mean[None, :, None, None]
                var = self.bn.running_var[None, :, None, None]

            x = self.alpha * 1 * (x - mean) / (torch.sqrt(var) + self.bn.eps)
            if self.bn.affine:
                x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]

        # not sync_norm, note: there are two bns
        else:
            x_index = int(x.shape[0] /2)
            x_1 = x[:x_index, ...]
            x_2 = x[x_index:, ...]
            if self.training:
                mean_1 = x_1.mean([0, 2, 3], keepdim=True)  # [batch, channels, height, width]
                mean_2 = x_2.mean([0, 2, 3], keepdim=True)
                var_1 = x_1.var([0, 2, 3], keepdim=True, unbiased=False)
                var_2 = x_2.var([0, 2, 3], keepdim=True, unbiased=False)
                n = x_1.numel() / x_1.size(1)
                with torch.no_grad():
                    self.bn.running_mean = exponential_average_factor * mean_1[0, :, 0, 0] \
                                           + (1 - exponential_average_factor) * self.bn.running_mean
                    self.bn2.running_mean = exponential_average_factor * mean_2[0, :, 0, 0] \
                                           + (1 - exponential_average_factor) * self.bn2.running_mean
                    self.bn.running_var = exponential_average_factor * var_1[0, :, 0, 0] * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.bn.running_var
                    self.bn2.running_var = exponential_average_factor * var_2[0, :, 0, 0] * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.bn2.running_var
            else:
                mean_1 = self.bn.running_mean[None, :, None, None]
                mean_2 = self.bn2.running_mean[None, :, None, None]
                var_1 = self.bn.running_var[None, :, None, None]
                var_2 = self.bn2.running_var[None, :, None, None]

            x1_updated = self.alpha * 1 * (x_1 - mean_1) / (torch.sqrt(var_1) + self.bn.eps)
            x2_updated = self.alpha * 1 * (x_2 - mean_2) / (torch.sqrt(var_2) + self.bn2.eps)

            if self.bn.affine:
                x1_updated = x1_updated * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
            if self.bn2.affine:
                x2_updated = x2_updated * self.bn2.weight[None, :, None, None] + self.bn2.bias[None, :, None, None]
            x = torch.cat([x1_updated, x2_updated], dim=0)
        return x


class SynctdBatchNorm(nn.Module):
    def __init__(self, bn, sync_norm=True, alpha=1):
        super(SynctdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha
        self.sync_norm = sync_norm
        if not sync_norm:
            self.bn2 = bn

    def forward(self, x):
        exponential_average_factor = 0.0

        # Use Synchronous Normalization
        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:
                    exponential_average_factor = self.bn.momentum

        if self.sync_norm:
            if self.training:
                # Compute mean and var
                mean = x.mean([0, 2, 3], keepdim=True)
                var = x.var([0, 2, 3], keepdim=True, unbiased=False)

                # Sync mean and var across all processes
                dist.all_reduce(mean)
                dist.all_reduce(var)
                mean /= dist.get_world_size()
                var /= dist.get_world_size()

                n = x.numel() / x.size(1)
                with torch.no_grad():
                    self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0] + \
                                           (1 - exponential_average_factor) * self.bn.running_mean
                    self.bn.running_var = exponential_average_factor * var[0, :, 0, 0] * n / (n - 1) + \
                                          (1 - exponential_average_factor) * self.bn.running_var
                # Broadcast running stats to ensure all processes have same values
                dist.broadcast(self.bn.running_mean, src=0)
                dist.broadcast(self.bn.running_var, src=0)
            else:
                mean = self.bn.running_mean[None, :, None, None]
                var = self.bn.running_var[None, :, None, None]

            x = self.alpha * (x - mean) / (torch.sqrt(var) + self.bn.eps)
            if self.bn.affine:
                x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]

        else:
            x_index = int(x.shape[0] / 2)
            x_1 = x[:x_index, ...]
            x_2 = x[x_index:, ...]
            if self.training:
                mean_1 = x_1.mean([0, 2, 3], keepdim=True)  # [batch, channels, height, width]
                mean_2 = x_2.mean([0, 2, 3], keepdim=True)
                var_1 = x_1.var([0, 2, 3], keepdim=True, unbiased=False)
                var_2 = x_2.var([0, 2, 3], keepdim=True, unbiased=False)

                # Sync mean and var across all processes
                dist.all_reduce(mean_1)
                dist.all_reduce(mean_2)
                dist.all_reduce(var_1)
                dist.all_reduce(var_2)
                mean_1 /= dist.get_world_size()
                mean_2 /= dist.get_world_size()
                var_1 /= dist.get_world_size()
                var_2 /= dist.get_world_size()

                n = x_1.numel() / x_1.size(1)
                with torch.no_grad():
                    self.bn.running_mean = exponential_average_factor * mean_1[0, :, 0, 0] \
                                           + (1 - exponential_average_factor) * self.bn.running_mean
                    self.bn2.running_mean = exponential_average_factor * mean_2[0, :, 0, 0] \
                                            + (1 - exponential_average_factor) * self.bn2.running_mean
                    self.bn.running_var = exponential_average_factor * var_1[0, :, 0, 0] * n / (n - 1) \
                                          + (1 - exponential_average_factor) * self.bn.running_var
                    self.bn2.running_var = exponential_average_factor * var_2[0, :, 0, 0] * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.bn2.running_var
                # Broadcast running stats to ensure all processes have same values
                dist.broadcast(self.bn.running_mean, src=0)
                dist.broadcast(self.bn2.running_mean, src=0)
                dist.broadcast(self.bn.running_var, src=0)
                dist.broadcast(self.bn2.running_var, src=0)
            else:
                mean_1 = self.bn.running_mean[None, :, None, None]
                mean_2 = self.bn2.running_mean[None, :, None, None]
                var_1 = self.bn.running_var[None, :, None, None]
                var_2 = self.bn2.running_var[None, :, None, None]

            x1_updated = self.alpha * 1 * (x_1 - mean_1) / (torch.sqrt(var_1) + self.bn.eps)
            x2_updated = self.alpha * 1 * (x_2 - mean_2) / (torch.sqrt(var_2) + self.bn2.eps)

            if self.bn.affine:
                x1_updated = x1_updated * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
            if self.bn2.affine:
                x2_updated = x2_updated * self.bn2.weight[None, :, None, None] + self.bn2.bias[None, :, None, None]
            x = torch.cat([x1_updated, x2_updated], dim=0)
        return x


    @classmethod
    def convert_sync_tdBatchNorm(cls, module, process_group=None):
        """
        recursively convert tdBatchNorm to sync_tdBatchNorm
        """
        module_output = module
        if isinstance(module, tdBatchNorm):
            module_output = SynctdBatchNorm(nn.BatchNorm2d(module.bn.num_features,
                                                           module.bn.eps,
                                                           module.bn.momentum,
                                                           module.bn.affine,
                                                           module.bn.track_running_stats,
                                                           process_group), module.sync_norm)
            if module.sync_norm:
                if module.bn.affine:
                    with torch.no_grad():
                        module_output.bn.weight = module.bn.weight
                        module_output.bn.bias = module.bn.bias
                module_output.bn.running_mean = module.bn.running_mean
                module_output.bn.running_var = module.bn.running_var
                module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
                if hasattr(module, "qconfig"):
                    module_output.qconfig = module.qconfig
            else:
                if module.bn.affine:
                    with torch.no_grad():
                        module_output.bn.weight = module.bn.weight
                        module_output.bn.bias = module.bn.bias
                if module.bn2.affine:
                    with torch.no_grad():
                        module_output.bn2.weight = module.bn2.weight
                        module_output.bn2.bias = module.bn2.bias

                module_output.bn.running_mean = module.bn.running_mean
                module_output.bn2.running_mean = module.bn2.running_mean
                module_output.bn.running_var = module.bn.running_var
                module_output.bn2.running_var = module.bn2.running_var
                module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
                module_output.bn2.num_batches_tracked = module.bn2.num_batches_tracked
                if hasattr(module, "qconfig"):
                    module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_tdBatchNorm(child, process_group))
        del module
        return module_output


    @classmethod
    def convert_sync_back_tdBatchNorm(cls, module, process_group=None):
        """
        recursively convert sync_tdBatchNorm back to tdBatchNorm
        """
        module_output = module
        if isinstance(module, SynctdBatchNorm):
            module_output = tdBatchNorm(nn.BatchNorm2d(module.bn.num_features,
                                                       module.bn.eps,
                                                       module.bn.momentum,
                                                       module.bn.affine,
                                                       module.bn.track_running_stats), module.sync_norm)
            if module.sync_norm:
                if module.bn.affine:
                    with torch.no_grad():
                        module_output.bn.weight = module.bn.weight
                        module_output.bn.bias = module.bn.bias
                module_output.bn.running_mean = module.bn.running_mean
                module_output.bn.running_var = module.bn.running_var
                module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
                if hasattr(module, "qconfig"):
                    module_output.qconfig = module.qconfig
            else:
                if module.bn.affine:
                    with torch.no_grad():
                        module_output.bn.weight = module.bn.weight
                        module_output.bn.bias = module.bn.bias
                if module.bn2.affine:
                    with torch.no_grad():
                        module_output.bn2.weight = module.bn2.weight
                        module_output.bn2.bias = module.bn2.bias

                module_output.bn.running_mean = module.bn.running_mean
                module_output.bn2.running_mean = module.bn2.running_mean
                module_output.bn.running_var = module.bn.running_var
                module_output.bn2.running_var = module.bn2.running_var
                module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
                module_output.bn2.num_batches_tracked = module.bn2.num_batches_tracked
                if hasattr(module, "qconfig"):
                    module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_back_tdBatchNorm(child, process_group))
        del module
        return module_output


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, v_threshold):
        ctx.save_for_backward(input)
        ctx.v_threshold = v_threshold

        output = torch.gt(input, v_threshold)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        v_threshold = ctx.v_threshold
        grad_input = grad_output.clone()
        hu = (abs(input - v_threshold) < (a/2)) / a
        return grad_input * hu, None

spikefunc = SpikeFunction.apply


