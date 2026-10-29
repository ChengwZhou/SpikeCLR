import torch
import torch.nn as nn
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, world_size, cross_temporal=False, simplified_loss=False, lambda_param=5e-3,
                 proj_dim=1024):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.world_size = world_size
        self.cross_temporal = cross_temporal
        self.simplified_loss = simplified_loss
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        if not self.cross_temporal:
            z_a = z_a.flatten(0, 1)
            z_b = z_b.flatten(0, 1)

            N, _ = z_a.shape
            c = self.bn(z_a).T @ self.bn(z_b)

            # sum the cross-correlation matrix between all gpus
            c.div_(N * self.world_size)
            if self.world_size > 1:
                torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambda_param * off_diag
        else:
            # compute cross temporal loss
            T, B, D = z_a.shape
            # only compute the loss composed of the first and the last time-step.
            if self.simplified_loss:
                # CREATE A NEW TENSOR
                z = torch.zeros((4, B, D), device=z_a.device)
                z[0, ...] = z_a[0, ...]
                z[1, ...] = z_a[T - 1, ...]
                z[2, ...] = z_b[0, ...]
                z[3, ...] = z_b[T - 1, ...]
                iteration = 2
            else:
                z = torch.cat((z_a, z_b), dim=0)
                iteration = T

            # loss initialization
            loss = 0
            for i in range(2 * iteration):
                for j in range(2 * iteration):
                    if i == j:
                        continue
                    z_a = z[i, ...]  # [B,D]
                    z_b = z[j, ...]
                    N = B

                    c = self.bn(z_a).T @ self.bn(z_b)

                    # sum the cross-correlation matrix between all gpus
                    c.div_(N * self.world_size)
                    if self.world_size > 1:
                        torch.distributed.all_reduce(c)

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).pow_(2).sum()
                    loss += on_diag + self.lambda_param * off_diag

            loss /= iteration  # [2T*(2T-1)/2]*2
        return loss

