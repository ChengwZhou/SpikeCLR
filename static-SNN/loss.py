import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from modules.gather import GatherLayer


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, world_size, lambda_param=5e-3, proj_dim=512):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.world_size = world_size
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)
        if proj_dim > 4048:
            self.scale_loss = 1
        else:
            self.scale_loss = 1

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # Method 1
        # normalize repr. along the batch dimension
        # if self.world_size > 1:
        #     z_a = torch.cat(GatherLayer.apply(z_a), dim=0)  # [B, C=D]
        #     z_b = torch.cat(GatherLayer.apply(z_b), dim=0)

        # z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6)  # NxD
        # z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6)  # NxD

        # N, D = z_a_norm.shape
        #
        # # cross-correlation matrix
        # c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # # loss
        # c_diff = (c - torch.eye(D, device=self.device)).pow(2)  # DxD
        # # multiply off-diagonal elems of c_diff by lambda
        # c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        # loss = c_diff.sum()

        # Method 2, empirical cross-correlation matrix
        N, _ = z_a.shape
        c = self.bn(z_a).T @ self.bn(z_b)

        # sum the cross-correlation matrix between all gpus
        c.div_(N*self.world_size)
        if self.world_size > 1:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss


def _gather_temporal(z, world_size):
    """
    Gather a (T, B, D) tensor from all ranks into (T, world_size*B, D).
    Uses a single all_gather call by flattening the temporal dim into feature dim.
    Preserves gradients via GatherLayer for the local portion.
    """
    T, B, D = z.shape
    # (T, B, D) -> (B, T*D) so GatherLayer gathers along the batch axis
    z_flat = z.permute(1, 0, 2).contiguous().view(B, T * D)
    z_flat = torch.cat(GatherLayer.apply(z_flat), dim=0)  # (world_size*B, T*D)
    B_all = z_flat.shape[0]
    return z_flat.view(B_all, T, D).permute(1, 0, 2).contiguous()  # (T, world_size*B, D)


class SimCLRLoss(torch.nn.Module):
    """NT-Xent loss for SimCLR."""

    def __init__(self, device, world_size, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        if self.world_size > 1:
            z_a = torch.cat(GatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(GatherLayer.apply(z_b), dim=0)

        N = z_a.shape[0]
        z = torch.cat([z_a, z_b], dim=0)  # [2N, D]
        z = F.normalize(z, dim=1)

        # Similarity matrix [2N, 2N]
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(self.device)

        loss = F.cross_entropy(sim, labels)
        return loss


class SimCLRTemporalLoss(torch.nn.Module):
    """
    Temporal NT-Xent loss for spiking SimCLR.

    Input z_a / z_b have shape (T, B, D) — the temporal output of a spiking projector.

    cross_temporal=False  Flatten (T,B,D) → (T*B,D) and run a single NT-Xent.
                          Positive for (t,b) in z_a is (t,b) in z_b.
    cross_temporal=True   Build 2T views and compute pairwise NT-Xent between every
                          pair (i≠j), treating same-index sample b as positive.
    simplified_loss=True  (requires cross_temporal=True) Use only the first and last
                          timestep per view → 4 views, 12 pairs total.
    """

    def __init__(self, device, world_size, temperature=0.5,
                 cross_temporal=False, simplified_loss=False):
        super(SimCLRTemporalLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.world_size = world_size
        self.cross_temporal = cross_temporal
        self.simplified_loss = simplified_loss

    def _nt_xent(self, z_a, z_b):
        """NT-Xent on a single pre-gathered (N, D) pair."""
        N = z_a.shape[0]
        z = F.normalize(torch.cat([z_a, z_b], dim=0), dim=1)  # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(self.device)
        return F.cross_entropy(sim, labels)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        if not self.cross_temporal:
            # flatten temporal → batch, gather, run NT-Xent once
            z_a = z_a.flatten(0, 1)
            z_b = z_b.flatten(0, 1)
            if self.world_size > 1:
                z_a = torch.cat(GatherLayer.apply(z_a), dim=0)
                z_b = torch.cat(GatherLayer.apply(z_b), dim=0)
            return self._nt_xent(z_a, z_b)

        # cross_temporal: pre-gather once then compute all pairs locally
        T, B, D = z_a.shape
        if self.world_size > 1:
            z_a = _gather_temporal(z_a, self.world_size)  # (T, world_size*B, D)
            z_b = _gather_temporal(z_b, self.world_size)

        if self.simplified_loss:
            z = torch.stack([z_a[0], z_a[T - 1], z_b[0], z_b[T - 1]])  # (4, B, D)
            iteration = 2
            t_scale = 12  # 4*3 ordered pairs
        else:
            z = torch.cat([z_a, z_b], dim=0)   # (2T, B, D)
            iteration = T
            t_scale = 2 * T * (2 * T - 1)

        loss = 0
        for i in range(2 * iteration):
            for j in range(2 * iteration):
                if i == j:
                    continue
                loss += self._nt_xent(z[i], z[j])
        return loss / t_scale


class VICRegLoss(torch.nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization (Bardes et al., 2022).

    Three terms:
      - invariance (sim_coeff):  MSE between the two projected views.
      - variance   (std_coeff):  hinge loss keeping per-dimension std above 1.
      - covariance (cov_coeff):  off-diagonal covariance penalty (no collapse).
    """

    def __init__(self, device, world_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.device = device
        self.world_size = world_size

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # spiking networks produce (T, B, D); reduce temporal dim by averaging
        if z_a.dim() == 3:
            z_a = z_a.mean(0)
            z_b = z_b.mean(0)

        if self.world_size > 1:
            z_a = torch.cat(GatherLayer.apply(z_a), dim=0)
            z_b = torch.cat(GatherLayer.apply(z_b), dim=0)

        N, D = z_a.shape

        # invariance: MSE between the two projections
        sim_loss = F.mse_loss(z_a, z_b)

        # variance: hinge on per-dimension std to keep representations spread out
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # covariance: penalise off-diagonal entries of the covariance matrices
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_a = (z_a.T @ z_a) / (N - 1)
        cov_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum() / D + off_diagonal(cov_b).pow_(2).sum() / D

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss


class VICRegTemporalLoss(torch.nn.Module):
    """
    Temporal VICReg for spiking networks.

    Input z_a / z_b have shape (T, B, D).

    cross_temporal=False  Flatten (T,B,D) → (T*B,D), gather, compute one VICReg.
    cross_temporal=True   Build 2T views, compute pairwise VICReg for every i≠j pair.
    simplified_loss=True  (requires cross_temporal=True) Only first/last timestep
                          per view → 4 views, 12 ordered pairs.
    """

    def __init__(self, device, world_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0,
                 cross_temporal=False, simplified_loss=False):
        super(VICRegTemporalLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.device = device
        self.world_size = world_size
        self.cross_temporal = cross_temporal
        self.simplified_loss = simplified_loss

    def _vicreg(self, z_a, z_b):
        """VICReg on a single pre-gathered (N, D) pair."""
        N, D = z_a.shape
        sim_loss = F.mse_loss(z_a, z_b)
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_a = (z_a.T @ z_a) / (N - 1)
        cov_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum() / D + off_diagonal(cov_b).pow_(2).sum() / D
        return self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        if not self.cross_temporal:
            z_a = z_a.flatten(0, 1)
            z_b = z_b.flatten(0, 1)
            if self.world_size > 1:
                z_a = torch.cat(GatherLayer.apply(z_a), dim=0)
                z_b = torch.cat(GatherLayer.apply(z_b), dim=0)
            return self._vicreg(z_a, z_b)

        T, B, D = z_a.shape
        if self.world_size > 1:
            z_a = _gather_temporal(z_a, self.world_size)
            z_b = _gather_temporal(z_b, self.world_size)

        if self.simplified_loss:
            z = torch.stack([z_a[0], z_a[T - 1], z_b[0], z_b[T - 1]])
            iteration = 2
            t_scale = 12
        else:
            z = torch.cat([z_a, z_b], dim=0)
            iteration = T
            t_scale = 2 * T * (2 * T - 1)

        loss = 0
        for i in range(2 * iteration):
            for j in range(2 * iteration):
                if i == j:
                    continue
                loss += self._vicreg(z[i], z[j])
        return loss / t_scale


class BarlowTwinsTemporalLoss(torch.nn.Module):

    def __init__(self, device, world_size, cross_temporal=False, simplified_loss=False, lambda_param=5e-3, proj_dim=512):
        super(BarlowTwinsTemporalLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.world_size = world_size
        self.cross_temporal = cross_temporal
        self.simplified_loss = simplified_loss
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)
        if proj_dim > 4048:
            self.scale_loss = 1
        else:
            self.scale_loss = 1

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
                t_scale = 12
            else:
                z = torch.cat((z_a, z_b), dim=0)
                iteration = T
                t_scale = 2*T*(2*T-1)

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

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
                    off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
                    loss += on_diag + self.lambda_param * off_diag

            loss /= t_scale  # [2T*(2T-1)/2]*2
        return loss
