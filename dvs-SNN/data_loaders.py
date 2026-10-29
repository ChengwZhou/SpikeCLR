import math
import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import warnings
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets.asl_dvs import ASLDVS

warnings.filterwarnings('ignore')


class DVSBarlowTwinsTransform:
    """
    Barlow Twins augmentation for DVS event frames of shape [T, C, H, W].
    Applies independent spatial, temporal, and polarity augmentations to
    generate two correlated views for contrastive learning.
    """

    def __call__(self, x: torch.Tensor):
        return self._augment(x.clone()), self._augment(x.clone())

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = self._random_flip(x)
        if random.random() < 0.5:
            x = self._random_vertical_flip(x)
        if random.random() < 0.5:
            x = self._random_rotation(x, max_angle=10.0)
        x = self._random_crop_pad(x, padding=8)
        if random.random() < 0.2:
            x = self._time_reversal(x)
        if random.random() < 0.5:
            x = self._frame_dropout(x, drop_prob=0.25)
        if random.random() < 0.8:
            x = self._random_polarity_jitter(x)
        if random.random() < 0.5:
            x = self._random_event_density_noise(x, std=0.03)
        x = self._random_gaussian_blur(x)
        return x

    # ── Spatial ──────────────────────────────────────────────

    def _random_flip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[3])

    def _random_vertical_flip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[2])

    def _random_rotation(self, x: torch.Tensor, max_angle: float = 10.0) -> torch.Tensor:
        """Small-angle random rotation (±max_angle degrees).
        The same rotation angle is applied to all timesteps to preserve temporal consistency.
        fill=0.0 corresponds to no-event regions.
        """
        angle = random.uniform(-max_angle, max_angle)
        T = x.shape[0]
        frames = [TF.rotate(x[t], angle=angle,
                            interpolation=TF.InterpolationMode.BILINEAR,
                            fill=0.0)
                  for t in range(T)]
        return torch.stack(frames)

    def _random_gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial Gaussian blur to simulate DVS pixel crosstalk and spatial uncertainty.
        The same kernel is applied to all timesteps to preserve temporal consistency.
        """
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        T = x.shape[0]
        frames = [TF.gaussian_blur(x[t], kernel_size=kernel_size, sigma=sigma)
                  for t in range(T)]
        return torch.stack(frames)

    def _random_crop_pad(self, x: torch.Tensor, padding: int = 8) -> torch.Tensor:
        """Pad then random crop as a spatial translation augmentation."""
        T, C, H, W = x.shape
        x = F.pad(x, [padding] * 4, mode='constant', value=0)
        top = random.randint(0, 2 * padding)
        left = random.randint(0, 2 * padding)
        return x[:, :, top:top + H, left:left + W]

    # ── Temporal ─────────────────────────────────────────────

    def _time_reversal(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the temporal axis and swap polarity channels.
        DVS event polarity is tied to motion direction, so flipping time requires
        swapping C=0 and C=1 to maintain physical consistency.
        """
        x = torch.flip(x, dims=[0])
        if x.shape[1] == 2:
            x = x[:, [1, 0], :, :]
        return x

    def _temporal_jitter(self, x: torch.Tensor, max_jitter: int = 3) -> torch.Tensor:
        """Random temporal crop: shift the start frame and zero-pad the tail."""
        T = x.shape[0]
        if max_jitter <= 0:
            return x
        start = random.randint(0, max_jitter)
        sliced = x[start:]
        pad_len = T - sliced.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, *x.shape[1:], dtype=x.dtype, device=x.device)
            sliced = torch.cat([sliced, pad], dim=0)
        return sliced

    def _frame_dropout(self, x: torch.Tensor, drop_prob: float = 0.4) -> torch.Tensor:
        """Vectorized frame dropout: generate a binary mask over T and apply in one pass."""
        T = x.shape[0]
        mask = (torch.rand(T) > drop_prob).float()
        return x * mask.view(T, 1, 1, 1).to(x.device)

    # ── Intensity / Polarity ──────────────────────────────────

    def _random_polarity_jitter(self, x: torch.Tensor,
                                scale_range=(0.7, 1.3)) -> torch.Tensor:
        """Multiplicative polarity scaling to simulate DVS pixel threshold variation."""
        C = x.shape[1]
        for c in range(C):
            scale = random.uniform(*scale_range)
            x[:, c] = x[:, c] * scale
        return x

    def _random_event_density_noise(self, x: torch.Tensor,
                                    std: float = 0.03) -> torch.Tensor:
        """Simulate DVS background activity noise by adding low-intensity Gaussian noise."""
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0.0)

    # ── Spatial masking ──────────────────────────────────────

    def _random_cutout(self, x: torch.Tensor,
                       num_holes: int = 2,
                       hole_size: int = 24) -> torch.Tensor:
        """Multi-hole cutout to force reliance on global structure rather than local texture."""
        T, C, H, W = x.shape
        for _ in range(num_holes):
            hs = hole_size + random.randint(-4, 4)
            hs = max(8, min(hs, min(H, W) // 2))
            top = random.randint(0, H - hs)
            left = random.randint(0, W - hs)
            x[:, :, top:top + hs, left:left + hs] = 0.0
        return x


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :param origin_dataset: the origin dataset
    :param num_classes: total classes number
    :param random_split: If False, the front ratio of samples in each class will be train set.
                         If True, samples are shuffled randomly (controlled by numpy.random.seed).
    :return: a tuple (train_set, test_set)
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)

    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


class ContrastiveDVSDataset(Dataset):
    """
    Wrapper for contrastive learning on neuromorphic DVS datasets.
    Input frame shape from spikingjelly: [T, C, H, W] (numpy float32)
    Output: ((frame_i, frame_j), label)
    target_size: if set (e.g. (128, 128)), resize before augmentation.
    """
    def __init__(self, subset, target_size=None):
        self.subset = subset
        self.transform = DVSBarlowTwinsTransform()
        self.target_size = target_size

    def __getitem__(self, index):
        frame, label = self.subset[index]
        frame = torch.FloatTensor(frame)  # [T, C, H, W]
        if self.target_size is not None:
            frame = F.interpolate(frame, size=self.target_size, mode='bilinear', align_corners=False)
        return self.transform(frame), label

    def __len__(self):
        return len(self.subset)


class ValDVSDataset(Dataset):
    """
    Wrapper for evaluation on neuromorphic DVS datasets.
    Input frame shape from spikingjelly: [T, C, H, W] (numpy float32)
    Output: (frame, label)
    target_size: if set (e.g. (128, 128)), resize the frame.
    """
    def __init__(self, subset, target_size=None):
        self.subset = subset
        self.target_size = target_size

    def __getitem__(self, index):
        frame, label = self.subset[index]
        frame = torch.FloatTensor(frame)  # [T, C, H, W]
        if self.target_size is not None:
            frame = F.interpolate(frame, size=self.target_size, mode='bilinear', align_corners=False)
        return frame, label

    def __len__(self):
        return len(self.subset)


def build_dvs128gesture(path, T=16):
    """
    DVS128 Gesture: 11 classes, 128x128 resolution.
    Has a built-in train/test split via spikingjelly.
    T=16 is the default number of frames.
    """
    train_origin = DVS128Gesture(root=path, train=True, data_type='frame', frames_number=T, split_by='number')
    val_origin = DVS128Gesture(root=path, train=False, data_type='frame', frames_number=T, split_by='number')
    train_dataset = ContrastiveDVSDataset(train_origin)
    val_dataset = ValDVSDataset(val_origin)
    return train_dataset, val_dataset


def build_ncaltech101(path, T=10, train_ratio=0.9):
    """
    N-Caltech101: 101 classes, 240x180 resolution.
    No built-in train/test split; uses split_to_train_test_set(0.9).
    Note: iterating the full dataset to build the split index takes a while on first call.
    """
    origin_set = NCaltech101(root=path, data_type='frame', frames_number=T, split_by='number')
    train_subset, val_subset = split_to_train_test_set(train_ratio, origin_set, 101)
    train_dataset = ContrastiveDVSDataset(train_subset)
    val_dataset = ValDVSDataset(val_subset)
    return train_dataset, val_dataset


def build_asldvs(path, T=10, train_ratio=0.9):
    """
    ASL-DVS: 24 classes (A-Y, excluding J), 100,800 total samples (4,200 per class),
    180x240 resolution resized to 128x128.
    No built-in train/test split.
    """
    origin_set = ASLDVS(root=path, data_type='frame', frames_number=T, split_by='number')
    train_subset, val_subset = split_to_train_test_set(train_ratio, origin_set, 24)
    train_dataset = ContrastiveDVSDataset(train_subset, target_size=(128, 128))
    val_dataset = ValDVSDataset(val_subset, target_size=(128, 128))
    return train_dataset, val_dataset


DATASET_NUM_CLASSES = {
    'dvs128-gesture': 11,
    'ncaltech101':    101,
    'asl-dvs':        24,
}


def build_dataset(name, path, T=10):
    """Unified entry for training: train split uses contrastive augmentation."""
    if name == 'dvs128-gesture':
        return build_dvs128gesture(path, T=T)
    elif name == 'ncaltech101':
        return build_ncaltech101(path, T=T)
    elif name == 'asl-dvs':
        return build_asldvs(path, T=T)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_NUM_CLASSES)}")


def build_dataset_for_eval(name, path, T=10):
    """Unified entry for evaluation: both splits return single frames (no contrastive pairs)."""
    if name == 'dvs128-gesture':
        train_subset = DVS128Gesture(root=path, train=True, data_type='frame', frames_number=T, split_by='number')
        val_subset = DVS128Gesture(root=path, train=False, data_type='frame', frames_number=T, split_by='number')
    elif name == 'ncaltech101':
        origin_set = NCaltech101(root=path, data_type='frame', frames_number=T, split_by='number')
        train_subset, val_subset = split_to_train_test_set(0.9, origin_set, 101)
    elif name == 'asl-dvs':
        origin_set = ASLDVS(root=path, data_type='frame', frames_number=T, split_by='number')
        train_subset, val_subset = split_to_train_test_set(0.9, origin_set, 24)
        return ValDVSDataset(train_subset, target_size=(128, 128)), ValDVSDataset(val_subset, target_size=(128, 128))
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_NUM_CLASSES)}")
    return ValDVSDataset(train_subset), ValDVSDataset(val_subset)


if __name__ == '__main__':
    train_set, test_set = build_dvs128gesture("./data")
