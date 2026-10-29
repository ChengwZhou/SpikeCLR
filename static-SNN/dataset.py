import os
from torch.utils.data import Dataset
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of TinyImageNet dataset.
            mode (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert mode in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.samples = []

        if mode == 'train':
            self._load_train()
        elif mode == 'val':
            self._load_val()
        else:
            self._load_test()

    def _load_train(self):
        train_dir = os.path.join(self.root_dir, 'train')
        wnids = sorted(os.listdir(train_dir))  # 200 class folders
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        for wnid in wnids:
            img_dir = os.path.join(train_dir, wnid, 'images')
            for img_name in os.listdir(img_dir):
                self.samples.append((os.path.join(img_dir, img_name), self.class_to_idx[wnid]))

    def _load_val(self):
        val_dir = os.path.join(self.root_dir, 'val')
        anno_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        self.class_to_idx = self._load_class_to_idx()
        for line in lines:
            tokens = line.strip().split('\t')
            img_name, wnid = tokens[0], tokens[1]
            img_path = os.path.join(val_dir, 'images', img_name)
            label = self.class_to_idx[wnid]
            self.samples.append((img_path, label))

    def _load_test(self):
        test_dir = os.path.join(self.root_dir, 'test', 'images')
        for img_name in sorted(os.listdir(test_dir)):
            self.samples.append((os.path.join(test_dir, img_name), -1))  # No label for test

    def _load_class_to_idx(self):
        wnids_path = os.path.join(self.root_dir, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return {wnid: idx for idx, wnid in enumerate(sorted(wnids))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label