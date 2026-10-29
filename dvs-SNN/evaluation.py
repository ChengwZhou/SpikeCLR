import numpy as np
import argparse

import torch
import torch.nn as nn
import data_loaders
import barlowtwins  # registers 'barlow_twins' into timm's model registry
from timm.models import create_model
from timm.utils import *

# from transforms_factory import contrastive_learning_transforms as DataTransforms

def parse_args():
    parser = argparse.ArgumentParser(description='Eval Config', add_help=False)
    parser.add_argument('--time_step', default=10, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=512, type=int, help='dimension of output feature')
    parser.add_argument('--num_classes', default=None, type=int,
                        help='number of label classes (inferred from --dataset-name if not set)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--dataset', default='', help='root path of dataset')
    parser.add_argument('--dataset-name', default='dvs128-gesture',
                        choices=['dvs128-gesture', 'ncaltech101', 'asl-dvs'],
                        help='which neuromorphic dataset to use')
    parser.add_argument('--eval_proportion', default='1', type=float,
                        help='proportion of test dataset for evaluation')

    parser.add_argument('--val_epochs', default=500, type=int)
    parser.add_argument('--val_batch_size', default=256, type=int)

    # Act func
    parser.add_argument('--act-func', default='MixedLIF', type=str,
                        choices=['LIF', 'MixedLIF', 'LIFt'],
                        help='spiking activation function')

    # Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        choices=['resnet14', 'resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet50', 'resnet101'],
                        help='ResNet backbone variant (must match the pretrained checkpoint)')

    args = parser.parse_args()
    return args


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


class MLP_classifation(nn.Module):
    def __init__(self, dim, num_classes):
        super(MLP_classifation, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def inference(args, loader, bt_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            h, _, _, _ = bt_model(x, x)
        h = h.flatten(0, 1).detach() # [T*B, C]

        feature_vector.extend(h.cpu().detach().numpy())

        # expand labels_vector to match T*B features
        y_temp = torch.cat([y] * args.time_step)  # [T*B]
        labels_vector.extend(y_temp)

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


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch_top1 = 0
    accuracy_epoch_top5 = 0
    for step, (x, y) in enumerate(loader):
        T, B = args.time_step, int(x.shape[0]/args.time_step)
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device).reshape(T, -1)[0]

        output = model(x)  # [T*B, num_classes]
        output = output.reshape(T, B, -1).mean(0)
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

    return loss_epoch, accuracy_epoch_top1, accuracy_epoch_top5


def test(args, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch_top1 = 0
    accuracy_epoch_top5 = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        T, B = args.time_step, int(x.shape[0]/args.time_step)
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device).reshape(T, -1)[0]

        output = model(x)  # [T*B, num_classes]
        output = output.reshape(T, B, -1).mean(0)
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


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.lr = float(args.lr)
    if args.num_classes is None:
        args.num_classes = data_loaders.DATASET_NUM_CLASSES[args.dataset_name]
    print(vars(args))

    train_dataset, test_dataset = data_loaders.build_dataset_for_eval(args.dataset_name, args.dataset, T=args.time_step)

    # Create a 1%/10%/100% data subset using the random indices
    num_samples = int(len(train_dataset) * args.eval_proportion)
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_subset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.val_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize ResNet, encoder is resnet/resnet_snn
    # load pre-trained model from checkpoint
    bt_model = barlowtwins.barlow_twins(
        act_func=args.act_func,
        name=args.backbone,
        timestep=args.time_step,
        num_classes=args.num_classes,
        sync_norm=True,
    )
    print(f"Backbone: {args.backbone}")

    # bt_model.load_state_dict(torch.load(args.resume, map_location=args.device.type))
    # resume_epoch = resume_checkpoint(
    #     bt_model, args.resume,
    #     optimizer=None,
    #     loss_scaler=None,
    #     log_info=0)
    checkpoint = torch.load(args.resume, map_location=args.device)
    # train.py saves model_without_ddp.state_dict() directly (not a dict with 'model' key)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        bt_model.load_state_dict(checkpoint['model'])
    else:
        bt_model.load_state_dict(checkpoint)
    print(f"Model loaded from checkpoint: {args.resume}")

    bt_model = bt_model.to(args.device)
    bt_model.eval()

    ## Logistic Regression
    model = LogisticRegression(args.dim, args.num_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        args, bt_model, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.val_batch_size * args.time_step  # time augmented
    )

    for epoch in range(args.val_epochs):
        loss_epoch, accuracy_epoch_top1, accuracy_epoch_top5= train(
            args, arr_train_loader, model, criterion, optimizer
        )
        if epoch % 20 == 0:
            print(
                f"Epoch [{epoch}/{args.val_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy top-1: {accuracy_epoch_top1 / len(arr_train_loader)}\t Accuracy top-5: {accuracy_epoch_top5 / len(arr_train_loader)}"
            )

    # final testing
    loss_epoch, accuracy_top1, accuracy_top5 = test(
        args, arr_test_loader, model, criterion
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy top-1: {accuracy_top1 / len(arr_test_loader)}\t Accuracy top-5: {accuracy_top5 / len(arr_test_loader)}"
    )
