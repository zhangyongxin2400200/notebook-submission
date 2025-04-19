import os
import numpy as np
from pathlib import Path
from data_prep_bbh import *
from utils import *

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch import nn

# 数据生成器部分保持不变
class DatasetGenerator(Dataset):
    def __init__(self, fs=8192, T=1, snr=20,
                 detectors=['H1', 'L1'],
                 nsample_perepoch=100,
                 Nnoise=25, mdist='metric', beta=[0.75, 0.95],
                 verbose=True):
        if verbose:
            print('GPU available?', torch.cuda.is_available())
        self.fs = fs
        self.T = T
        safe = 2
        self.T *= safe
        self.detectors = detectors
        self.snr = snr
        self.generate(nsample_perepoch, Nnoise, mdist, beta)

    def generate(self, Nblock, Nnoise=25, mdist='metric', beta=[0.75, 0.95]):
        ts, par = sim_data(self.fs, self.T, self.snr, self.detectors, Nnoise, size=Nblock, mdist=mdist,
                           beta=beta, verbose=False)
        self.strains = np.expand_dims(ts[0], 1)  # (nsample, 1, len(det), fs*T)
        self.labels = ts[1]

    def __len__(self):
        return len(self.strains)

    def __getitem__(self, idx):
        return self.strains[idx], self.labels[idx]

# 定义 ResNet 模型
class MyResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MyResNet, self).__init__()
        # 使用预训练的 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一层卷积层的输入通道数为 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 修改全连接层的输出类别数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 模型保存和加载函数保持不变
def load_model(checkpoint_dir=None):
    net = MyResNet()

    if (checkpoint_dir is not None) and (Path(checkpoint_dir).is_dir()):
        p = Path(checkpoint_dir)
        files = [f for f in os.listdir(p) if '.pt' in f]

        if (files != []) and (len(files) == 1):
            checkpoint = torch.load(p / files[0])
            net.load_state_dict(checkpoint['model_state_dict'])
        print('Load network from', p / files[0])
        
        epoch = checkpoint['epoch']
        train_loss_history = np.load(p / 'train_loss_history_cnn.npy').tolist()
        return net, epoch, train_loss_history
    else:
        print('Init network!')
        return net, 0, []

def save_model(epoch, model, optimizer, scheduler, checkpoint_dir, train_loss_history, filename):
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    assert '.pt' in filename
    for f in [f for f in os.listdir(p) if '.pt' in f]:
        os.remove(p / f)

    np.save(p / 'train_loss_history_cnn', train_loss_history)

    output = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    if scheduler is not None:
        output['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(output, p / filename)

# 训练和评估函数部分
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, dim=1)        
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy_gpu(net, data_iter, loss_func, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            y_hat = net(X)
            loss = loss_func(y_hat, y)
            metric.add(accuracy(y_hat, y), y.numel(), loss.sum())
    return metric[0] / metric[1], metric[2] / metric[1]

def train(net, lr, nsample_perepoch, epoch, total_epochs,
          dataset_train, data_loader, test_iter,
          train_loss_history, checkpoint_dir, device, notebook=True):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    torch.cuda.empty_cache()
    if notebook:
        animator = Animator(xlabel='epoch', xlim=[1, total_epochs], legend=['train loss', 'test loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(dataset_train)

    for epoch in range(epoch, epoch + total_epochs):
        dataset_train.generate(nsample_perepoch)

        if not notebook:
            print('Learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        metric = Accumulator(3)
        net.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            timer.start()
            optimizer.zero_grad()

            data = x.to(device, non_blocking=True).to(torch.float)
            label = y.to(device, non_blocking=True).to(torch.long)

            pred = net(data)
            loss = loss_func(pred, label)

            with torch.no_grad():
                metric.add(loss.sum(), accuracy(pred, label), x.shape[0])
            timer.stop()

            loss.backward()
            optimizer.step()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if notebook and (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
                animator.add(epoch + (batch_idx + 1) / num_batches, (train_l, None, train_acc, None))

        scheduler.step()

        test_acc, test_l = evaluate_accuracy_gpu(net, test_iter, loss_func, device)

        train_loss_history.append([epoch+1, train_l, test_l, train_acc, test_acc])

        if notebook:
            animator.add(epoch + 1, (train_l, test_l, train_acc, test_acc))
        else:
            print(f'Epoch: {epoch+1} \tTrain Loss: {train_l:.4f} Test Loss: {test_l:.4f} \tTrain Acc: {train_acc} Test Acc: {test_acc}')

        if (test_l <= min(np.asarray(train_loss_history)[:,1])):
            save_model(epoch, net, optimizer, scheduler, checkpoint_dir=checkpoint_dir, train_loss_history=train_loss_history, filename=f'model_e{epoch}.pt')

    print(f'loss {train_l:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * total_epochs / timer.sum():.1f} examples/sec on {str(device)}')

if __name__ == "__main__":
    nsample_perepoch = 100
    dataset_train = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)
    dataset_test = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)

    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_iter = DataLoader(dataset_test, batch_size=32, shuffle=True)

    device = torch.device('cuda')

    checkpoint_dir = './checkpoints_resnet/'

    net, epoch, train_loss_history = load_model(checkpoint_dir)
    net.to(device)

    lr = 0.003
    total_epochs = 100
    total_epochs += epoch
    output_freq = 1

    train(net, lr, nsample_perepoch, epoch, total_epochs, data_loader, test_iter, notebook=False)