import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import time
import copy
import torchvision.models as models
from scipy import sparse
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--dist-rank', default=0, type=int)
parser.add_argument('--world-size', default=2, type=int)
parser.add_argument('--dir-data', default='/home/kd/PycharmProjects/cifar10/data', type=str)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--epochs',default=30,type=int)
parser.add_argument('--file-name',default='AlexNet_Baseline',type=str)
parser.add_argument('--init-method',default='tcp://172.18.233.41:24546',type=str)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


Total_param_num = 0
Sparse_param_num = 0

def average_gradients(model):
    size = float(dist.get_world_size())
    global Total_param_num
    Total_param_num = 0
    for param in model.parameters():
        Total_param_num += param.grad.data.nelement()
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
    return model

# 获取数据
def run():

    #Logging Prepare
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='{}_Log.log'.format(args.file_name),
                        filemode='w')
    # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    #Data Prepare
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=(train_sampler is None),num_workers=1, pin_memory=True, sampler=train_sampler)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=(train_sampler is None),num_workers=1, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=1)


    #Model Prepare
    net = AlexNet()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-3)
    net.train()
    net = net.cuda()
    # net = nn.parallel.DistributedDataParallel(net)


    logs = []
    print ("Training Start")
    start_time = time.time()
    for epoch in range(args.epochs):
        print ("Training for epoch {}".format(epoch))
        # train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader,0):
            batch_start_time = time.time()
            x, label = data
            x, label = Variable(x).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output = net(x)
            loss = F.cross_entropy(output, label)
            loss.backward()
            average_gradients(net)
            optimizer.step()

            _, train_predict = torch.max(output,1)
            train_accuracy = accuracy_score(train_predict,label)

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.data.item(),
                'training_accuracy': train_accuracy,
                'total_param': Total_param_num,
                'sparse_param': Total_param_num,
                'mini_batch_time': (time.time()-batch_start_time)
            }
            if i % 20 ==0:
                print ("Timestamp: {timestamp} | "
                       "Iteration: {iteration:6} | "
                       "Loss: {training_loss:6.4f} | "
                       "Accuracy: {training_accuracy:6.4f} | "
                       "Total_param: {total_param:6} | "
                       "Sparse_param: {sparse_param:6} | "
                       "Mini_Batch_Time: {mini_batch_time:6.4f} | ".format(**log_obj))

            logs.append(log_obj)
        if True:
            logs[-1]['test_loss'],logs[-1]['test_accuracy'] = evaluate(net,test_loader,)
            print  ("Timestamp: {timestamp} | "
                   "Iteration: {iteration:6} | "
                   "Loss: {training_loss:6.4f} | "
                   "Accuracy: {training_accuracy:6.4f} | "
                   "Total_param: {total_param:6} | "
                   "Sparse_param: {sparse_param:6} | "
                   "Mini_Batch_Time: {mini_batch_time:6.4f} | "
                   "Test Loss: {test_loss:6.4f} | "
                   "Test Accuracy: {test_accuracy:6.4f}".format(**logs[-1]))
        val_loss, val_accuracy = evaluate(net, test_loader)
        scheduler.step(val_loss)

    df = pd.DataFrame(logs)
    # print (df)
    df.to_csv('./log/{}_Node{}_{}.csv'.format(args.file_name,args.dist_rank,datetime.now().strftime("%Y-%m-%d %H:%M:%S")), index_label='index')
    print ("Finished Training")


def evaluate(net,test_loader,):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            output = net(inputs)
            _, predicted = torch.max(output, 1)
            test_loss += F.cross_entropy(output, labels).item()

    test_accuracy = accuracy_score(predicted, labels)
    return test_loss, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args()
    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.dist_rank,world_size=args.world_size)
    # dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,group_name='mygroup')
    run()

