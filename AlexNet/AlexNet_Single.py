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

parser = argparse.ArgumentParser()
parser.add_argument('--dist-rank', default=0, type=int)
parser.add_argument('--world-size', default=2, type=int)
parser.add_argument('--dir-data', default='./data', type=str)
parser.add_argument('--lr',default=0.1,type=float)
parser.add_argument('--epochs',default=30,type=int)
parser.add_argument('--file-name',default='AlexNet_Single',type=str)

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


def run():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=False,download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=False,num_workers=1)

    net = AlexNet()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-3)
    net.train()
    net = net.cuda()

    logs = []
    #Training

    for epoch in range(args.epochs):
        print("Training for epoch {}".format(epoch))
        for i,data in enumerate(train_loader,0) :
            Total_param_num = 0
            Sparse_param_num = 0
            batch_start_time = time.time()
            inputs , labels = data
            inputs, labels = Variable(inputs).cuda() , Variable(labels).cuda()
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()

            for param in net.parameters():
                Total_param_num += param.grad.data.nelement()

            _, predicted = torch.max(output, 1)
            train_accuracy = accuracy_score(predicted, labels)

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
            logs[-1]['test_loss'], logs[-1]['test_accuracy'] = evaluate(net, test_loader, )
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
    df.to_csv('./log/{}_{}.csv'.format(args.file_name,datetime.now().strftime("%Y-%m-%d %H:%M:%S")),index_label='index')
    print('Finished Training')

def evaluate(net, testloader):

    net.eval()
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            output = net(inputs)
            _, predicted = torch.max(output, 1)
            test_loss += F.cross_entropy(output, labels).item()

    test_accuracy = accuracy_score(predicted, labels)
    return test_loss, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args()
    run()