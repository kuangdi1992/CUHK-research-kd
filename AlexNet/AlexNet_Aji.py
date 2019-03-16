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
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--epochs',default=30,type=int)
parser.add_argument('--file-name',default='AlexNet_Aji',type=str)
parser.add_argument('--init-method',default='tcp://172.18.233.41:22369',type=str)

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

def trans_gradients(net):
    world_size = float(dist.get_world_size())
    values = []
    global index
    global Total_param_num
    global Sparse_param_num
    Total_param_num = 0
    Sparse_param_num = 0
    for i, param in enumerate(net.parameters()):
        temp_param = copy.deepcopy(param.grad.data)
        Total_param_num += param.grad.data.nelement()
        values.append(temp_param[temp_param != 0])
        if len(values[i]) != 0:
            index = torch.LongTensor(np.where(temp_param.cpu() != 0)).cuda()
        else:
            index = torch.LongTensor([]).cuda()

        global index_list
        global value_list
        global size_list

        # Get index and values size
        # selfsize
        size = torch.tensor([len(index), len(index[0]) if len(index) != 0 else 0]).cuda()
        # initial size_list for gather whole group tensor of size
        size_list = [torch.tensor([2, 2]).cuda() for j in range(args.world_size)]
        dist.all_gather(size_list, size)

        max_index0 = int(max(size_list[j][0] for j in range(args.world_size)))
        max_index1 = int(max(size_list[j][1] for j in range(args.world_size)))

        if max_index0==0 :
            continue
        else :
            # Get index of sparse matrix
            # selfindex and scalar index
            supplement_index = torch.LongTensor([-1 for j in range(max_index1 - (len(index[0]) if len(index) != 0 else 0))]).cuda()
            temp_index = torch.LongTensor(max_index0, max_index1).cuda()
            if len(index) == 0:
                for j in range(max_index0):
                    temp_index[j] = supplement_index
            else:
                for j in range(max_index0):
                    temp_index[j] = torch.cat((index[j], supplement_index), 0)

            index = temp_index
            Sparse_param_num += (index.nelement() + max_index1)
            # initial index_list for gather whole group tensor of index
            index_list = [torch.LongTensor([[2 for k in range(max_index1)] for j in range(max_index0)]).cuda() for l in range(args.world_size)]
            dist.all_gather(index_list, index)

            # Get values of sparse matrix
            # selfvalues
            # initial supplement_value for values[i]
            supplement_value = torch.FloatTensor([-1 for j in range(max_index1 - len(values[i]))]).cuda()
            # initial value_list for gather whole group tensor of value
            value_list = [torch.FloatTensor([2 for k in range(max_index1)]).cuda() for j in range(args.world_size)]
            dist.all_gather(value_list, torch.cat((values[i], supplement_value), 0).cuda())

            # print(size_list)
            # Wash the data of index and values
            for j in range(args.world_size):
                index_list[j] = index_list[j][index_list[j] != -1]
                value_list[j] = value_list[j][value_list[j] != -1]
                if len(index_list[j]) != 0:
                    index_list[j] = index_list[j].view(size_list[j][0], size_list[j][1])
                    param.grad.data += torch.sparse.FloatTensor(index_list[j], value_list[j],param.size()).to_dense().cuda()
            param.grad.data /= world_size


def gradient_execute(net):
    threshold = 0.002
    paralist = []
    for param in net.parameters():
        temp = copy.deepcopy(param.grad.data)
        topn = torch.topk(abs(temp.view(1,-1)),int(temp.nelement()*0.01) if int(temp.nelement()*0.01) != 0 else 1)
        threshold = float(topn[0][0][len(topn[0][0])-1])
        temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0
        paralist.append(temp)

    trans_gradients(net) 
    return paralist

# 获取数据
def run():

    #Data Prepare
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=(train_sampler is None),num_workers=1, pin_memory=True, sampler=train_sampler)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dir_data, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=(train_sampler is None),num_workers=1, pin_memory=True)


    #Model Prepare
    net = AlexNet()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=5e-3)
    net.train()
    net = net.cuda()
    # net = nn.parallel.DistributedDataParallel(net).cuda()


    logs = []
    print("Training Start")
    start_time = time.time()
    for epoch in range(args.epochs):
        print("Training for epoch {}".format(epoch))
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader,0):
            batch_start_time = time.time()
            x, label = data
            x, label = Variable(x).cuda(), Variable(label).cuda()
            # optimizer.zero_grad()
            output = net(x)
            loss = F.cross_entropy(output, label)
            loss.backward()
            paralist = gradient_execute(net)
            optimizer.step()
            for para1, para2 in zip(paralist, net.parameters()):
                para2.grad.data = para1

            _, train_predict = torch.max(output,1)
            train_accuracy = accuracy_score(train_predict,label)

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.data.item(),
                'training_accuracy': train_accuracy,
                'total_param': Total_param_num,
                'sparse_param': Sparse_param_num,
                'mini_batch_time': (time.time() - batch_start_time)
            }
            if i % 20 == 0:
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Accuracy: {training_accuracy:6.4f} | "
                      "Total_param: {total_param:6} | "
                      "Sparse_param: {sparse_param:6} | "
                      "Mini_Batch_Time: {mini_batch_time:6.4f} | ".format(**log_obj))

            logs.append(log_obj)
        if True:
            logs[-1]['test_loss'], logs[-1]['test_accuracy'] = evaluate(net, test_loader, )
            print("Timestamp: {timestamp} | "
                  "Iteration: {iteration:6} | "
                  "Loss: {training_loss:6.4f} | "
                  "Accuracy: {training_accuracy:6.4f} | "
                  "Total_param: {total_param:6} | "
                  "Sparse_param: {sparse_param:6} | "
                  "Mini_Batch_Time: {mini_batch_time:6.4f} | "
                  "Test Loss: {test_loss:6.4f} | "
                  "Test Accuracy: {test_accuracy:6.4f}".format(**logs[-1]))
        val_loss, val_accuracy = evaluate(net, test_loader)
        #scheduler.step(val_loss)

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
    # dist.init_process_group(backend='nccl', init_method='file:///home/mengu/share_file_chenmq',world_size=args.world_size)
    run()

