import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.distributed as dist
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import data
import model

# Add ckp
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--dir-data', type=str, default='/home/kd/PycharmProjects/LSTM-PTB-kd/PTB/data', # /input
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='',
                    help='model checkpoint to use')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',default='ture',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',default='true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='/home/kd/PycharmProjects/LSTM-PTB-kd/output/model2.pt',
                    help='path to save the final model')
parser.add_argument('--file-name',default='PTB_Baseline',type=str)
parser.add_argument('--init-method',default='tcp://172.18.233.41:24546',type=str)
parser.add_argument('--dist-rank', default=0, type=int)
parser.add_argument('--world-size', default=2, type=int)
args = parser.parse_args()



# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.dir_data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    train_data = Variable(source[i:i+seq_len], volatile=evaluation)
    train_target = Variable(source[i+1:i+1+seq_len].view(-1))
    return train_data, train_target

def dataset(source):
    data = []
    targets = []
    for i in range(0,source.size(0)-args.bptt,args.bptt):
        data1,target1 = get_batch(source,i)
        data.append(data1)
        targets.append(target1)
    data = torch.stack(data,0)
    print(data.size())
    target = torch.stack(targets,0)
    return data,target


eval_batch_size = 10
train_data_dataset = batchify(corpus.train, args.batch_size)
class DataSet_LSTM(Dataset):
    def __init__(self,data,data_set=dataset):
        self.data_set = data_set
        self.data, self.targets = self.data_set(data)
    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        return data,target
    def __len__(self):
        return len(self.data)
train_dataset = DataSet_LSTM(train_data_dataset)
sampler = DistributedSampler(dataset=train_dataset,num_replicas=2,rank=args.dist_rank)
#sampler = DistributedSampler(dataset=dataset(train_data_dataset),num_replicas=2,rank=args.dist_rank)
train_data = DataLoader(dataset=train_dataset,sampler=sampler,shuffle=(sampler is None),batch_size=args.batch_size)
val_data_dataset = batchify(corpus.valid, eval_batch_size)
val_dataset = DataSet_LSTM(val_data_dataset)
val_data = DataLoader(dataset=val_dataset,shuffle=(sampler is None),batch_size=args.batch_size)
test_data_dataset = batchify(corpus.test, eval_batch_size)
test_dataset = DataSet_LSTM(test_data_dataset)
test_data = DataLoader(dataset=test_dataset,shuffle=(sampler is None),batch_size=args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

# Load checkpoint
if args.checkpoint != '':
    if args.cuda:
        model = torch.load(args.checkpoint)
    else:
        # Load GPU model on CPU
        model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

if args.cuda:
    model.cuda()
else:
    model.cpu()

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

###############################################################################
# Training code
###############################################################################
Total_param_num = 0
def average_gradients(model):
    size = float(dist.get_world_size())
    global Total_param_num
    Total_param_num = 0
    for param in model.parameters():
        Total_param_num += param.grad.data.nelement()
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
    return model

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    # if type(h) == Variable:
    #     return Variable(h.data)
    # else:
    #     return tuple(repackage_hidden(v) for v in h)
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    #for i in range(0, data_source.size(0) - 1, args.bptt):
    for i, data in enumerate(data_source, 0):
        data, targets = data
        len_data = data.size(0)
        for j in range(len_data):
            output, hidden = model(data[j], hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets[j]).data
            hidden = repackage_hidden(hidden)
    len1 = len(data_source)*args.batch_size
    print(len1)
    return total_loss[0] / len1


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    #for batch, i in enumerate(range(0, train_data_loader.size(0) - 1, args.bptt)):
    for i,data in enumerate(train_data,0):
        #data, targets = get_batch(train_data, i)
        #data, targets = get_batch(data, i*args.bptt)
        datai, targets = data
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        for batch_num in range(datai.size(0)):
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(datai[batch_num], hidden)
            loss = criterion(output.view(-1, ntokens), targets[batch_num])
            loss.backward()
            average_gradients(model)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # 没有用SGD进行更新，使用了clip_grad_norm函数来进行更新
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            # 这一段代码是对所有的参数进行更新
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.data
        cur_loss = total_loss[0] / args.batch_size
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | Total {:10.2f}'
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data), lr,
                elapsed * 1000 / len(train_data), Total_param_num, cur_loss, math.exp(cur_loss)))
        total_loss = 0
        start_time = time.time()
        # if i % args.log_interval == 0 and i > 0:
        #     cur_loss = total_loss[0] / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | Total {:10.2f}'
        #             'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, i, len(train_data) // args.bptt, lr,
        #         elapsed * 1000 / args.log_interval, Total_param_num, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    print("start")
    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.dist_rank,
                            world_size=args.world_size)
    print("end")
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        print("start1")
        train()
        print("end1")
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)