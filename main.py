from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import energyflow as ef
import warnings
import logging
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from dataset import  FalconDataset


from edgenet import EdgeNet

warnings.filterwarnings("ignore")
logging.basicConfig(filename='training.log',level=logging.INFO)


def preprocess_emd(torch_batch):
    batch_size = torch_batch.batch[-1] + 1
    ret = []
    for batch_idx in range(batch_size):
        ret.append(torch_batch.x[torch_batch.batch == batch_idx].cpu().detach().numpy())
    return ret

def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    batch_loss = []
    for batch_idx, (data_h) in enumerate(train_loader):
        data = data_h.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        #if (batch_idx not in target_emds):
        #    pixel_list = preprocess(data_h)
        #    target_emds[batch_idx] = torch.from_numpy(ef.emd.emds(pixel_list, R=40.0)).float().to(device)
        #target = target_emds[batch_idx]
        nodes_list = preprocess_emd(data_h)
        target = torch.from_numpy(ef.emd.emds(nodes_list, R=1.0)).float().to(device)
        output_dist = torch.cdist(output, output, p=2.1)
        
        loss = F.mse_loss(output_dist, target)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        
#        print(output)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    print('Train Epoch: {}  Batch loss:{}'.format(epoch, sum(batch_loss)/len(batch_loss)))
    logging.info('Train Epoch: {}  Batch loss:{}'.format(epoch, sum(batch_loss)/len(batch_loss)))
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Falcon EMD')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', type=str, default=None, help='dir of trained model to continue training from')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device('cuda:0')

    kwargs = {'num_workers': 1, 'pin_memory': True} 
    dataset = FalconDataset('')
    print('training on {} events'.format(dataset.len()))
    
    train_loader = DataLoader(dataset,
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    model = EdgeNet().to(device)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    target_emds = {}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        #test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
