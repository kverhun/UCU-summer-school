from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./checkpoints/')
parser.add_argument('--reconstruct_path', type=str, default='./reconstructions')
parser.add_argument('--train', action='store_true', default=True)
parser.add_argument('--continue_epoch', type=int, default=10)
parser.add_argument('--d_z', type=int, default=20, help='dimensionality of latent space')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.reconstruct_path):
    os.makedirs(args.reconstruct_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    # Task 2. Define NN layers for VAE task
    # Follow this architecture
    #            fc (400)
    #               |
    #            fc (400)
    #            |     |
    #       fc(400) fc(400)
    #            |     |
    #       fc_mu (20)  fc_logvar(20)
    #            |       |
    #             -------
    #                |
    #              fc(400)
    #                |
    #              fc(784)

    def __init__(self):
        super(VAE, self).__init__()
        #Define your layers hear
        raise NotImplementedError

    def encode(self, x):
        """Task2. Define encoder"""
        raise NotImplementedError

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Task2. Define decoder"""
        raise NotImplementedError


    def forward(self, x):
        """Task2 define forward path. which should consist of
        encode reparametrize and decode function calls"""
        x = x.view(-1, 784)
        raise NotImplementedError

    def save_model(self, epoch):
        model_file = os.path.join(args.save_path, 'vae_{}.th'.format(epoch))
        torch.save(self.state_dict(), model_file)
        print('model saved after epoch: {:03d}'.format(epoch))

    def load_model(self, epoch):
        model_file = os.path.join(args.save_path, 'vae_{}.th'.format(epoch))
        self.load_state_dict(torch.load(model_file))
        print('model restored from checkpoint on epoch: {:03d}'.format(epoch))



reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False

model = VAE()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    recosntruction_loss = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return recosntruction_loss + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, ):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        recon_file = os.path.join(args.reconstruct_path, 'im_{}_{}_mse.jpg'.format(epoch, i))
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            utils.save_image(recon_batch.data.resize_((args.batch_size, 1, 28, 28)), recon_file, normalize=True)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def manifold_visulisation():
    """Task 3. Train VAE with len(z) = 2
    sample z values from closed interval of [-3, 3].
    visualize generated examples on the grid"""
    xs = None
    latent_sp_file = os.path.join(args.reconstruct_path, 'latent_{}_{}.jpg'.format(epoch, 'test'))
    utils.save_image(xs, latent_sp_file, normalize=True)
    raise NotImplementedError



if __name__ == '__main__':
    if args.train:
        print('training model')
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            model.save_model(epoch)
            test(epoch)
    else:
        print('testing model')
        if args.continue_epoch != -1:
            epoch = args.continue_epoch
            model.load_model(epoch)
            manifold_visulisation()


