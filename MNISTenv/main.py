import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # Convolutional layer with 1 input channel, 32 output channels, kernel size of 3x3 and stride of 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # Convolutional layer with 32 input channel from layer 1's output, 64 output channels, kernel size of 3x3 and stride of 1
        self.dropout1 = nn.Dropout(0.25) # Drops 25 percent of neurons
        self.dropout2 = nn.Dropout(0.5) # Drops 50 percent of neurons
        self.fc1 = nn.Linear(9216, 128) 
        self.fc2 = nn.Linear(128, 10)

    # Define the flow of our image data through the model 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Reduce size of image, i.e reduce by a factor of 2
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # Reduce image data from multidimensional to 1D
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Move images to defined device (mps, cuda, cpu)
        optimizer.zero_grad() # PyTorch accumulates zeroes, reset gradient
        output = model(data)
        loss = F.cross_entropy(output, target) # Loss function
        loss.backward() # Calculate gradients
        optimizer.step() # Uses the loss function to update weights of our model to calculate loss

        # visualize to tensorBoard
        if writer:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, writer):
    model.eval() # Set model to evaluation mode, don't update weights while testing
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Disables gradient calculations to speed up testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if writer:
        writer.add_scalar('Accuracy/test', accuracy, epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # For visualizing on tensorboard
    writer = SummaryWriter()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        print('using cuda device')
        device = torch.device("cuda")
    elif use_mps:
        print('using mps')
        device = torch.device("mps")
    else:
        print('using cpu')
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, writer)
        scheduler.step()
    writer.close()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

def visualize_results(data, target, output, num_samples=8):
    data, target = data.cpu(), target.cpu()
    preds = output.argmax(dim=1)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
    for i in range(num_samples):
        axes[i].imshow(data[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {preds[i]}\nTrue: {target[i]}')
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    main()