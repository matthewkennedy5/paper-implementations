import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import pdb

TEST_SIZE = 10000

class SequentialMNIST(Dataset):
    """Contains the Sequential MNIST data from https://arxiv.org/abs/1504.00941.

    This dataset consists of MNIST images flattened and read pixel-by-pixel from
    left to right.
    """

    def __init__(self, train=True):
        """
        Inputs:
            train - If True, load the train set. If False, load the validation
            set.
        """
        self.train = train

        # Load the MNIST images and flatten into a new data tensor
        mnist_train_dataloader = torchvision.datasets.MNIST(root='', train=True, download=True, transform=ToTensor())
        mnist_test_dataloader = torchvision.datasets.MNIST(root='', train=False, download=True, transform=ToTensor())
        N = len(mnist_train_dataloader) + len(mnist_test_dataloader)
        sequence_len = 28 * 28  # MNIST image size
        data = torch.empty(N, sequence_len)
        truth = torch.LongTensor(N)
        for i, sample in enumerate(mnist_train_dataloader + mnist_test_dataloader):
            flattened_image = sample[0].view(sequence_len)
            data[i, :] = flattened_image
            truth[i] = sample[1]

        # Partition the train and test data. This is a different partition than
        # in the original MNIST dataset, and is presumably different than the
        # partition the authors used.
        self.train_data = data[TEST_SIZE:, :]
        self.test_data = data[:TEST_SIZE, :]
        self.train_truth = truth[TEST_SIZE:]
        self.test_truth = truth[:TEST_SIZE]
        # TODO: Should I normalize based on mean and std?

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.train:
            return (self.train_data[index, :], self.train_truth[index])
        else:
            return (self.test_data[index, :], self.test_truth[index])



############### TEST CODE #######################

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    dataset = SequentialMNIST()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for sequence, truth in dataloader:
        print(sequence.shape)
        plt.figure()
        plt.title(str(truth.item()))
        plt.imshow(sequence.view(28, 28), cmap='gray')
        plt.show()

################################################

