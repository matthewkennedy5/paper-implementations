import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
from torchvision.transforms import ToTensor
import pdb

N_TEST = 10000
N_VAL = 2000

class SequentialMNIST(Dataset):
    """Contains the Sequential MNIST data from https://arxiv.org/abs/1504.00941.

    This dataset consists of MNIST images flattened and read pixel-by-pixel from
    left to right.
    """

    def __init__(self, train=True):
        """
        Inputs:
            train - True if train set, False for the test set.
        """
        self.train = train

        # Load the MNIST images and flatten into a new data tensor
        mnist_train_dataloader = torchvision.datasets.MNIST(root='', train=True,
                                                            download=True,
                                                            transform=ToTensor())
        mnist_test_dataloader = torchvision.datasets.MNIST(root='', train=False,
                                                           download=True,
                                                           transform=ToTensor())
        N = len(mnist_train_dataloader) + len(mnist_test_dataloader)
        sequence_len = 28 * 28  # MNIST image size
        data = torch.empty(N, sequence_len)
        truth = torch.LongTensor(N)
        for i, sample in enumerate(mnist_train_dataloader + mnist_test_dataloader):
            flattened_image = sample[0].view(sequence_len)
            data[i, :] = flattened_image
            truth[i] = sample[1]

        data = torch.unsqueeze(data, dim=2)
        # Partition the train and test data. This is a different partition than
        # in the original MNIST dataset, and is presumably different than the
        # partition the authors used.
        self.train_data = data[N_TEST:, :]
        self.test_data = data[:N_TEST, :]
        self.train_truth = truth[N_VAL:]
        self.test_truth = truth[:N_VAL]
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

    dataset = SequentialMNIST()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for sequence, truth in dataloader:
        plt.figure()
        plt.title(str(truth.item()))
        plt.imshow(sequence.view(28, 28), cmap='gray')
        plt.show()

################################################
