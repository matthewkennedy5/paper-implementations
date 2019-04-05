import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import pdb

from mnist_dataloader import SequentialMNIST
from models import LSTM

N_VAL = 1000

class Trainer:

    def __init__(self, model, data, learning_rate=1e-3, momentum=0.9,
                 val_split=0.1, batch_size=32):
        self.train_data, self.val_data = self.make_dataloaders(data, val_split, batch_size)
        self.model = model
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, momentum=momentum)
        self.loss = nn.CrossEntropyLoss()

    def make_dataloaders(self, data, val_split, batch_size):
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        n_val = int(val_split * len(data))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_data = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        val_data = DataLoader(data, batch_size=batch_size, sampler=val_sampler)
        return train_data, val_data

    def train(self, epochs):
        iters_per_epoch = len(self.train_data)
        loss_history = []
        print('[INFO] Beginning training.')
        for epoch in range(epochs):
            print('Epoch', epoch+1)
            for X_batch, y_batch in self.train_data:
                self.optimizer.zero_grad()
                out = self.model(X_batch)
                loss_fn = self.loss(out, y_batch)
                loss_history.append(loss_fn.item())
                print(loss_fn.item())
                loss_fn.backward()
                self.optimizer.step()
        return loss_history

    def validate(self):
        pass
        # TODO
        # Iterate over validation set:
        #     perform forward pass
        #     record loss and accuracy
        # return average loss and accuracy

# run method running the trainer using the two models.

if __name__ == '__main__':
    lstm = LSTM(input_size=1, hidden_size=100, output_size=10)
    data = SequentialMNIST(train=True)
    trainer = Trainer(lstm, data, learning_rate=1e-4, batch_size=32)
    trainer.train(1)
