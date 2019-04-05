import torch
from torch import nn
import pdb


class LSTM(nn.Module):
    """Standard LSTM with the option of adding batch normalization."""

    def __init__(self, input_size, hidden_size, output_size, batchnorm=False):
        """
        Inputs:
            input_size - Number of features in the input
            hidden_size - Number of features of h and c
            output_size - Number of output features
            batchnorm - Whether to use batch normalization as described in the
                paper.
        """
        super(LSTM, self).__init__()
        if batchnorm:
            self.cell = BatchNormLSTMCell(input_size, hidden_size)
        else:
            self.cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        """ Perform a forward pass of the LSTM over a sequence.

        Inputs:
            x - (batch_size, seq_len, features) tensor of input data.

        Returns:
            out - (batch_size, output_size) tensor of the output of the LSTM.
                  This model only has one output per sequence.
        """
        # x is (batch_size, seq_len, features)
        seq_len = x.size()[1]
        self.cell.zero_grad()
        hidden = None
        for i in range(seq_len):
            hidden = self.cell(x[:, i, :], hidden)
        out = self.hidden2output(hidden[0])
        return out


class BatchNormLSTMCell(nn.Module):
    """LSTM cell with batch normalization as described by the paper. """
    # TODO: Implement batch normalization for this.

    def __init__(self, input_size, hidden_size):
        """
        Inputs:
            input_size - Number of input features in x
            hidden_size - Number of hidden features (size of h and c)
        """
        super(BatchNormLSTMCell, self).__init__()
        self.Wh = nn.Parameter(torch.randn(4*hidden_size, hidden_size))
        self.Wx = nn.Parameter(torch.randn(4*hidden_size, input_size))
        self.b = nn.Parameter(torch.randn(1, 4*hidden_size))
        self.hidden_size = hidden_size

    def forward(self, x, hidden):
        """Perform a single timestep forward pass of the cell.

        Inputs:
            x - (batch_size, features) tensor of input data for the time step.
            hidden - Tuple containing (h_prev, c_prev).

        Returns:
            h - Next hidden state, (batch_size, hidden_size)
            c - Next cell state, (batch_size, hidden_size)
        """
        batch_size, features = x.size()
        if hidden is None:
            # Initiaze the first hidden state to be all zeros.
            h_prev = torch.zeros(batch_size, self.hidden_size)
            c_prev = torch.zeros(batch_size, self.hidden_size)
        else:
            h_prev, c_prev = hidden

        ifog = (torch.mm(h_prev, self.Wh.transpose(0, 1))
                + torch.mm(x, self.Wx.transpose(0, 1)) + self.b)
        # TODO: Simpler way of splitting this up?
        i, f, o, g = [ifog[:, n*self.hidden_size:(n+1)*self.hidden_size] for n in range(4)]
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


################## TEST CODE ##############

if __name__ == '__main__':

    # lstm = LSTM(input_size=1, hidden_size=100, output_size=1)
    # print(lstm(torch.zeros(500, 728, 1)))

    bn_lstm = LSTM(input_size=1, hidden_size=100, output_size=1, batchnorm=False)
    print(bn_lstm(torch.zeros(32, 728, 1)))
