
from development.sp import sp
from development.so import so
from development.se import se
import torch
from torch import nn
from development.expm import expm
from development.nn import development_layer
from Nbody.utils import count_parameters


class RNN(nn.Module):
    def __init__(self, n_inputs=4, n_atoms=5, n_hid1=10, n_hid2=50, n_out=2, method='rnn'):
        super(RNN, self).__init__()
        # self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_out = n_out
        self.fc1_1 = nn.Linear(n_inputs, n_hid1)
        self.fc1_2 = nn.Linear(n_hid1, n_hid1)
        if method == 'rnn':
            self.rnn = nn.RNN(input_size=n_atoms*n_hid1,
                              hidden_size=n_hid2,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=n_atoms*n_hid1,
                               hidden_size=n_hid2,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)
        self.fc2_1 = nn.Linear(n_hid2, n_hid2)
        self.fc2_2 = nn.Linear(n_hid2, n_atoms * n_out)

    def forward(self, X):
        N, T, K, C = X.shape
        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)
        X = X.view(N, T, -1)
        X = self.rnn(X)[0][:, -1]
        X = nn.ReLU()(self.fc2_1(X))
        X = self.fc2_2(X)

        return X.view(N, K, -1)


class LSTM_development(nn.Module):
    def __init__(self, n_inputs=4, n_atoms=5, n_hid1=10, n_hid2=20, dev_input=8, n_out=4,
                 param=sp, triv=expm, method='lstm'):
        super(LSTM_development, self).__init__()
        self.n_inputs = n_inputs
        self.n_out = n_out
        self.fc1_1 = nn.Linear(n_inputs, n_hid1)
        self.fc1_2 = nn.Linear(n_hid1, n_hid1)
        self.n_atoms = n_atoms
        if method == 'rnn':
            self.rnn = nn.RNN(input_size=n_atoms*n_hid1,
                              hidden_size=n_hid2,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=n_atoms*n_hid1,
                               hidden_size=n_hid2,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)
        self.fc2_1 = nn.Linear(n_hid2, dev_input)
        self.dev = development_layer(
            input_size=dev_input, hidden_size=3, channels=n_atoms, param=param,
            triv=triv, return_sequence=False, complexification=False)

    def forward(self, X):
        N, T, K, C = X.shape
        input = X[:, -1, :, :2]  # N,K,2
        input = torch.cat((input, torch.ones_like(
            input)[..., :1]), dim=-1)
        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)
        X = X.view(N, T, -1)
        X = self.rnn(X)[0]
        X = self.fc2_1(X)
        output = []
        se2 = self.dev(X)
        for i in range(self.n_atoms):
            next = torch.bmm(
                se2[:, i, :, :], input[:, i].unsqueeze(2))[:, :2][:, :2]  # N,2,1
            output.append(next)

        out = torch.cat(output, 2).permute(0, 2, 1)

        return out


class LSTM_development_so(nn.Module):
    def __init__(self, n_inputs=4, n_atoms=5, n_hid1=10, n_hid2=20, dev_input=8, n_out=4,
                 param=so, triv=expm, method='lstm'):
        super(LSTM_development_so, self).__init__()
        self.n_inputs = n_inputs
        self.n_out = n_out
        self.fc1_1 = nn.Linear(n_inputs, n_hid1)
        self.fc1_2 = nn.Linear(n_hid1, n_hid1)
        self.n_atoms = n_atoms
        if method == 'rnn':
            self.rnn = nn.RNN(input_size=n_atoms*n_hid1,
                              hidden_size=n_hid2,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=n_atoms*n_hid1,
                               hidden_size=n_hid2,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)
        self.fc2_1 = nn.Linear(n_hid2, dev_input)
        self.dev = development_layer(
            input_size=dev_input, hidden_size=2, channels=n_atoms, param=param,
            triv=triv, return_sequence=False, complexification=False)

    def forward(self, X):
        N, T, K, C = X.shape
        input = X[:, -1, :, :2]

        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)
        X = X.view(N, T, -1)
        X = self.rnn(X)[0]
        X = self.fc2_1(X)
        output = []
        so2 = self.dev(X)
        for i in range(self.n_atoms):
            next = torch.bmm(
                so2[:, i, :, :], input[:, i].unsqueeze(2))[:, :2][:, :2]  # N,2,1
            output.append(next)

        out = torch.cat(output, 2).permute(0, 2, 1)

        return out


def get_model(config):
    in_channels = 4

    if config.model == 'LSTM':
        model_name = "%s_%s" % (config.dataset, config.model)
    else:
        model_name = "%s_%s_%s" % (config.dataset, config.model, config.param)

    model = {
        "Nbody_LSTM": lambda: RNN(
            n_inputs=in_channels,
            n_atoms=5,
            n_hid1=config.n_hid1,
            n_hid2=config.n_hid2,
            n_out=2,
            method='lstm'),
        "Nbody_LSTM_DEV_SE": lambda: LSTM_development(
            n_inputs=in_channels,
            n_atoms=5,
            n_hid1=config.n_hid1,
            n_hid2=config.n_hid2,
            dev_input=config.dev_input,
            n_out=4,
            param=se),
        "Nbody_LSTM_DEV_SO": lambda: LSTM_development_so(
            n_inputs=in_channels,
            n_atoms=5,
            n_hid1=config.n_hid1,
            n_hid2=config.n_hid2,
            dev_input=config.dev_input,
            n_out=4,
            param=so)}[model_name]()
    # print number parameters
    print("Number of parameters:", count_parameters(model))
    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    model.to(config.device)
    torch.backends.cudnn.benchmark = True

    return model
