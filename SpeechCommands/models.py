from development.sp import sp
from development.unitary import unitary
from development.so import so
from torch import nn
from development.expm import expm
from development.nn import development_layer
import torch
from SpeechCommands.utils import count_parameters
import signatory


class development_model(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_outputs=10, channels=1, param=sp, triv=expm):
        super(development_model, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_outputs = n_outputs
        self.param = param
        if self.param.__name__ == 'unitary':
            self.complex = True
            self.development = development_layer(
                input_size=n_hidden1, hidden_size=n_hidden1, channels=1, param=param,
                triv=triv, return_sequence=False, complexification=True)
            self.FC = nn.Linear(self.n_hidden1*self.n_hidden1,
                                self.n_outputs).to(torch.cfloat)

        else:
            self.complex = False

            self.development = development_layer(
                input_size=n_inputs, hidden_size=n_hidden1, channels=1, param=param,
                triv=triv, return_sequence=False, complexification=False)

            self.fc = nn.Linear(self.n_hidden1*self.n_hidden1, n_outputs)

    def forward(self, X):

        X = self.development(X)

        X = torch.flatten(X, start_dim=1)

        out = self.fc(X)
        if self.complex:
            out = out.abs()
        else:
            pass

        return out.view(-1, self.n_outputs)


class LSTM_development(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_hidden2=10, n_outputs=10, channels=1,
                 dropout=0.3, param=sp, triv=expm, method='lstm'):
        super(LSTM_development, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.param = param

        if method == 'rnn':
            self.rnn = nn.RNN(input_size=self.n_hidden1,
                              hidden_size=self.n_hidden1,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=self.n_inputs,
                               hidden_size=self.n_hidden1,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)

        if self.param.__name__ == 'unitary':
            self.complex = True
            self.development = development_layer(
                input_size=n_hidden1, hidden_size=n_hidden2, channels=1, param=param,
                triv=triv, return_sequence=False, complexification=True)
            self.FC = nn.Linear(self.n_hidden*self.n_hidden2,
                                self.n_outputs).to(torch.cfloat)

        else:
            self.complex = False

            self.development = development_layer(
                input_size=n_hidden1, hidden_size=n_hidden2, channels=1, param=param,
                triv=triv, return_sequence=False, complexification=False)

        self.fc = nn.Linear(self.n_hidden2**2, self.n_outputs)

    def forward(self, X):

        X = self.rnn(X)[0]

        X = self.development(X)

        X = torch.flatten(X, start_dim=1)

        out = self.fc(X)
        if self.complex:
            out = out.abs()
        else:
            pass

        return out.view(-1, self.n_outputs)


class RNN(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_outputs=10, method='rnn'):
        super(RNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = n_hidden1

        if method == 'rnn':
            self.rnn = nn.RNN(input_size=self.n_hidden1,
                              hidden_size=self.n_hidden1,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=self.n_inputs,
                               hidden_size=self.n_hidden1,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)

        self.fc = nn.Linear(self.n_hidden1, n_outputs)

    def forward(self, X):

        X = self.rnn(X)[0][:, -1]
        out = self.fc(X)

        return out  # batch_size X n_output


class signature_model(nn.Module):
    def __init__(self, n_inputs=20, depth=3, n_outputs=10):
        super(signature_model, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.depth = depth
        self.sig_dim = signatory.signature_channels(
            channels=n_inputs, depth=depth)
        print('siganture has feature dimension={}'.format(self.sig_dim))

        self.fc = nn.Linear(self.sig_dim, self.n_outputs)

    def forward(self, X):
        X = signatory.signature(X, depth=self.depth)
        out = self.fc(X)
        return out.view(-1, self.n_outputs)


def get_model(config):
    if config.mfcc:
        in_channels = 20
    else:
        in_channels = 1

    if config.model == 'DEV' or config.model == 'LSTM_DEV':
        model_name = "%s_%s_%s" % (config.dataset, config.model, config.param)
    elif config.model:
        model_name = "%s_%s" % (config.dataset, config.model)
    model = {
        "SpeechCommands_LSTM": lambda: RNN(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_outputs=10,
            method='lstm'),
        "SpeechCommands_DEV_SO": lambda: development_model(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_outputs=10,
            param=so),
        "SpeechCommands_DEV_Sp": lambda: development_model(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_outputs=10,
            param=sp),
        "SpeechCommands_signature": lambda: signature_model(
            n_inputs=in_channels,
            n_outputs=10),
        "SpeechCommands_LSTM_DEV_SO": lambda: LSTM_development(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=10, param=so, method='lstm'),
        "SpeechCommands_LSTM_DEV_Sp": lambda: LSTM_development(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=10, param=sp, method='lstm')}[model_name]()
    # print number parameters
    print("Number of parameters:", count_parameters(model))
    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)  # Required for multi-GPU
    model.to(config.device)
    torch.backends.cudnn.benchmark = True

    return model
