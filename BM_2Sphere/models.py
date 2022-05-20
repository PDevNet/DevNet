from tkinter import N
from turtle import forward
from development.sp import sp
from development.unitary import unitary
from development.so import so
from development.se import se
from torch import nn
from development.expm import expm, expm_1
from development.nn import development_layer
import torch
from .utils import count_parameters
from torch.nn import ModuleList
from baselines.trivialization.orthogonal import OrthogonalRNN
from baselines.trivialization.initialization import cayley_init_
import signatory

init = cayley_init_


class Exprnn(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_hidden2=10, n_outputs=10, mode=("dynamic", 100, 100)):
        super(Exprnn, self).__init__()
        #permute = np.random.RandomState(92916)
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.fc1_1 = nn.Linear(n_inputs, n_hidden1)
        self.fc1_2 = nn.Linear(n_hidden1, n_hidden1)
        self.rnn = OrthogonalRNN(
            n_hidden1, n_hidden2, initializer_skew=init, mode=mode, param=expm)
        self.fc2_1 = nn.Linear(n_hidden2, n_hidden2)
        self.fc2_2 = nn.Linear(n_hidden2, self.n_outputs)

    def forward(self, X):
        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)

        if isinstance(self.rnn, OrthogonalRNN):
            state = self.rnn.default_hidden(X[:, 0, ...])
        # print(state.shape)
        rnn_outs = []
        for input in torch.unbind(X, dim=1):
            # print(input.shape)
            out_rnn, state = self.rnn(input, state)
           # print(out_rnn.shape)
            rnn_outs.append(out_rnn.unsqueeze(1))
        X = torch.cat(rnn_outs, dim=1)
        # print(X.shape)

        #print('shape of state:', state.shape)

        X = self.fc2_1(X)
        out = self.fc2_2(X)
        # print(out.shape)
        return out


class development_model(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_hidden2=20, n_outputs=10, channels=1, param=sp, triv=expm):
        super(development_model, self).__init__()
        # self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.param = param
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden1)

        if self.param.__name__ == 'unitary':
            self.complex = True
            self.development = development_layer(
                input_size=n_inputs, hidden_size=n_hidden1, channels=1, param=param,
                triv=triv, return_sequence=False, complexification=True, include_inital=True)
            self.fc2_2 = nn.Linear(self.hidden2*self.hidden2,
                                   self.n_outputs).to(torch.cfloat)

        else:
            self.complex = False

            self.development1 = development_layer(
                input_size=n_hidden1, hidden_size=n_hidden1, channels=1, param=param,
                triv=triv, return_sequence=True, complexification=False)
            self.fc2 = nn.Linear(self.n_hidden1*self.n_hidden1, self.n_hidden2)

            self.development2 = development_layer(
                input_size=n_hidden2, hidden_size=n_outputs, channels=1, param=param,
                triv=triv, return_sequence=True, complexification=False)

    def forward(self, X):
        X = self.fc1(X)

        X = self.development1(X)
        X = torch.flatten(X, start_dim=2)
        X = self.fc2(X)
        #X = nn.ReLU()(X)
#       X = torch.flatten(X, start_dim=1)
        # print(X.shape)
        X = self.development2(X)

        out = X.squeeze(2)[:, :, :, -1]

        #X = torch.flatten(X, start_dim=1)
        #out = self.fc(X)

        return out


class LSTM_development(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_hidden2=10, n_outputs=10, channels=1,
                 dropout=0.3, param=sp, triv=expm_1, method='lstm'):

        super(LSTM_development, self).__init__()
        # self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.param = param
        self.fc1_1 = nn.Linear(n_inputs, n_hidden1)
        self.fc1_2 = nn.Linear(n_hidden1, n_hidden1)

        if method == 'rnn':
            self.rnn = nn.RNN(input_size=self.n_inputs,
                              hidden_size=self.n_hidden1,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=self.n_hidden1,
                               hidden_size=self.n_hidden2,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)
        self.fc2_1 = nn.Linear(n_hidden2, n_hidden2)
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
                input_size=n_hidden2, hidden_size=3, channels=1, param=param,
                triv=triv, return_sequence=True, complexification=False)
        #self.fc2_2 = nn.Linear(self.n_hidden2**2, self.n_outputs)

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        input = X
        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)
        X = self.rnn(X)[0]
        X = self.fc2_1(X)
        X = self.development(X)

#        X = torch.flatten(X, start_dim=1)
        # print(X.shape)

        out = X.squeeze(2)[:, :, :, -1]

 #       out = self.fc2_2(X)
        if self.complex:
            out = out.abs()
        else:
            pass

        return out


class RNN(nn.Module):
    def __init__(self, n_inputs=2, n_hidden1=10, n_hidden2=10, n_outputs=10, method='rnn'):
        super(RNN, self).__init__()
        # self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.fc1_1 = nn.Linear(n_inputs, n_hidden1)
        self.fc1_2 = nn.Linear(n_hidden1, n_hidden1)

        if method == 'rnn':
            self.rnn = nn.RNN(input_size=self.n_inputs,
                              hidden_size=self.n_hidden2,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=False)
        elif method == 'lstm':
            self.rnn = nn.LSTM(input_size=self.n_hidden1,
                               hidden_size=self.n_hidden2,
                               num_layers=1,
                               bias=True,
                               dropout=0,
                               batch_first=True,
                               bidirectional=False)

        # self.development = development_layer(C=self.n_inputs, Lambda=1, train_lambda=train_lambda, train_weights=False,
        # return_sequence=True,
        # method=self.method)
        self.fc2_1 = nn.Linear(n_hidden2, n_hidden2)
        self.fc2_2 = nn.Linear(n_hidden2, self.n_outputs)

    def forward(self, X):
        X = self.fc1_1(X)
        X = nn.ReLU()(X)
        X = self.fc1_2(X)
        X = nn.ReLU()(X)
        X = self.rnn(X)[0]
        X = nn.ReLU()(self.fc2_1(X))
        X = self.fc2_1(X)
        out = self.fc2_2(X)
       # print(out.shape)
        return out  # batch_size X n_output


class signature_model(nn.Module):
    def __init__(self, n_inputs=2, depth=4, n_outputs=10):
        super(signature_model, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.depth = depth
        self.sig_dim = signatory.signature_channels(
            channels=n_inputs, depth=depth)
        self.fc = nn.Linear(self.sig_dim, self.n_outputs)

    def forward(self, X):
        X = signatory.signature(X, depth=self.depth)
        out = self.fc(X)
        self.fc1 = nn.Linear(self.sig_dim, self.hidden1)

        self.fc2 = nn.Linear(self.sig_dim, self.n_outputs)

        return out.view(-1, self.n_outputs)


def get_model(config):
    print(config)

    in_channels = 2
    n_outputs = 3

    if config.model == 'development' or config.model == 'LSTM_development':
        model_name = "%s_%s_%s" % (config.dataset, config.model, config.param)
    else:
        model_name = "%s_%s" % (config.dataset, config.model)
    model = {
        "BM_2Sphere_LSTM": lambda: RNN(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=n_outputs,
            method='lstm'),
        "BM_2Sphere_development_orthogonal": lambda: development_model(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=n_outputs,
            param=so),
        "BM_2Sphere_development_real_symp": lambda: development_model(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_outputs=n_outputs,
            param=sp),
        "BM_2Sphere_LSTM_development_real_symp": lambda: LSTM_development(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=n_outputs, param=sp),
        "BM_2Sphere_LSTM_development_orthogonal": lambda: LSTM_development(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=n_outputs, param=so),
        "BM_2Sphere_Exprnn": lambda: Exprnn(
            n_inputs=in_channels,
            n_hidden1=config.n_hidden1,
            n_hidden2=config.n_hidden2,
            n_outputs=n_outputs)}[model_name]()
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
