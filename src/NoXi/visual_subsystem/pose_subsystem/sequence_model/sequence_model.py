from typing import Optional, Tuple

import torch
from torchinfo import summary


class Seq2One_model(torch.nn.Module):

    activation_functions_mapping = {'relu': torch.nn.ReLU,
                                    'sigmoid': torch.nn.Sigmoid,
                                    'tanh': torch.nn.Tanh,
                                    'softmax': torch.nn.Softmax,
                                    'linear': None
                                    }


    def __init__(self, input_shape:Tuple[int,...], LSTM_neurons:Tuple[int,...], dropout:Optional[float]=None,
                 dense_neurons:Tuple[int,...]=(128,),
                 dense_neurons_activation_functions:Tuple[str,...]=('relu',), dense_dropout:Optional[float]=None,
                 output_layer_neurons:int=5, output_activation_function:str='linear'):
        super(Seq2One_model, self).__init__()
        self.LSTM_neurons = LSTM_neurons
        self.input_shape = input_shape
        self.dropout = dropout
        self.dense_neurons = dense_neurons
        self.dense_neurons_activation_functions = dense_neurons_activation_functions
        self.dense_dropout = dense_dropout
        self.output_layer_neurons = output_layer_neurons
        self.output_activation_function = output_activation_function
        # build the model
        self._build_layers()


    def _build_layers(self):
        batch_size, seq_len, input_shape = self.input_shape
        # build LSTM layers
        self.LSTM_layers = torch.nn.ModuleList()
            # first layer
        self.LSTM_layers.append(torch.nn.LSTM(input_size=input_shape, hidden_size=self.LSTM_neurons[0], batch_first=True))
            # other layers
        for i in range(1,len(self.LSTM_neurons)):
            self.LSTM_layers.append(torch.nn.LSTM(input_size=self.LSTM_neurons[i-1], hidden_size=self.LSTM_neurons[i], batch_first=True))

        # Dense layers
        self.dense_layers = torch.nn.ModuleList()
            # first layer
        self.dense_layers.append(torch.nn.Linear(self.LSTM_neurons[-1], self.dense_neurons[0]))
        if self.activation_functions_mapping[self.dense_neurons_activation_functions[0]] is not None:
            self.dense_layers.append(self.activation_functions_mapping[self.dense_neurons_activation_functions[0]]())
            # other dense layers
        for i in range(1,len(self.dense_neurons)):
            self.dense_layers.append(torch.nn.Linear(self.dense_neurons[i-1], self.dense_neurons[i]))
            if self.activation_functions_mapping[self.dense_neurons_activation_functions[i]] is not None:
                self.dense_layers.append(self.activation_functions_mapping[self.dense_neurons_activation_functions[i]]())

        # output layer
        self.output_layer = torch.nn.Linear(self.dense_neurons[-1], self.output_layer_neurons)
        if self.activation_functions_mapping[self.output_activation_function] is not None:
            self.output_layer_activation = self.activation_functions_mapping[self.output_activation_function]()



    def forward(self, x):
        for layer in self.LSTM_layers:
            x, _ = layer(x)
        # take the last state of the LSTM
        x = x[:,-1,:]
        # flatten the tensor
        x = x.view(x.shape[0], -1)
        # go throught all Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        # output layer
        x = self.output_layer(x)
        if self.activation_functions_mapping[self.output_activation_function] is not None:
            x = self.output_layer_activation(x)
        return x


def initialize_weights(model):
    if type(model) in [torch.nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight.data)
    elif type(model) in [torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU]:
        torch.nn.init.xavier_uniform_(model.weight_hh_l0)
        torch.nn.init.xavier_uniform_(model.weight_ih_l0)



if __name__ == '__main__':
    model = Seq2One_model(input_shape=(32, 40, 256), LSTM_neurons=(512, 256, 128), dense_neurons=(256, 128, 64),
                          dense_neurons_activation_functions=('relu', 'relu', 'relu'), output_layer_neurons=5,
                          output_activation_function='linear')
    model.apply(initialize_weights)
    print(model)
    summary(model, input_size=(32, 40, 256))