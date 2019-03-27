import torch
import torch.nn

class BatchScalar33MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scalar, mat):
        s = scalar.unsqueeze(2)
        s = s.expand_as(mat)
        return s*mat

class GetIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bn):
        I = torch.eye(3, dtype=torch.float)
        if torch.cuda.is_available():
            I = I.cuda()
        I = I.reshape((1, 3, 3))
        I = I.repeat(bn, 1, 1)
        return I

class Batch33MatVec3Mul(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, mat, vec):
        vec = vec.unsqueeze(2)
        result = torch.matmul(mat, vec)
        return result.squeeze(2)

class GetSkew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dw):
        bn = dw.shape[0]
        skew = torch.zeros((bn, 3, 3), dtype=torch.float)
        if torch.cuda.is_available():
            skew = skew.cuda()
        skew[:, 0, 1] = -dw[:,2]
        skew[:, 0, 2] = dw[:,1]
        skew[:, 1, 2] = -dw[:,0]

        skew[:, 1, 0] = dw[:, 2]
        skew[:, 2, 0] = -dw[:, 1]
        skew[:, 2, 1] = dw[:, 0]
        return skew


class Mat33Vec3Mul(torch.nn.Module):
    def __init__(self, LSTM_input_size, LSTM_num_layer, LSTM_hidden_size,
                 fc_output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_input_size, hidden_size=LSTM_hidden_size,
                                   num_layers=LSTM_num_layer, batch_first=True,
                                   bidirectional=False)
        self.fc_lstm = nn.Sequential(nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                        nn.PReLU(),
                                        nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                        nn.PReLU(),
                                        nn.Linear(LSTM_hidden_size, fc_output_size))

        self.num_layers = LSTM_num_layer
        self.hiddenSize = LSTM_hidden_size
        self.num = 1

    def forward(self, x):
        bn = x.shape[0]
        x, (h, c) = self.lstm(x, self.init_hidden(bn))
        x = x.squeeze(0)
        x = self.fc_lstm(x)
        return x







