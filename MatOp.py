import torch

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








