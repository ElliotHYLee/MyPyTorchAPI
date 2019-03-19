import torch
class MahalanobisLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pr_x, x, chol_cov):
        md = self.MahalanobisLoss(pr_x, x, chol_cov)
        # ed = self.EuclideanLoss(pr_x, x)
        # mean_loss = torch.add(md, ed)
        return md

    def MahalanobisLoss(self, pr_x, x, chol_cov):
        error = pr_x-x
        Q = self.getCovMat(chol_cov)
        #print(Q)
        md = self.getMD(error, Q)
        normQ = self.norm(Q)
        #normQ = torch.add(normQ, 0)
        logQ = torch.log(normQ)
        md_loss = torch.add(md, logQ)
        return torch.mean(md_loss)

    def getMD(self, error, Q):
        invQ = torch.inverse(Q)
        md = error.unsqueeze(1).matmul(invQ)
        md = torch.matmul(md, error.unsqueeze(2))
        md = md.squeeze(2)
        return md

    def norm(self, Q):
        lin = Q.reshape(-1, 9)
        norm = torch.sum(lin, dim=1).unsqueeze(1)
        return norm

    def getCovMat(self, chol_cov):
        bn = chol_cov.shape[0]
        L = torch.zeros(bn, 3, 3, dtype=torch.float).cuda()
        LT = torch.zeros(bn, 3, 3, dtype=torch.float).cuda()
        index = 0
        for j in range(0, 3):
            for i in range(0, j + 1):
                L[:, j, i] = chol_cov[:, index]
                LT[:, i, j] = chol_cov[:, index]
                index += 1
        Q = torch.matmul(L, LT)
        return Q

    def EuclideanLoss(self, pr_x, x):
        error = torch.abs(pr_x - x)
        return torch.mean(error)

if __name__ == '__main__':
    import numpy as np
    x = np.array([[1, 2, 3, 4, 5, 6],
                  [2, 3, 4, 5, 6, 7]], dtype=np.float32)
    x = torch.from_numpy(x)
    L = torch.zeros(2, 3, 3, dtype=torch.float)
    LT = torch.zeros(2, 3, 3, dtype=torch.float)
    index = 0
    for j in range(0, 3):
        for i in range(0, j + 1):
            L[:, j, i] = x[:, index]
            LT[:, i, j] = x[:, index]
            index += 1

    print(x)
    print(L)
    Q = torch.matmul(L, LT)
    invQ = torch.inverse(Q)

    error = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    dd = error.unsqueeze(1).matmul(invQ)
    dd = torch.matmul(dd, error.unsqueeze(2))
    dd = dd.squeeze(2)
    print(dd)

    lin = Q.reshape(-1, 9)
    lin = torch.sum(lin, dim=1).unsqueeze(1)