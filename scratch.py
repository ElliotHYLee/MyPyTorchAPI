import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    arr = torch.from_numpy(arr).cuda()
    print(arr)
    print(arr.shape)

    arr = arr.cumsum(0)
    print(arr)
    print(arr.shape)




