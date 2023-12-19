import struct
import matplotlib.pyplot as plt
import numpy as np


def test1():
    data_fp = f'/Users/reddyj/Desktop/workspace/nyu/courses/idls/project/SOSD/data/normal_200M_uint32'
    with open(data_fp, 'rb') as f:
        packed_data = f.read()
        num_records = struct.unpack('Q', packed_data[:8])[0]
        data = struct.unpack(f'{num_records}I', packed_data[8:])
    plt.hist(data[200000000-1000000:], bins=100)
    plt.show()

def test2():
    import torch

    torch.set_default_device("cuda:0")
    @torch.compile
    def test_fn(x):
      return torch.sin(x)

    a = torch.zeros(100)
    print(test_fn(a))
    
if __name__ == '__main__':
    test2()
