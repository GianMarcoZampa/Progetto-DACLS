import os
from scipy.io import wavfile
import numpy as np


class Dataset():
    def __init__(self, dir):
        self.dir = os.path.join(dir)

    def get_data(self):
        data_x = []
        data_y = []
        
        for file in os.listdir(self.dir + '/noisy'):
            sample_rate, data = wavfile.read(self.dir + '/noisy/' + file)
            data_x.append(data.astype(np.float32))
        for file in os.listdir(self.dir + '/clean/'):
            sample_rate, data = wavfile.read(self.dir + '/clean/' + file)
            data_y.append(data.astype(np.float32))
    
        return np.asarray(data_x), np.asarray(data_y), sample_rate


def test(dir):
    dataset = Dataset(dir)
    x, y, sample_rate = dataset.get_data()
    print(len(x), len(y), sample_rate)
    print(type(x), type(y))


if __name__ == '__main__':
    test('dataset/test')