import os
from scipy.io import wavfile
import torch
from torch._C import dtype
import torch.nn as nn
import pandas as pd



def parse_data(meta_file, target_dir):
    """
    Args:
        meta_file (string): nome del file csv da salvare.
        target_dir (string): path alle cartelle 'noisy' o 'clean'.
    """
    mymeta = pd.DataFrame(columns=['filepath', 'len', 'samplerate'])
    wav_index = 0

    print('Parse dataset and generate meta file')

    for wf in os.listdir(target_dir):
        if wf[-3:] != 'wav':
            continue  # skip a loop if file is not a wavefile
        sr, sample = wavfile.read(os.path.join(target_dir, wf))
        mymeta.loc[wav_index] = ({'filepath': wf,
                                  'len': len(sample),
                                  'samplerate': sr})
        wav_index += 1

    mymeta.to_csv(meta_file, index=False, header=True)

    print('done')



class Audio_dataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, data_dir):
        self.meta_file = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.pad_length = self.max_length()

    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        wav_file = self.meta_file.iloc[idx, 0]

        # in questo modo sono sicuro che i due files caricati sono accoppiati correttamente perchè hanno lo stesso nome
        # il sample rate mi serve??
        sample_rate, noisy_sample = wavfile.read(os.path.join(self.data_dir, 'noisy', wav_file))
        sample_rate, clean_sample = wavfile.read(os.path.join(self.data_dir, 'clean', wav_file))

        noisy_sample = torch.from_numpy(noisy_sample)
        clean_sample = torch.from_numpy(clean_sample)
        # qui chiamo le trasformazioni pad e upsample per uniformare la lunghezza dei campioni

        if noisy_sample.size()[-1] < self.pad_length:
            noisy_sample = self.padding(noisy_sample)
            clean_sample = self.padding(clean_sample)
        
        return noisy_sample.type(torch.FloatTensor), clean_sample.type(torch.FloatTensor)

    def max_length(self):
        max_length = 0
        for i in range(self.__len__()):
            wav_file = self.meta_file.iloc[i, 0]
            _, sample = wavfile.read(os.path.join(self.data_dir, 'noisy', wav_file))
            max_length = max(max_length, len(sample))
        return max_length

    def padding(self, x):
        x = nn.functional.pad(x, pad=[0, self.pad_length-x.size()[-1]])
        return x



def test(dir):

    #parse_data(dir+'_noisy.csv', os.path.join(dir, 'noisy'))
    #parse_data(dir+'_clean.csv', os.path.join(dir, 'clean'))
    data = Audio_dataset(dir+'_noisy.csv', dir)
    print(data.__len__())
    data.__getitem__(123)
    print(data.max_length())


if __name__ == '__main__':
    test('dataset/test')