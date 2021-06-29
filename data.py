import os
from scipy.io import wavfile
import torch
from torch._C import dtype
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torchaudio



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



def preload_audio(meta_file, data_dir, s_len, s_stride=None):
    """
    todo: implement stride management and dynamic stride
    :param meta_file:
    :param data_dir:
    :param s_len:
    :param s_stride:
    :return:
    """
    meta_data = pd.read_csv(meta_file)
    data_pairs = torch.empty(1, 2 * s_len)
    for clip_id in tqdm(range(len(meta_data))):
        noisy_clip, noisy_sr = torchaudio.load(os.path.join(data_dir, 'noisy', meta_data.iloc[clip_id, 0]))
        clean_clip, clean_sr = torchaudio.load(os.path.join(data_dir, 'clean', meta_data.iloc[clip_id, 0]))

        pad_len = s_len - noisy_clip.shape[1] % s_len

        if pad_len != 0:
            noisy_clip = torch.nn.functional.pad(noisy_clip, (0, pad_len), "constant", 0)
            clean_clip = torch.nn.functional.pad(clean_clip, (0, pad_len), "constant", 0)

        for t in range(int(noisy_clip.shape[1] / s_len)):
            start = t * s_len
            end = (t + 1) * s_len
            noisy_segment = noisy_clip[:, start:end]
            clean_segment = clean_clip[:, start:end]
            noisy_clean_pair = torch.cat((noisy_segment, clean_segment), dim=1)
            data_pairs = torch.cat((data_pairs, noisy_clean_pair))

    print(f'Created segmented dataset tensor with shape {data_pairs.shape}')

    return data_pairs



class Audio_dataset(torch.utils.data.Dataset):
    def __init__(self, data_file, segm_len):
            self.segm_dataset = torch.load(data_file)
            self.segm_len = segm_len

    def __len__(self):
        return self.segm_dataset.size()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        noisy_segm = self.segm_dataset[idx, 0:self.segm_len]
        clean_segm = self.segm_dataset[idx, self.segm_len:2*self.segm_len]

        if len(noisy_segm.size()) == 1:
                noisy_segm = torch.unsqueeze(noisy_segm, 0)
                clean_segm = torch.unsqueeze(clean_segm, 0)

        return noisy_segm, clean_segm



def generate_parsed_file(dataset):

    dataset_dir = os.path.join('dataset', dataset)
    parse_data(dataset+'_meta_file.csv', os.path.join(dataset_dir, 'noisy'))
    segm_dataset = preload_audio(dataset+'_meta_file.csv', dataset_dir, 48000)
    torch.save(segm_dataset, 'dataset_'+dataset+'.pt')

    train_dataset = Audio_dataset('dataset_'+dataset+'.pt', 48000)
    print('Dataset done')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    print('Dataloader done')
    for n, data in enumerate(train_dataloader):
        print(f'Elemento {n} - {data[0].shape} - {data[1].shape}')


if __name__ == '__main__':
    generate_parsed_file('test')