import torch
import os
from tqdm import tqdm
import numpy as np
import librosa
from utils.utils_augmentation import Augmentation_audio

def flatten_list(list_):
    
    flat_list = []
    for sublist in list_:
        for item in sublist:
            flat_list.append(item)
    
    return flat_list

def pad_trunc(audio, target_sampling_rate, max_ms=5200, name_model=None):
    sig, rate = librosa.load(audio)
    
    if name_model == 'TorontoNeuroFace_init':
        sig = sig[:int(len(sig)/3)]
    elif name_model == 'TorontoNeuroFace_mid':
        sig = sig[int(len(sig)/3):int(2*len(sig)/3)]
    elif name_model == 'TorontoNeuroFace_end':
        sig = sig[int(2*len(sig)/3):]
    
    sig = librosa.resample(sig, orig_sr=rate, target_sr=target_sampling_rate)
    
    if name_model != 'gnn':
        max_len = target_sampling_rate//1000 * max_ms
        sig_len = len(sig)
        
        if sig_len > max_len:
            sig = sig[:max_len]
        
        elif sig_len < max_len:
            max_offset = max_len - sig_len
            sig = np.pad(sig, (0, max_offset), "constant")
    
    return sig, target_sampling_rate

@torch.no_grad()  
class DatasetRAVDESS(torch.utils.data.Dataset):
    
    def __init__ (self, device, path_audio, people_set, num_augmented, name_model, 
                  processor, hyp, model_wav2vec=None, pca=None, train=True):
        
        self.labels = []; self.feat = []; self.max_size = 0;
        self.name_model = name_model;
        self.processor = processor; self.target_sampling_rate = hyp['sample_rate']
        self.mask_all_data = []
        
        # pbar = tqdm(total=len(people_set), colour="white")
        # pbar.set_description(f'Loading data')
        for idx, name in enumerate(people_set):
            path_audio_aux = os.path.join(path_audio, name)
            name_audios = os.listdir(path_audio_aux)
            
            pbar2 = tqdm(total=len(name_audios), colour="magenta")
            pbar2.set_description(f'Loading data from {name}')
            for audio in name_audios:
                path_audio_aux2 = os.path.join(path_audio_aux, audio)
                classe = int(audio.split('-')[2]) - 1
                    
                feat, labels = wav2vec(path_audio_aux2, num_augmented, classe, hyp['audio_ms'])
                self.feat.append(feat)
                self.labels.append(labels)
                pbar2.update(1)
                # break
            pbar2.close()
            # pbar.update(1)
            # break
        # pbar.close()
        
        self.feat = flatten_list(self.feat)
        self.labels = flatten_list(self.labels)
            # self.feat = flatten_list(self.feat)
            # self.labels = flatten_list(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getdata__(self):
        return self.feat
        
    def __getitem__(self, idx):
        
        results = self.processor(self.feat[idx], sampling_rate=self.target_sampling_rate)
        results["labels"] = self.labels[idx]
        return results
    

class DataCollatorCTCWithPadding():

    def __init__(self, processor, padding=True, max_length=None, max_length_labels=None, pad_to_multiple_of=None, pad_to_multiple_of_labels=None):
        
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels
        

    def collate(self, result):
        input_features = [{"input_values": feature["input_values"]} for feature in result]
        label_features = [feature["labels"] for feature in result]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

def wav2vec(path_audio, num_augmented, classe, audio_ms, sample_rate, name_model=None):
    feat = []; labels = [] 
    
    waveform, sampling_rate = pad_trunc(path_audio, target_sampling_rate=sample_rate, max_ms=audio_ms, name_model=name_model)
    speech = waveform.squeeze()
    
    for num in range(num_augmented+1):
        if num: y_sample = Augmentation_audio(speech, sampling_rate)
        else: y_sample = speech
        # encoding =  self.tokenizer(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        labels.append(classe)
        feat.append(y_sample)

    return feat, labels