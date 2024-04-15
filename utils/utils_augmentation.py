import librosa
import librosa.display
import random
import numpy as np
import torch

def adding_noise_to_audio(y_noise, sample_rate):
    
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    value = 0.005
    # print(f'value = {value}')
    noise_amp = value*np.random.uniform()*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
    
    return y_noise

def pitch_change(y_sample, sample_rate):
    
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    # print("pitch_change = ",pitch_change)
    y_pitch = librosa.effects.pitch_shift(y=y_sample.astype('float64'), 
                                      sr=sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return y_pitch

def hpps_effect(y_sample, sample_rate):
    
    y_hpss = librosa.effects.hpss(y_sample.astype('float64'))
    return y_hpss[1]

def value_augmentation(y_sample, sample_rate):
    
    dyn_change = np.random.uniform(low=1.5,high=3)
    y_aug = y_sample * dyn_change
    return y_aug

def time_shift(y_sample, sample_rate):
    
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
    start = int(y_sample.shape[0] * timeshift_fac)
    if (start > 0):
        y_sample = np.pad(y_sample,(start,0),mode='constant')[0:y_sample.shape[0]]
    else:
        y_sample = np.pad(y_sample,(0,-start),mode='constant')[0:y_sample.shape[0]]

    return y_sample

def shift_silent(y_sample, sample_rate):
    
    sampling=y_sample[(y_sample > 200) | (y_sample < -200)]
    shifted_silent = np.array(sampling.tolist()+np.zeros((y_sample.shape[0]-sampling.shape[0])).tolist(), dtype=np.float32)
    
    return shifted_silent

def Augmentation_audio(y_sample, sample_rate):
    
    y_changed = y_sample.copy()
    select_random_num = np.random.randint(0, 33)

    if select_random_num == 0:
        y_changed = time_shift(y_changed, sample_rate)
    elif select_random_num == 1:
        y_changed = pitch_change(y_changed, sample_rate)
    elif select_random_num == 2:
        y_changed = hpps_effect(y_changed, sample_rate)
    elif select_random_num == 3:
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 4:
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 7:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
    elif select_random_num == 8:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
    elif select_random_num == 9:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 10:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 11:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
    elif select_random_num == 12:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 13:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 14:
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 15:
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 16:
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 17:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
    elif select_random_num == 18:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 19:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 20:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 21:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 22:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 23:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 24:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 25:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 26:
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 27:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
    elif select_random_num == 28:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 29:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 30:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 31:
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
    elif select_random_num == 32:
        y_changed = time_shift(y_changed, sample_rate)
        y_changed = pitch_change(y_changed, sample_rate)
        y_changed = hpps_effect(y_changed, sample_rate)
        y_changed = adding_noise_to_audio(y_changed, sample_rate)
        y_changed = value_augmentation(y_changed, sample_rate)
        
    
    
    # y_changed = value_augmentation(y_changed, sample_rate)
    
    return y_changed

class MixupAugmentation():
    def __init__(self, device, alpha):
        super(MixupAugmentation, self).__init__()
        self.alpha = alpha
        self.device = device
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int_(W * cut_rat)
        cut_h = np.int_(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y, alpha=0.4, use_cuda=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def mixup_data(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        if self.device == 'cuda':
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        # return np.array(mixup_lambdas)
        #torch tensor
        return torch.FloatTensor(mixup_lambdas)
    

