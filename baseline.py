from __basic__ import basic_info
import pandas as pd
import os
import numpy as np
import torch
print(torch.cuda.is_available())
import torch.nn as nn
from utils.pretrain_inference import PretrainInference, construct_graph
from utils.utils_load import read_data, choose_model, save_hyperparams, load_config_model
from utils.utils_augmentation import Mixup
from datetime import datetime
import random

def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    ROOT_DIR = os.getcwd().replace('\\', '/')
    fine_tune = False;
    datasets = ['RAVDESS']
    
    #choose the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = datasets[0]

    models = ['wav2vec']
    #choose the model
    type_model = models[0]
    
    if 'RAVDESS' in dataset_name:
        audio_ms = 5200 #5200
        classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        folds = [
                ['01', '04', '09', '22'],
                ['02', '05', '14', '15', '16'], 
                ['03', '06', '07', '13', '18'],
                ['10', '11', '12', '19', '20'], 
                ['08', '17', '21', '23', '24'],
                ]
    
    elif dataset_name == 'CREMA-D':
        audio_ms = 5200
        classes = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
        folds = [['1001', '1040', '1089', '1050', '1013', '1025', '1078', '1045', '1021', '1034', '1072', '1004', '1090']]
    
            
    out_channels = len(classes); 
    
    path_config = os.path.join(ROOT_DIR, 'config', 'config_' + type_model + '.json')
    
    hyp = load_config_model(path_config)
    hyp['out_channels'] = out_channels
    hyp['audio_ms'] = audio_ms
    
    
    time =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    use_graph = False; all_connected = True;
    mode = "w"
    
    save_hyperparams(hyp, time, type_model, ROOT_DIR, dataset_name)
    for fold_idx in range(0, len(folds)):
        if hyp['mixup'] == 'mixup':
            hyp['train_batch'] = hyp['train_batch']*2
            mixup_augmenter = Mixup(mixup_alpha=1.0, random_seed=fold_idx)
            hyp['mixup_lambda'] = mixup_augmenter.get_lambda(hyp['train_batch']).to(device)
            
        set_seed(2)
        df_train = None; df_test = None
    
        path_audios = os.path.join(ROOT_DIR, 'pretrain_dataset', dataset_name)
        get_files = np.array(os.listdir(path_audios))
            
        people_train = []; people_test = []   
        for file_name in get_files:
            
            if dataset_name == 'CREMA-D': num_item = file_name.split('_')[0]
            else: num_item = file_name.split('_')[-1]
            
            if(num_item in folds[fold_idx]): people_test.append(file_name)
            else: people_train.append(file_name)
            task = 'pretrain_' + str(folds[fold_idx]); 
            
        fold = folds[fold_idx]
        
        
        
        model, processor, target_sampling_rate = choose_model(hyp, type_model, fold_idx, device, dataset_name, 
                                                              classes, fine_tune=False, just_embedding=False, 
                                                              task=task, audio_ms=audio_ms)
        model_wav2vec = None; just_embedding = False; feature_extractor = None
        
        
        loss_op = nn.CrossEntropyLoss()
        
        model.to(device)
        # loss_op.to(device)

        audio_inference = PretrainInference(hyp, device, model, loss_op, ROOT_DIR, task, type_model, 
                                            dataset_name, processor, audio_ms, target_sampling_rate, 
                                            model_pretrain=model_wav2vec, feature_extractor=feature_extractor)
        
        audio_inference.pretrain(num_fold=fold_idx, path_open=path_audios,
                                people_train=people_train, people_test=people_test, time_now=time, 
                                num_classes=len(classes), name_dataset=dataset_name, 
                                classes_names=classes, df_train=df_train, df_test=df_test)
        mode = "a"
    
    if 'TorontoNeuroFace' in dataset_name:
        construct_graph(ROOT_DIR, dataset_name, task, type_model, time, folds)