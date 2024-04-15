from tqdm import tqdm
import torch
from sklearn import metrics
from utils.utils_load import move_data_to_device, extract_features_cnn14
from models.CNN14 import do_mixup

import torch

import numpy as np
import librosa
import sklearn
from utils.utils_augmentation import MixupAugmentation

def train(type_model, model, optimizer, device, train_loader, fold_teste,
          epoch, num_classes, loss_op, alpha=0.4, processor=None, hyp=None):
    
    model.train()
    train_losses = []; pred_train = []; label_train = [];
    pbar = tqdm(total=len(train_loader.dataset), colour="white")
    pbar.set_description(f'Fold {fold_teste} - Epoch {epoch} - Training')
    x_train = []; y_train = [];
    
    for dado in train_loader:  
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            
            if type_model == 'cnn_X':
                x, y = dado
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                pred = model(x)
            
            elif type_model == 'cnn14':
                X, y = dado
                # Spectrogram extractor
                x = extract_features_cnn14(X, device, hyp)
                
                if hyp['mixup']:
                    y_aux = []
                    for i in range(0, len(y)):
                        vec = [0 for _ in range(0, num_classes)]
                        vec[y[i]] = 1;
                        y_aux.append(torch.FloatTensor(vec))
                    
                    y = torch.stack(y_aux)
                    y = torch.FloatTensor(y)
                
                y = y.to(device)
                
                if hyp['mixup']:
                    pred = model(x, mixup_lambda=hyp['mixup_lambda'])
                    y_target = do_mixup(y, hyp['mixup_lambda'])
                else:
                    pred = model(x)
                pred = pred['clipwise_output']
                
            elif 'cnn' in type_model:
                x, y = dado
                if hyp['mixup']:
                    mixup_object = MixupAugmentation(device, alpha)
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                    x = move_data_to_device(x, device)
                    y = move_data_to_device(y, device)
                    
                    if hyp['mixup'] == 'cutmix': x_batch, y_batch_a, y_batch_b, lam = mixup_object.cutmix_data(x, y)
                    else: x_batch, y_batch_a, y_batch_b, lam = mixup_object.mixup_data(x, y)
                    
                    if type_model == 'cnn1D':
                        x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[3], x_batch.shape[2])
                    
                    pred = model(x_batch)
                    if type_model == 'cnn14':
                        pred = pred['clipwise_output']
                else:
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                    if type_model == 'cnn1D': x = x.reshape(x.shape[0], x.shape[3], x.shape[2])
                    x = x.to(device)
                    # x = move_data_to_device(x, device)
                    y = y.to(device)
                    pred = model(x)
                
            elif 'wav2vec' in type_model:
                results = dado
                x = move_data_to_device(results['input_values'], device)
                y = move_data_to_device(results['labels'], device)
                x = torch.squeeze(x, dim=1)
                # attention_mask = move_data_to_device(attention_mask, device)
                pred =  model(input_values=x)
                pred = pred['logits']
            
            elif 'tcn' in type_model:
                x, y = dado
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                pred = model(x)
            
            elif 'graphAudio' in type_model:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.batch)
                y = dado.y
            
            elif 'decoder' in type_model:
                dado = dado.to(device)
                # dado_cnn14 = dado.cnn14
                # num_batchs = len(torch.unique(dado.batch))
                # dado_cnn14 = torch.reshape(dado_cnn14, (num_batchs, -1))
                # dado_cnn14 = extract_features_cnn14(dado_cnn14, device, hyp)
                pred = model(dado.x, dado.edge_index, dado.batch)
                y = dado.y
                
            elif 'last_try' in type_model:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.batch)
                y = dado.y
            else:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.edge_attr, dado.batch)
                y = dado.y
        
        if 'wav2vec' in type_model:
            loss = loss_op(pred, y)
        elif 'cnn14' == type_model: 
            if hyp['mixup']: loss = loss_op(pred, y_target)
            else: loss = loss_op(pred, y)
        else: 
            if hyp['mixup']: loss = mixup_object.mixup_criterion(loss_op, pred, y_batch_a, y_batch_b, lam)
            else: loss = loss_op(pred, y)
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    
        for i in range(0, len(pred)):
            pred_train.append(int(torch.argmax(pred[i]).to("cpu")))
            if type_model == 'cnn14':
                if hyp['mixup']:
                    label_train.append(int(torch.argmax(y[i]).to("cpu")))
                else:
                    label_train.append(int(y[i].to("cpu")))
            else:
                label_train.append(int(y[i].to("cpu")))
                
        pbar.update(len(pred))

    pbar.close()
    train_losses = torch.Tensor(train_losses)
    train_losses = torch.mean(train_losses)
    
    acc_treino = metrics.accuracy_score(label_train, pred_train)
    print(f'Treino -> {acc_treino}, Loss -> {train_losses}\n')
    
    return label_train, pred_train, float(train_losses)