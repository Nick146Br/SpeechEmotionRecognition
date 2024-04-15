from tqdm import tqdm
import torch
import copy
from sklearn import metrics
import statistics
from sklearn.metrics import confusion_matrix
from utils.utils_load import move_data_to_device, extract_features_cnn14
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def test(type_model, model, device, test_loader, fold_teste, epoch, num_classes, loss_op,
        processor=None, hyp=None):
    
    model.eval(); 
    test_losses = []; pred_teste = []; label_teste = []
    pbar = tqdm(total=len(test_loader.dataset), colour="magenta")
    pbar.set_description(f'Fold {fold_teste} - Epoch {epoch} - Testing')
    dicionario = {}
    for dado in test_loader:
        pred_vec = []

        with torch.cuda.amp.autocast():
            if type_model == 'cnn_X':
                x, y = dado
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                pred = model(x)
                
            elif 'decoder' in type_model:
                dado = dado.to(device)
                # dado_cnn14 = dado.cnn14
                # num_batchs = len(torch.unique(dado.batch))
                # dado_cnn14 = torch.reshape(dado_cnn14, (num_batchs, -1))
                # dado_cnn14 = extract_features_cnn14(dado_cnn14, device, hyp)
                pred = model(dado.x, dado.edge_index, dado.batch)
                y = dado.y
            
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
                    y = torch.FloatTensor(y).to(device)
                # y = move_data_to_device(y, device)
                y = y.to(device)
                pred = model(x)
                pred = pred['clipwise_output']
                
            elif type_model == 'cnn1D':
                x, y = dado
                x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            
            elif 'cnn' in type_model:
                x, y = dado
                x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                pred = model(x)
                
            elif 'wav2vec' in type_model:
                results = dado
                x = move_data_to_device(results['input_values'], device)
                y = move_data_to_device(results['labels'], device)
                x = torch.squeeze(x, dim=1)
                # attention_mask = move_data_to_device(attention_mask, device)
                pred =  model(input_values=x)
                pred = pred[0]
                
            elif 'tcn' in type_model:
                x, y = dado
                x = move_data_to_device(x, device)
                y = move_data_to_device(y, device)
                pred = model(x)    
            
            elif 'graphAudio' in type_model:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.batch)
                pred = pred.detach().cpu().numpy()
                y = dado.y
            
            elif 'last_try' in type_model:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.batch)
                y = dado.y
                
            else:
                dado = dado.to(device)
                pred = model(dado.x, dado.edge_index, dado.edge_attr, dado.batch)
                y = dado.y
                 
        loss = loss_op(pred, y)
        test_losses.append(loss.item())

        for i in range(0, len(pred)):
            pred_teste.append(int(torch.argmax(pred[i]).to("cpu")))
            
            if type_model == 'cnn14':
                if hyp['mixup']:
                    label_teste.append(int(torch.argmax(y[i]).to("cpu")))
                else:
                    label_teste.append(int(y[i].to("cpu")))
            else: 
                label_teste.append(int(y[i]))
                
            pbar.update(1)
            
    pbar.close()
    
    acc = metrics.accuracy_score(label_teste, pred_teste)
    test_losses = torch.Tensor(test_losses)
    test_losses = torch.mean(test_losses)
    print(f'Teste -> {acc}, Loss -> {test_losses}\n')

    return label_teste, pred_teste, float(test_losses)