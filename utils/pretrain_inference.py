import torch
from torch_geometric.loader import DataLoader as DataLoaderPyG
from transformers import TrainingArguments
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.utils_load import PCA_
import torch.optim as optim
import csv
import os
import pandas as pd
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from utils import loader,test_, train_
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

class PretrainInference():
    def __init__(self, hyp, device, model, loss, root_dir, task, type_model, name_dataset,
                 processor, audio_ms, target_sampling_rate=None, model_pretrain=None, feature_extractor=None):
        self.model = model
        self.model_pretrain = model_pretrain
        self.audio_ms = audio_ms
        self.root_dir = root_dir
        self.name_dataset = name_dataset
        self.task = task
        self.device = device
        self.processor = processor
        self.hyp = hyp
        self.loss_op = loss
        self.type_model = type_model
        self.feature_extractor = feature_extractor
        self.alpha = 0.4
        # self.optimizer = optim.Adam(model.parameters(), lr=hyp['learning_rate'], 
        # betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
        # self.optimizer = optim.SGD(model.parameters(), lr=hyp['learning_rate'], momentum=0.8)
        self.optimizer = optim.Adam(model.parameters(), lr=hyp['learning_rate'])
        self.data_collator = loader.DataCollatorCTCWithPadding(processor=self.processor, padding=True)
    
    def saves(self, name_dataset, label_test, pred_test, time_now, fold, classes_names): 
        
        if 'TorontoNeuroFace' in self.name_dataset:
            path_save_model = os.path.join(self.root_dir, 'logs', name_dataset, 'checkpoints', self.type_model, self.task, time_now, 'pretrain_' + str(fold) + '.pt')
        
        else:
            path_save_model = os.path.join(self.root_dir, 'logs', name_dataset, 'checkpoints', self.type_model, time_now, 'pretrain_' + str(fold) + '.pt')
        
        if self.hyp['save_model']:
            path_checkpoint = os.path.join(path_save_model)
            os.makedirs(os.path.dirname(path_checkpoint), exist_ok=True)
            torch.save(self.model.state_dict(), path_checkpoint)
        
        if 'TorontoNeuroFace' not in self.name_dataset: 
            path_confusion_matrix = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, self.task, time_now, 'confusion_matrix_' + str(fold)  + '.jpg')
            os.makedirs(os.path.dirname(path_confusion_matrix), exist_ok=True)
            confusion_matrix = metrics.confusion_matrix(label_test, pred_test)
            
            plt.figure(figsize=(10,10), dpi = 300)

            # label_font = {'size':'18'}
            sns.set(font_scale=2.5)
            ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cbar=False)

            # set x-axis label and ticks.
            ax.set_title('Test', fontsize=35, weight='bold', pad=40)

            ax.set_xlabel("Inference", fontsize=25, weight='bold', labelpad=22)
            ax.xaxis.set_ticklabels(classes_names, fontsize=20, weight='bold', rotation=45)

            # set y-axis label and ticks
            ax.set_ylabel("Label", fontsize=25, weight='bold', labelpad=22)
            ax.yaxis.set_ticklabels(classes_names, fontsize=20, weight='bold', rotation=45)

            plt.savefig(path_confusion_matrix, dpi=300)
            plt.close()
            
    def collate_fn(self, batch):
        dict_batch = {}
        for item in batch:
            dict_batch["speech"] = item[0]
            dict_batch["label"] = torch.LongTensor(item[1])
        return dict_batch 
    
    def save_epoch(self, epoch, num_fold, label_test, pred_test, label_train, pred_train, time_now, name_dataset, mode, loss_train, loss_test):
        
        if 'TorontoNeuroFace' in self.name_dataset:
            path_metrics = os.path.join(self.root_dir, 'logs', name_dataset,  'results', self.type_model, self.task, time_now, 'metrics_' + str(num_fold) + '.csv')
        else:
            path_metrics = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, time_now, 'metrics_' + str(num_fold) + '.csv')
        
        os.makedirs(os.path.dirname(path_metrics), exist_ok=True)
        acc_train = metrics.accuracy_score(label_train, pred_train)
        acc_test = metrics.accuracy_score(label_test, pred_test)
        
        with open(path_metrics, mode=mode) as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if(mode == "w"): writer.writerow(['Fold', 'Epoch', 'AccuracyTrain', 'AccuracyTest', 'LossTrain', 'LossTest'])
            writer.writerow([num_fold, epoch, acc_train, acc_test, loss_train, loss_test])
            
    def pretrain(self, num_fold, path_open, people_train, 
                 people_test, time_now, num_classes, name_dataset, 
                 classes_names, df_train, df_test):
        
        df_train_new, df_test_new = self.extrair_features(self.hyp['num_augmented'], path_open, people_train, people_test, df_train, df_test)
        
        
        if 'wav2vec' in self.type_model:
            train_data = DataLoader(df_train_new, collate_fn=self.data_collator.collate, batch_size=self.hyp['tam_batch'], shuffle=True)
            test_data = DataLoader(df_test_new, collate_fn=self.data_collator.collate, batch_size=1)
        
        elif 'cnn14' == self.type_model:
            train_data = DataLoader(df_train_new, batch_size=self.hyp['train_batch'], shuffle=True, drop_last=True)
            test_data = DataLoader(df_test_new, batch_size=self.hyp['test_batch'])  
        
        elif 'cnn' in self.type_model or 'Cnn' in self.type_model:
            train_data = DataLoader(df_train_new, batch_size=self.hyp['train_batch'], shuffle=True, drop_last=True)
            test_data = DataLoader(df_test_new, batch_size=self.hyp['test_batch'], drop_last=True)

        elif 'tcn' in self.type_model:
            train_data = DataLoader(df_train_new, batch_size=self.hyp['train_batch'], shuffle=True)
            test_data = DataLoader(df_test_new, batch_size=self.hyp['test_batch'])
            
        else:
            df_train_new = df_train_new.__getdata__()
            train_data = DataLoaderPyG(df_train_new, batch_size=self.hyp['train_batch'], shuffle=True, drop_last=True)
            
            df_test_new = df_test_new.__getdata__()
            test_data = DataLoaderPyG(df_test_new, batch_size=self.hyp['test_batch'], drop_last=True)
        
        mode_ = 'w'
        for epoch in range(self.hyp['epochs']):
            label_train, pred_train, loss_train = train_.train(self.type_model, self.model, self.optimizer, self.device, train_data, 
                                                num_fold, epoch, num_classes, self.loss_op, self.alpha, self.processor, self.hyp)
            
            label_test, pred_test, loss_test = test_.test(self.type_model, self.model, self.device, test_data, num_fold, epoch,
                                            num_classes, self.loss_op, self.processor, self.hyp)

            self.save_epoch(epoch, num_fold, label_test, pred_test, label_train, pred_train, time_now, name_dataset, mode_, loss_train, loss_test)
            mode_ = 'a'
            
        self.construct_acc_curve(name_dataset, time_now, num_fold)
        self.construct_loss_curve(name_dataset, time_now, num_fold)
        
        self.saves(name_dataset, label_test, pred_test, time_now, num_fold, classes_names)
    
    def extrair_features(self, num_augmented, path_audio, people_train, people_test, df_train, df_test):
        
        if 'TorontoNeuroFace' in self.name_dataset:
            
            df_train_new = loader.DatasetToronto(self.device, df_train, self.task, path_audio, self.type_model, num_augmented,
                                                 self.processor, self.hyp['sample_rate'], self.audio_ms, self.hyp['len_window'], 
                                                 self.hyp['stride'], self.model_pretrain, self.feature_extractor)
            
            df_test_new = loader.DatasetToronto(self.device, df_test, self.task, path_audio, self.type_model, 0, self.processor,
                                                self.hyp['sample_rate'], self.audio_ms, self.hyp['len_window'], self.hyp['stride'], 
                                                self.model_pretrain, self.feature_extractor)

        elif 'RAVDESS' in self.name_dataset:
            pca = None
            if 'gnn_decoder': pca = PCA_(self.hyp['pca'])
            
            df_train_new = loader.DatasetRAVDESS(self.device, path_audio, people_train, num_augmented, self.type_model, self.processor, self.hyp
                                            , self.model_pretrain, pca, train=True)
            df_test_new = loader.DatasetRAVDESS(self.device, path_audio, people_test, 0, self.type_model, self.processor, 
                                            self.hyp, self.model_pretrain, pca, train=False)
        
        elif self.name_dataset == 'CREMA-D':
            df_train_new = loader.DatasetCREMA(self.device, path_audio, people_train, num_augmented, 
                                               self.type_model, self.processor, self.hyp['sample_rate'], 
                                               self.audio_ms)
            df_test_new = loader.DatasetCREMA(self.device, path_audio, people_test, 0, self.type_model, 
                                              self.processor, self.hyp['sample_rate'], self.audio_ms)
            
        else:
            df_train_new = loader.Pretrain(num_augmented, people_train, path_audio, self.stride, self.len_window, self.connect_k_nodes)
            df_test_new = loader.Pretrain(num_augmented, people_test, path_audio, self.stride, self.len_window, self.connect_k_nodes)
        
        return df_train_new, df_test_new

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
    
    def construct_loss_curve(self, name_dataset, time_now, fold):
        
        if 'TorontoNeuroFace' in self.name_dataset:
            path_checkpoint = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, self.task, time_now, 'metrics_' + str(fold) + '.csv')
            path_loss_curve = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, self.task, time_now, 'loss_curve_' + str(fold)  + '.jpg')
        else:
            path_checkpoint = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, time_now, 'metrics_' + str(fold) + '.csv')
            path_loss_curve = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, time_now, 'loss_curve_' + str(fold)  + '.jpg')
            
        with open(path_checkpoint, mode='r') as f:
            reader = csv.reader(f)
            loss_train = []; loss_test = []; epochs = []
            for row in reader:
                if row[0] == 'Fold': continue
                loss_train.append(float(row[4]))
                loss_test.append(float(row[5]))
                epochs.append(int(row[1]))
                
        plt.figure(figsize=(10,10), dpi = 300)
        plt.plot(epochs, loss_train, label='Train', color='magenta', linewidth=2)
        plt.plot(epochs, loss_test, label='Test', color='mediumslateblue', linewidth=2)
        plt.legend(loc='upper right', fontsize=20)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Loss Curve', fontsize=20)
        os.makedirs(os.path.dirname(path_loss_curve), exist_ok=True)
        plt.savefig(path_loss_curve, dpi=300)
        plt.close()
        
    def construct_acc_curve(self, name_dataset, time_now, fold):
        if 'TorontoNeuroFace' in self.name_dataset:
            path_checkpoint = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, self.task, time_now, 'metrics_' + str(fold) + '.csv')
        else:
            path_checkpoint = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, time_now, 'metrics_' + str(fold) + '.csv')
        with open(path_checkpoint, mode='r') as f:
            reader = csv.reader(f)
            acc_train = []; acc_test = []; epochs = []
            for row in reader:
                if row[0] == 'Fold': continue
                acc_train.append(float(row[2]))
                acc_test.append(float(row[3]))
                epochs.append(int(row[1]))
        plt.figure(figsize=(10,10), dpi = 300)
        plt.plot(epochs, acc_train, label='Train', color='magenta', linewidth=2)
        plt.plot(epochs, acc_test, label='Test', color='mediumslateblue', linewidth=2)
        plt.legend(loc='lower right', fontsize=20)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Accuracy Curve', fontsize=20)
        path_acc_curve = os.path.join(self.root_dir, 'logs', name_dataset, 'results', self.type_model, time_now, 'acc_curve_' + str(fold)  + '.jpg')
        os.makedirs(os.path.dirname(path_acc_curve), exist_ok=True)
        plt.savefig(path_acc_curve, dpi=300)
        plt.close()
    
def construct_graph(root_dir, name_dataset, task, type_model, time_now, people_set):
    
    if 'TorontoNeuroFace' in name_dataset:
        path_files = os.path.join(root_dir, 'logs', name_dataset, 'results', type_model, task, time_now)
    else:
        path_files = os.path.join(root_dir, 'logs', name_dataset, 'results', type_model, time_now)
        
    all_csv_files = os.listdir(path_files)
    all_csv_files = [fname for fname in all_csv_files if fname.endswith('.csv')]
    all_csv_files = sorted(all_csv_files)
    all_csv_files = [fname for fname in all_csv_files if 'metrics' in fname]
    acc_test = []; name_person = [];
    for idx, file_csv in enumerate(all_csv_files):
        df = pd.read_csv(os.path.join(path_files, file_csv))
        acc_test.append(df.iloc[-1]['AccuracyTest'])
        name_person.append(people_set[idx])
    
    dict_ = {}
    dict_[type_model] = acc_test
    df_new = pd.DataFrame(dict_, index=name_person)
    # path_save = os.path.join(root_dir, 'logs', name_dataset, 'results', type_model, time_now)
    # os.makedirs(os.path.dirname(path_save, exist_ok=True))
    color1 = sns.color_palette('tab10')
    df_new.plot(kind='bar', color=[color1[9]])
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title(task + " - " + type_model)
    plt.ylim([0, 1.2])
    plt.legend(loc='upper center', ncol=4)
    plt.savefig(os.path.join(path_files, task + "_" + type_model + "_bar.png"))