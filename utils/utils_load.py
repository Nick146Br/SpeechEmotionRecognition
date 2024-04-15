from models.wav2vec2 import Wav2Vec2ForSpeechClassification
from transformers import AutoConfig, Wav2Vec2Processor
import torch
import json
import os

def choose_model(hyp, type_model, fold, device, dataset_name, classes=None, fine_tune=False, just_embedding=False, task=None, audio_ms=None):
    processor = None; target_sampling_rate = 16000
    
    model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    pooling_mode = "mean"
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=hyp['out_channels'],
        label2id={label: i for i, label in enumerate(classes)},
        id2label={i: label for i, label in enumerate(classes)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
        use_graph=hyp['use_graph'],
        all_connected=hyp['all_connected'],
        just_embedding=just_embedding,
    )
    model.freeze_feature_extractor()
    model.to(device)

    return model, processor, target_sampling_rate

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.FloatTensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def save_hyperparams(hyperparemeters, time_now, type_model, ROOT_DIR, dataset_name):
    dict_params = hyperparemeters
    
    path_json_save = os.path.join(ROOT_DIR,'logs', dataset_name, 'checkpoints', type_model, time_now, 'hyperparams' + time_now + '.json')
    
    os.makedirs(os.path.dirname(path_json_save), exist_ok=True)
    
    with open(path_json_save, 'w') as fp:
        json.dump(dict_params, fp)
    

def load_config_model(path_config):
    
    with open(path_config, 'r') as f:
        hyp = json.load(f)
    
    return hyp