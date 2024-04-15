import torch
import torch.nn as nn
import heapq
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2PreTrainedModel, Wav2Vec2Model)
from torch_geometric.nn import GATConv, global_mean_pool, SSGConv


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config, just_embedding):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.just_embedding = just_embedding

    
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class GAT(nn.Module):
    def __init__(self, num_features, num_classes, num_heads, hidden_units, dropout):
        super(GAT, self).__init__()
        self.conv1 = SSGConv(num_features, hidden_units*2, alpha=0.3)
        self.conv2 = SSGConv(2*hidden_units, hidden_units*2, alpha=0.3)
        self.conv3 = SSGConv(2*hidden_units, hidden_units*2, alpha=0.3)
        self.dropout = nn.Dropout(dropout)
        # self.conv3 = SSGConv(2*hidden_units, num_classes, alpha=0.3)
        # self.conv1 = GATConv(num_features, num_classes, heads=num_heads, dropout=dropout)
        # self.conv2 = GATConv(hidden_units * num_heads, num_classes, heads=num_heads, dropout=dropout)
        # self.conv3 = GATConv(hidden_units * num_heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        # x = F.tanh(self.conv1(x, edge_index, edge_attr=edge_weight))
        x = F.tanh(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.tanh(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.tanh(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.3, training=self.training)
        
        return global_mean_pool(x, batch=batch)
    
class GATheadClassifier(nn.Module):
    
    def __init__(self, config, k_nearest, all_connected=False):
        super().__init__()
        self.k_nearest = k_nearest
        self.gat_model = GAT(config.hidden_size, config.hidden_size, 4, config.hidden_size, config.final_dropout)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.all_connected = all_connected
        
    def forward(self, features):
        
        num_batchs = features.shape[0]
        batch_features = list()
        
        num_nodes = features.shape[1]
        edges_to_list = list(); edge_weight_list = list(); edges_from_list = list(); batches_list = list()
        
        for batch in range(num_batchs):
            euclidean_distances_between_all_nodes = torch.cdist(features[batch], features[batch], p=2)
            #Create an adjacency list
            batches_list.append([batch]*num_nodes)
            # result2 = self.all_connected(euclidean_distances_between_all_nodes)
            if self.all_connected: edges_to, edges_from, edges_weight = self.f_all_connected(euclidean_distances_between_all_nodes, batch, num_nodes)
            else: edges_to, edges_from, edges_weight = self.prim(euclidean_distances_between_all_nodes, batch, num_nodes)
            
            # result1 = self.prim_np(euclidean_distances_between_all_nodes)
            # result2 = self.minimum_spanning_tree(euclidean_distances_between_all_nodes)
            # assert result1 == result2
            
            edges_to_list.append(edges_to)
            edges_from_list.append(edges_from)
            edge_weight_list.append(edges_weight)
                   
        batches_list = torch.tensor(batches_list).to(features.device)
        batches_list = batches_list.view(-1)
        x = features.view(-1, features.shape[-1]).to(features.device)
        #edge_index shape (2, num_edges)
        torch_edges_to = torch.tensor(edges_to_list).view(-1).to(features.device)
        torch_edges_from = torch.tensor(edges_from_list).view(-1).to(features.device)
        edge_index = torch.stack([torch_edges_to, torch_edges_from], dim=0)
        edge_weight = torch.tensor(edge_weight_list).view(-1).to(features.device)
        
        # x = self.dropout(x)
        x = self.gat_model(x, edge_index, edge_weight, batches_list)
        # x = x.view(num_batchs, -1, x.shape[-1])
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x
    
    def f_all_connected(self, distances, batch, num_nodes):
        #weights, heads, tails
        heads = []; tails = []; weights = []; edges = []; edges_weight = []

        rows, cols = torch.triu_indices(len(distances), len(distances[0]), offset=1)
        heads = rows
        tails = cols
        heads = heads + int(batch*num_nodes)
        tails = tails + int(batch*num_nodes)
        weights = distances[rows, cols]
        # edges = [heads, tails]
        edges_weight = weights
        heads = heads.tolist()
        tails = tails.tolist()
        edges_weight = edges_weight.tolist()
        # edges = torch.stack(edges, dim=0)
        
        return heads, tails, edges_weight
    
    def prim(self, distances, batch, num_nodes):
        #weights, heads, tails
        rows, cols = torch.triu_indices(len(distances), len(distances[0]), offset=1)
        heads = rows
        tails = cols
        weights = distances[rows, cols]

        w_indexes = torch.arange(len(weights))
        edges_in = torch.zeros(len(weights), dtype=torch.bool)

        rem_nodes = set(heads.tolist() + tails.tolist())
        has_nodes = set([heads[0].item()])
        rem_nodes.remove(heads[0].item())
        edges_to = list(); edges_from = list(); edges_weight = list()
        
        while len(rem_nodes) > 0:
            
            has_head = torch.isin(heads, torch.tensor(list(has_nodes)))
            has_tail = torch.isin(tails, torch.tensor(list(has_nodes)))
            between_edges = torch.not_equal(has_head, has_tail)

            w = weights[between_edges]
            w_i = w_indexes[between_edges]
            n_edge = w_i[torch.argmin(w).item()].item()
            n_node = tails[n_edge].item() if heads[n_edge].item() in has_nodes else heads[n_edge].item()
            has_nodes.add(n_node)
            rem_nodes.remove(n_node)
            edges_in[n_edge] = True

            edges_to.append(heads[n_edge].item() + batch*num_nodes)
            edges_from.append(tails[n_edge].item() + batch*num_nodes)
            edges_weight.append(weights[n_edge].item())
            
            edges_to.append(tails[n_edge].item() + batch*num_nodes)
            edges_from.append(heads[n_edge].item() + batch*num_nodes)
            edges_weight.append(weights[n_edge].item())
            

        return edges_to, edges_from, edges_weight
    
    def minimum_spanning_tree(self, adj_matrix):
        num_vertices = len(adj_matrix)
        
        mst = list(); edge_weight = list(); edge = list()
        heap = [(0, None, 0)]
        visited = set()
        
        while heap:
            weight, parent, current = heapq.heappop(heap)
            
            if current in visited: continue          
            visited.add(current)
            
            if parent is not None:
                mst.append((parent, current, weight))
                edge.append([parent, current])
                edge_weight.append(weight)
                
            for neighbor, weight in enumerate(adj_matrix[current]):
                if neighbor not in visited:
                    heapq.heappush(heap, (weight, current, neighbor))
                
        return edge, edge_weight
    
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, use_graph=False, all_connected=False, just_embedding=False):
        
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.all_connected = all_connected
        self.wav2vec2 = Wav2Vec2Model(config)
        if use_graph:
            self.classifier = GATheadClassifier(config, 5, all_connected)
        else:
            self.classifier = Wav2Vec2ClassificationHead(config, False)
        self.just_embeddings = just_embedding
        self.use_graph = use_graph
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self, input_values, attention_mask=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, labels=None,):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if self.just_embeddings:
            return hidden_states
        if not self.use_graph: 
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
            
        logits = self.classifier(hidden_states)

        loss = None
        
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

#TODO: Remove this function
def prim_np(self, distances):
    #weights, heads, tails
    heads = []; tails = []; weights = []; edges = []; edges_weight = []

    for i in range(len(distances)):
        for j in range(i+1, len(distances[i])):
            heads.append(i)
            tails.append(j)
            weights.append(distances[i][j])

    heads = torch.tensor(heads)
    tails = torch.tensor(tails)
    weights = torch.tensor(weights)

    w_indexes = torch.arange(len(weights))
    edges_in = torch.zeros(len(weights), dtype=torch.bool)

    rem_nodes = set(heads.tolist() + tails.tolist())
    has_nodes = set([heads[0].item()])
    rem_nodes.remove(heads[0].item())

    while len(rem_nodes) > 0:
        has_head = torch.isin(heads, torch.tensor(list(has_nodes)))
        has_tail = torch.isin(tails, torch.tensor(list(has_nodes)))
        between_edges = torch.not_equal(has_head, has_tail)

        w = weights[between_edges]
        w_i = w_indexes[between_edges]
        n_edge = w_i[torch.argmin(w).item()].item()
        n_node = tails[n_edge].item() if heads[n_edge].item() in has_nodes else heads[n_edge].item()
        has_nodes.add(n_node)
        rem_nodes.remove(n_node)
        edges_in[n_edge] = True

        edges.append([heads[n_edge].item(), tails[n_edge].item()])
        # edges.append([tails[n_edge].item(), heads[n_edge].item()])
        edges_weight.append(weights[n_edge].item())
        # edges_weight.append(weights[n_edge].item())

    return edges, edges_weight