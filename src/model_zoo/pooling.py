import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings

class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):
        # x: (n_samples, num_classes, num_tokens)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        # norm_att = torch.softmax(self.att(x), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=-1)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features,attention_mask):
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask==0]=-1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights*weights_mask*features, dim=1)
        return context_vector


class GeMText(nn.Module):
    def __init__(self, dim=1, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = ((x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p)).sum(self.dim)
        ret = (x/(attention_mask_expanded.sum(self.dim))).clip(min=self.eps)
        ret = ret.pow(1/self.p)
        return ret

class NLPPooling(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if self.pooling_name =="SED":
            self.pooler = AttBlock(self.in_features, self.out_features, activation='linear')
        elif self.pooling_name =="AttentionHead":
            self.pooler = AttentionHead(self.in_features, self.out_features)
        elif self.pooling_name not in ("CLS",''):
            self.pooler = eval(self.pooling_name)(**self.params)

        # print(f'Pooling: {self.pooling_name}')

    def forward(self, last_hidden_state, attention_mask):

        if self.pooling_name in ['MeanPooling','MaxPooling','MinPooling']:
            # Pooling between cls and sep / cls and sep embedding are not included
            # last_hidden_state = self.pooler(last_hidden_state[:,1:-1,:],attention_mask[:,1:-1])
            last_hidden_state = self.pooler(last_hidden_state,attention_mask)
        elif self.pooling_name=="CLS":
            # Use only cls embedding
            last_hidden_state = last_hidden_state[:,0,:]
        elif self.pooling_name=="GeMText":
            # Use Gem Pooling on all tokens
            last_hidden_state = self.pooler(last_hidden_state,attention_mask)
        elif self.pooling_name=='SED':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state = last_hidden_state*input_mask_expanded
            last_hidden_state = last_hidden_state.permute(0,2,1) # BS x Emb x token_size
            last_hidden_state,_,_ = self.pooler(last_hidden_state) # BS x Emb
        
        elif self.pooling_name=="AttentionHead":
            last_hidden_state = self.pooler(last_hidden_state,attention_mask)
        else:
            # No pooling
            last_hidden_state = last_hidden_state
            # print(f"{self.pooling_name} not implemented")
        return last_hidden_state 