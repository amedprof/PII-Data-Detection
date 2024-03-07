import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint
import torch.nn.functional as F
import gc

# from mmcv.cnn import bias_init_with_prob
from torchvision.ops import roi_align, nms
from model_zoo.pooling import NLPPooling

def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep

class FeedbackModel(nn.Module):
    def __init__(self,
                 model_name,
                 num_labels = 9,
                 config_path=None,
                 pretrained_path = None,
                 use_dropout=False,
                 use_gradient_checkpointing = False,
                 pooling_params = {},
                 max_len = 512*4
                 ):
        super().__init__()

        self.num_labels = num_labels
        k = max_len//512 if (max_len//512)>0 else max_len//256
        self.max_len = max_len
        self.inner_len = 64*k if (max_len//512)>0 else 32*k
        self.edge_len = 384*k if (max_len//512)>0 else 192*k
        self.pooling_params = pooling_params
        self.pretrained_path = pretrained_path
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) if not config_path else torch.load(config_path)

        self.use_dropout = use_dropout
        if not self.use_dropout:
            self.config.update(
                                {
                                    "hidden_dropout_prob": 0.0,
                                    "attention_probs_dropout_prob": 0.0,


                                    # "hidden_size": 1536,
                                    # "initializer_range": 0.02,
                                    # "intermediate_size": 6144,
                                    # "max_position_embeddings": 512,
                                    # "relative_attention": true,
                                    # "position_buckets": 256,
                                    # "norm_rel_ebd": "layer_norm",
                                    # "share_att_key": true,
                                    # "pos_att_type": "p2c|c2p",
                                    # "layer_norm_eps": 1e-7,
                                    # "conv_kernel_size": 3,
                                    # "conv_act": "gelu",
                                    # "max_relative_positions": -1,
                                    # "position_biased_input": false,
                                    # "num_attention_heads": 32,
                                    # "attention_head_size": 64,
                                    # "num_hidden_layers": 48,
                                    # "type_vocab_size": 0,
                                    # "vocab_size": 128100

                                }
                                    )
            # print(self.config)

        self.backbone = AutoModel.from_pretrained(model_name,config=self.config,ignore_mismatched_sizes=True) if not config_path else AutoModel.from_config(self.config)        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        # print(self.max_len)
        if self.pretrained_path:
            try:
                self.load_from_cp()
            except:
                print('Loading failled !')
        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        # self.fc.bias.data[0].fill_(bias_init_with_prob(0.02))
        # self.fc.bias.data[3:-3].fill_(bias_init_with_prob(1 / num_label_discourse_type))
        # self.fc.bias.data[-3:].fill_(bias_init_with_prob(1 / num_label_effectiveness))

    def load_from_cp(self):
        print("Using Pretrained Weights")
        print(self.pretrained_path)
        state_dict = torch.load(self.pretrained_path, map_location=lambda storage, loc: storage)
        # print(state_dict.keys())
        del state_dict['fc.bias']
        del state_dict['fc.weight']

        if 'fc_seg.bias' in state_dict.keys():
            del state_dict['fc_seg.bias']
            del state_dict['fc_seg.weight']
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.deberta.', '')] = state_dict.pop(key)
        else:
            for key in list(state_dict.keys()):
                state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
        
        # print(self.backbone)
        self.backbone.load_state_dict(state_dict, strict=True)
        print('Loading successed !')

    def forward(self,b):
        
        B,L=b["input_ids"].shape # BS x token_size
        
        if L<=self.max_len:
            # print(L)
            # print(b["input_ids"].shape)
            x=self.backbone(input_ids=b["input_ids"],attention_mask=b["attention_mask"]).last_hidden_state
        else:
            # print(b["input_ids"].shape)
            # Slidding window
            segments=(L-self.max_len)//self.inner_len
            if (L-self.max_len)%self.inner_len>self.edge_len:
                segments+=1
            elif segments==0:
                segments+=1

            x=self.backbone(input_ids=b["input_ids"][:,:self.max_len],
                            attention_mask=b["attention_mask"][:,:self.max_len]).last_hidden_state
            # print(b["input_ids"].shape)
            for i in range(1,segments+1):
                start=self.max_len-self.edge_len+(i-1)*self.inner_len
                end=self.max_len-self.edge_len+(i-1)*self.inner_len+self.max_len
                end=min(end,L)
                x_next=b["input_ids"][:,start:end]
                mask_next=b["attention_mask"][:,start:end]
                x_next=self.backbone(input_ids=x_next,attention_mask=mask_next).last_hidden_state
                if i==segments:
                    x_next=x_next[:,self.edge_len:]
                else:
                    x_next=x_next[:,self.edge_len:self.edge_len+self.inner_len]
                x=torch.cat([x,x_next],1)
                # print(x.shape)


        # x = self.backbone(b["input_ids"],b["attention_mask"]).last_hidden_state
        # print(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.squeeze()[:6,:])
        x = aggregate_tokens_to_words(x, b['word_boxes'])
        # x = self.fc(x) # fullconnect before averaging embeddings
        # print(x.shape)
        
        # print(x.shape)
        # x = aggregate_tokens_to_words(x, b['word_boxes']).unsqueeze(0)
        
        pred = {"pred":x.squeeze()[:,:8]}
        if self.num_labels==9:
            # b['attention_mask'] = torch.ones(1,x.shape[1]).to(x.device)
            # y = self.pool_ly(x,b['attention_mask'])[:,-1]
            pred['y'] = x.squeeze()[:,-1]
        # print(pred)
        return pred 