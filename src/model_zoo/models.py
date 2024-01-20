import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint
import torch.nn.functional as F
import gc

# from mmcv.cnn import bias_init_with_prob
from torchvision.ops import roi_align, nms

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
                 num_labels = 8,
                 config_path=None,
                 pretrained_path = None,
                 use_dropout=False,
                 use_gradient_checkpointing = False
                 ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) if not config_path else torch.load(config_path)

        self.use_dropout = use_dropout
        if not self.use_dropout:
            self.config.update(
                                {
                                    "hidden_dropout_prob": 0.0,
                                    "attention_probs_dropout_prob": 0.0,
                                }
                                    )

        self.backbone = AutoModel.from_pretrained(model_name,config=self.config) if not config_path else AutoModel.from_config(self.config)        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        
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
        x = self.backbone(b["input_ids"],b["attention_mask"]).last_hidden_state
        x = self.dropout(x)
        x = self.fc(x)
        x = aggregate_tokens_to_words(x, b['word_boxes'])
        # obj_pred = x[..., 0]
        # reg_pred = x[..., 1:3]
        # type_pred = x[..., 3:-3]
        # eff_pred = x[..., -3:]
        return x