import re
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.data_utils import clean_text,get_start_end,get_offset_mapping,get_start_end_offset

from tqdm.auto import tqdm


LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}
ID_TYPE = {"0-0":0,"0-1":1,
           "1-0":2,"1-1":3,
           "2-0":4,"2-1":5,
           "3-0":6,"3-1":7,
           "4-0":8,"4-1":9,
           "5-0":10,"5-1":11,
           "6-0":12,"6-1":13
          }

ID_NAME = {"0-0":"B-NAME_STUDENT","0-1":"I-NAME_STUDENT",
           "1-0":"B-EMAIL","1-1":"I-EMAIL",
           "2-0":"B-USERNAME","2-1":"I-USERNAME",
           "3-0":"B-ID_NUM","3-1":"I-ID_NUM",
           "4-0":"B-PHONE_NUM","4-1":"I-PHONE_NUM",
           "5-0":"B-URL_PERSONAL","5-1":"I-URL_PERSONAL",
           "6-0":"B-STREET_ADDRESS","6-1":"I-STREET_ADDRESS",
           "7-0":"O","7-1":"O"
          }


## =============================================================================== ##
class FeedbackDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 mask_prob=0.0,
                 mask_ratio=0.0
                 ):
        
        self.tokenizer = tokenizer
        self.is_train = "labels" in df.columns
        if len(self.tokenizer.encode("\n\n"))==2:
            print("Warning : n SEP will be replace by | ")
            df["full_text"] = df['full_text'].transform(lambda x:x.str.replace("\n\n"," | "))
            df["tokens"] = df['tokens'].transform(lambda x:[str(i).replace("\n\n"," | ") for i in x])

        self.df = self.prepare_df(df)

        print(f'Loaded {len(self)} samples.')

        assert 0 <= mask_prob <= 1
        assert 0 <= mask_ratio <= 1
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df = self.df.iloc[index]
        text = df['full_text']
        text_id = df['document']
        labels = [] if not self.is_train else df['labels']
        # has_label = -1 if not self.is_train else df['has_label']
        
        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
#         offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = np.array(df['offset'])
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1)
#         assert (ious > 0).any(-1).all()

        word_boxes = []
#         err = []
        for i,row in enumerate(ious):
            inds = row.nonzero()[0]
            try:
                word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
            except:
                word_boxes.append([-100, 0, -99, 1])
#                 err.append(i)
                
        word_boxes = torch.FloatTensor(word_boxes)

        # word slices of ground truth spans
        gt_spans = []        
        for i,label in enumerate(labels) :
#             if i not in err:
            # gt_spans.append([i,TYPE2LABEL[label.split('-')[1] if label!="O" else "O"],0 if label.split('-')[0]=="B" else 1,has_label])
            gt_spans.append([i,TYPE2LABEL[label.split('-')[1] if label!="O" else "O"],0 if label.split('-')[0]=="B" else 1])
            
        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(
                    text_id=text_id,
                    # text=text,
                    tokens = df['tokens'],
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)
    
    def prepare_df(self,test_df):
        test_df['full_text'] = test_df['full_text'].transform(clean_text)        
        test_df['tokens'] = test_df['tokens'].transform(lambda x:[clean_text(i) for i in x])
        test_df['offset'] = test_df.apply(get_start_end_offset('tokens'),axis=1)

        if "labels" in test_df.columns:
            for name in LABEL2TYPE:
                test_df[name] = test_df['labels'].transform(lambda x:len([i for i in x if i.split('-')[-1]==name ]))

            test_df['has_label'] = (test_df['labels'].transform(lambda x:len([i for i in x if i!="O" ]))>0)*1
#         test_df['nb_labels'] = test_df['labels'].transform(lambda x:len([i for i in x if i!="O" ]))
        return test_df
    
    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)
    
## =============================================================================== ##
class CustomCollator(object):
    def __init__(self, tokenizer, model):
        self.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (model.config.attention_window
                                     if isinstance(
                                         model.config.attention_window, int)
                                     else max(model.config.attention_window))
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        tokens = sample['tokens']
        # text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    # text=text,
                    tokens = tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)