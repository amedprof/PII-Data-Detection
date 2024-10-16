import os
import re
import gc
import sys
import json 
import random
import torch 
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

import codecs
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# import joblib
from text_unidecode import unidecode
 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig
from data.data_utils import to_gpu,to_np
from model_zoo.models import FeedbackModel,span_nms,aggregate_tokens_to_words

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


try:
    from faker import Faker
    fake = Faker(locale = ["fr_FR","fr_CA","en_US",'en_UK','de_DE','en_GB','en_IN','it_IT','fr_BE','es_ES'])
    # Faker.seed(0)
except:
    print('No faker installed')
    
try:
    from spacy.lang.en import English
    EN_TOK = English().tokenizer
except:
    print("No spacy")



# ======================================================================================== #
import spacy
from spacy import displacy
from pylab import cm, matplotlib
import os

colors = {
            '@NAME_STUDENT': '#8000ff',
            '@EMAIL': '#2b7ff6',
            '@USERNAME': '#2adddd',
            '@ID_NUM': '#80ffb4',
            '@PHONE_NUM': 'd4dd80',
            '@URL_PERSONAL': '#ff8042',
            '@STREET_ADDRESS': '#ff0000'
         }

# Define colors for placeholders
colors = {
    '@NAMES': '#8000ff',
    '@EMAIL': '#2b7ff6',
    '@USERNAME': '#2adddd',
    '@ID_NUM': '#80ffb4',
    '@PHONE_NUM': '#d4dd80',
    '@URL_PERSONAL': '#ff8042',
    '@STREET_ADDRESS': '#ff0000'
}

def visualize(full_text, offset_mapping):
    ents = []
    for offset in offset_mapping:
        ents.append({
            'start': int(offset["start_offset"]), 
            'end': int(offset["end_offset"]), 
            'label': offset["placeholder"]  # Keeping the label here for displacy, but it won't be shown in output
        })
    
    doc2 = {
        "text": full_text,
        "ents": ents,
    }

    options = {"ents": list(colors.keys()), "colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)

# ======================================================================================== #

def remove_double_spaces(text):
    # Use a regular expression to replace consecutive spaces with a single space
    # cleaned_text = re.sub(r'\s{2,}', ' | ', text)
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text

# ======================================================================================== #

def clean_text(text):
    text = text.replace(u'\x9d', u' ')
    text = remove_double_spaces(text)
    text = text.strip()
    return text
# ======================================================================================== #
def get_text_start_end(txt, s, search_from=0):
    txt = txt[int(search_from):]
    try:
        idx = txt.find(s)
        if idx >= 0:
            st = idx
            ed = st + len(s)
        else:
            raise ValueError('Error')
    except:
        res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
        if len(res):
            st, ed = res[0][0], res[0][1]
        else:
            m = SequenceMatcher(None, s, txt).get_opcodes()
            for tag, i1, i2, j1, j2 in m:
                if tag == 'replace':
                    s = s[:i1] + txt[j1:j2] + s[i2:]
                if tag == "delete":
                    s = s[:i1] + s[i2:]

            res = [(m.start(0), m.end(0)) for m in re.finditer(s, txt)]
            if len(res):
                st, ed = res[0][0], res[0][1]
            else:
                idx = txt.find(s)
                if idx >= 0:
                    st = idx
                    ed = st + len(s)
                else:
                    st, ed = 0, 0
    return st + search_from, ed + search_from

# ======================================================================================== #

def get_offset_mapping(full_text, tokens):
    offset_mapping = []

    current_offset = 0
    for token in tokens:
        start, end = get_text_start_end(full_text, token, search_from=current_offset)
        offset_mapping.append((start, end))
        current_offset = end

    return offset_mapping

# ======================================================================================== #

def tokenize_with_spacy(text,tok=EN_TOK):
    tokenized_text = tok(text)
    tokens = [token.text for token in tokenized_text]
    offset_mapping = [(token.idx,token.idx+len(token)) for token in tokenized_text]
    return {'tokens': tokens, 'offset_mapping': offset_mapping}

# ======================================================================================== #

def DemoText(text,tokenizer):
    
    
    text = clean_text(text.replace("\n\n"," | ").replace("\n"," [BR] "))
    
    sp_tokens = tokenize_with_spacy(text)
    sp_offset_mapping = get_offset_mapping(text, sp_tokens['tokens'])
#     sp_offset_mapping = sp_tokens['offset_mapping']
    
    hf_tokens = tokenizer(text, return_offsets_mapping=True)
    input_ids = torch.LongTensor(hf_tokens['input_ids'])
    attention_mask = torch.LongTensor(hf_tokens['attention_mask'])
    hf_offset_mapping = np.array(hf_tokens['offset_mapping'])
    
    
    num_tokens = len(input_ids)

    # token slices of words
    woff = np.array(sp_offset_mapping)
    toff = np.array(hf_offset_mapping)
    wx1, wx2 = woff.T
    tx1, tx2 = toff.T
    ix1 = np.maximum(wx1[..., None], tx1[None, ...])
    ix2 = np.minimum(wx2[..., None], tx2[None, ...])
    ux1 = np.minimum(wx1[..., None], tx1[None, ...])
    ux2 = np.maximum(wx2[..., None], tx2[None, ...])
    ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1)
#         assert (ious > 0).any(-1).all()

    word_boxes = []
    for i,row in enumerate(ious):
        inds = row.nonzero()[0]
        try:
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        except:
            word_boxes.append([-100, 0, -99, 1])
#                 err.append(i)

    word_boxes = torch.FloatTensor(word_boxes)
    
    return dict(word_boxes=word_boxes,text=text,
                sp_tokens=sp_tokens,sp_offset_mapping=sp_offset_mapping,
                input_ids = input_ids.unsqueeze(0),hf_offset_mapping=hf_offset_mapping,
                attention_mask = attention_mask.unsqueeze(0)
               )
    
# ======================================================================================== #
def reverse_transformation(text):
    # Replace ' | ' with double newlines
    text = text.replace(" | ", "\n\n")
    # Replace ' [BR] ' with single newlines
    text = text.replace(" [BR] ", "\n")
    return text
# ======================================================================================== #
def replace_text_elements(full_text, text_elements, text_elements_replaced, offset_mapping):
    # Convert full_text to a list of characters to handle replacements
    full_text_list = list(full_text)
    offset_adjustment = 0

    for i, (start, end) in enumerate(offset_mapping):
        replacement = text_elements_replaced[i]

        # Calculate the current positions considering the offset adjustment
        start += offset_adjustment
        end += offset_adjustment

        # Replace the text in full_text_list
        full_text_list[start:end] = list(replacement)

        # Calculate the length difference caused by the replacement
        length_difference = len(replacement) - (end - start)
        
        # Update the offset adjustment for subsequent replacements
        offset_adjustment += length_difference

    # Join the list back into a string
    return ''.join(full_text_list)

# ======================================================================================== #
def find_placeholders_offsets(text):
    placeholders = ['@NAMES', '@EMAIL', '@USERNAME', '@ID_NUM', '@PHONE_NUM', '@URL_PERSONAL', '@STREET_ADDRESS']
    offsets = []
    
    for placeholder in placeholders:
        for match in re.finditer(re.escape(placeholder), text):
            offsets.append({
                'placeholder': placeholder,
                'start_offset': match.start(),
                'end_offset': match.end()
            })
    
    return offsets
# ======================================================================================== #
def replace_successive_placeholders(text):
    placeholders = ['@NAMES', '@EMAIL', '@USERNAME', '@ID_NUM', '@PHONE_NUM', '@URL_PERSONAL', '@STREET_ADDRESS']
    
    for placeholder in placeholders:
        pattern = re.compile(r'(' + re.escape(placeholder) + r')(\s*\1)+')
        text = pattern.sub(placeholder, text)
    
    return text
# ======================================================================================== #

def text_anonym_fun(full_text, offset_mapping):
    ents = []
    for offset in offset_mapping:
        ents.append({
            'start': int(offset["start_offset"]), 
            'end': int(offset["end_offset"]), 
            'entity': offset["placeholder"][1:]  # Keeping the label here for displacy, but it won't be shown in output
        })
    
    doc2 = {
        "text": full_text,
        "entities": ents,
    }
    return doc2


# ======================================================================================== #
def make_one_predict(nets,tokenizer,input_text):

    dic = DemoText(input_text,tokenizer)
    
    predict = None
    for model in nets:
        preds = model(to_gpu(dic,device))['pred'].softmax(-1).detach().cpu().to(torch.float32)
        if predict is None:
            predict = preds
        else:
            predict+=preds
#     print(predict.shape)
    
    
    s,i = predict.max(-1)

    pred_df = pd.DataFrame({
                                    "token" : dic['sp_tokens']['tokens'],
                                    "token_id":np.arange(len(s)),
                                    "label" : i.numpy() ,
                                    "score" : np.float32(s.numpy())                                     })
    
    pred_df["label_next_e_prev"] = pred_df['label'].transform(lambda x: (x.shift(1)==x.shift(-1))*1)
    pred_df["label_next"] = pred_df['label'].transform(lambda x: x.shift(1))
    pred_df["label_next_e_prev"] = ((pred_df["label_next_e_prev"]==1) & (pred_df["label_next"]==6))*1
    pred_df["score_next"] = pred_df['score'].transform(lambda x: x.shift(1))
    pred_df.loc[pred_df["label_next_e_prev"]==1,"label"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"label_next"]
    pred_df.loc[pred_df["label_next_e_prev"]==1,"score"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"score_next"]


    threshold = 0.15
    if threshold>0:
        pred_df = pred_df[(pred_df.label!=7) & ((pred_df.score>threshold))].reset_index(drop=True)


    LABEL2TYPE = ('@NAMES','@EMAIL','@USERNAME','@ID_NUM', '@PHONE_NUM','@URL_PERSONAL','@STREET_ADDRESS','O')
    TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}
    LABEL2TYPE = {l: t for l, t in enumerate(LABEL2TYPE)}

    pred_df['label'] = pred_df['label'].map(LABEL2TYPE)


    full_text = dic['text']
    text_elements =  pred_df.token.values.tolist()
    text_elements_replaced = pred_df.label.values.tolist()
    offset_mapping = [dic['sp_offset_mapping'][i] for i in pred_df.token_id.values.tolist()]


    text_anonymized = replace_successive_placeholders(reverse_transformation(replace_text_elements(full_text, text_elements, 
                                                   text_elements_replaced, offset_mapping)))
    
    return text_anonymized

# ======================================================================================== #

import copy
from tqdm.auto import tqdm


CHECKPOINT_PATH = Path(r"/database/kaggle/PII/checkpoint/")
model_name = 'microsoft/deberta-v3-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
FOLD_NAME = "fold_msk_5_seed_42"
model_name = "deberta-v3-large"
exp_name = "2024-04-06--dv3l_cp_nboard_add_00_rep00_4608"
folder = str(CHECKPOINT_PATH/Path(fr'{FOLD_NAME}/{model_name}/{exp_name}'))

# CHECKPOINT_PATH = Path(r"../../checkpoint/")
# model_name = 'microsoft/deberta-v3-xsmall'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# FOLD_NAME = "fold_0_v1"
# model_name = "deberta-v3-xsmall"
# exp_name = "fold_0_v1"
# folder = str(CHECKPOINT_PATH/Path(fr'{model_name}/{exp_name}'))

# print(folder)
# ==== Loading Args =========== #
f = open(f'{folder}/params.json')
args = json.load(f)
args = SimpleNamespace(**args)
args.model['pretrained_tokenizer'] = f"{folder}/tokenizer"
args.model['model_params']['config_path'] = f"{folder}/config.pth"
args.model['pretrained_weights'] = None
args.model["model_params"]['pretrained_path'] = None
args.model["pooling_params"] = {}
args.model["pooling_params"]["pooling_name"] = 'MeanPooling'
args.model["pooling_params"]["params"] = {}

args.device = 0
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
checkpoints = [x.as_posix() for x in (Path(folder)).glob("*.pth") if f"config" not in x.as_posix()]

checkpoints = [x.as_posix() for x in (Path(folder)).glob("*.pth") if f"config" not in x.as_posix()]
# print(checkpoints[0])

nets = []
for cp in tqdm(checkpoints):
    net = FeedbackModel(**args.model["model_params"])
    net.load_state_dict(torch.load(cp, map_location=lambda storage, loc: storage))
    net = net.to(device)
    net.eval()
    nets.append(copy.deepcopy(net))

del net
gc.collect()


input_text = f"""Hi Team,

Quick update:

We’ve wrapped up phase 1, and the team led by Emma Taylor is on schedule to finish phase 2 by November 15, 2024. Our client, Tom Reed (tom.reed@clientcompany.com), is happy with the progress.

Our next meeting is set for Wednesday, October 18, 2024, at 2:00 PM in Room 203. Please check the attached file before then.

For urgent issues, you can reach me at (987) 654-3210.

Best,
Alex Brown
alex.brown@companymail.com
(987) 654-3210"""



import gradio as gr

if __name__ == "__main__":

    # text_anonymized = make_one_predict(nets,tokenizer,input_text)
    # offsets = find_placeholders_offsets(text_anonymized)
    # visualize(text_anonymized, offsets)

    def ner(text):
        text_anonymized = make_one_predict(nets,tokenizer,text)
        offsets = find_placeholders_offsets(text_anonymized)
        res = text_anonym_fun(text_anonymized, offsets)
        return res  

    demo = gr.Interface(ner,
                gr.Textbox(placeholder="Enter the text that you want to anonymize here..."), 
                gr.HighlightedText(),
                examples=[input_text]
                    )

    demo.launch(share=True)
