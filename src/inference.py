import argparse
import json 
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from tqdm.auto import tqdm
import re
import os
import gc
import random
import torch 

from train_utils import inference_step


 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig
 
from data.data_utils import to_gpu,to_np
from data.dataset import FeedbackDataset,CustomCollator
from torch.utils.data import DataLoader

from model_zoo.models import FeedbackModel,span_nms,aggregate_tokens_to_words
from metrics_loss.metrics import score_feedback,score,pii_fbeta_score_v2,compute_metrics,compute_metrics_new
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup

from sklearn.metrics import log_loss 
from tqdm.auto import tqdm

from utils.utils import count_parameters
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

data_path = Path(r"/database/kaggle/PII/data")
CHECKPOINT_PATH = Path(r"/database/kaggle/PII/checkpoint")


from datetime import date



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

def inference_blendings(df,folders,bs=1,folds=[0],selected_device=0,max_len=4096):
    

    doc_ids = []
    tokens = []
    tokens_v = []
    predictions = None
    gt_df = []

    for gi,folder in enumerate(folders):
        
        # ==== Loading Args =========== #
        f = open(f'{folder}/params.json')
        args = json.load(f)
        args = SimpleNamespace(**args)
        args.val_loader['batch_size'] = bs
        args.model['pretrained_tokenizer'] = f"{folder}/tokenizer"
        args.model['model_params']['config_path'] = f"{folder}/config.pth"
        args.model['pretrained_weights'] = None
        args.model["model_params"]['pretrained_path'] = None
        args.model["model_params"]['max_len'] = max_len
        args.data['params_valid'] = {"add_text_prob":0,
                                        "replace_text_prob":0,
                                        "use_re":False
                                        }
        
        args.device = selected_device
        f.close()
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        
        # ==== Loading dataset =========== #
        tokenizer = AutoTokenizer.from_pretrained(args.model["model_params"]['model_name'])
        valid_dataset = eval(args.dataset)(df,tokenizer,**args.data["params_valid"])
        
        
        
        # ==== Loading checkpoints =========== #
        checkpoints = [x.as_posix() for x in (Path(folder)).glob("*.pth") if f"config" not in x.as_posix()]
        checkpoints = [ x for x in checkpoints if any([f"fold_{fold}" in x for fold in folds])]
        
        weights = [1/len(checkpoints)]*len(checkpoints)
    
    
        # ==== Loop Inference =========== #
        for j,(checkpoint,weight) in enumerate(zip(checkpoints,weights)):
            
            net = FeedbackModel(**args.model["model_params"])
            net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
            net = net.to(device)
            net.eval()
            
            collator = CustomCollator(tokenizer,net)
            val_loader = DataLoader(valid_dataset,**args.val_loader,collate_fn=collator)
        

            
            preds = []
            with torch.no_grad():
                for data in tqdm(val_loader):
                    data = to_gpu(data, device)
                    
                    pred = net(data)['pred']
                    preds.append(pred.detach().cpu().to(torch.float32))
    # #                 pred  = pred.softmax(-1)
                    
                    
                    if j==0 and gi==0:
                    
                        doc_ids+=[data['text_id']]*pred.shape[0]
                        tokens+=np.arange(pred.shape[0]).tolist()
                        tokens_v += data['tokens']
                        data = to_np(data)
                        gt = pd.DataFrame({
                                        "document":data['text_id'],
                                        "token":np.arange(pred.shape[0]),
                                        "label":data["gt_spans"][:,1],
                                        "I":data["gt_spans"][:,2],
                                        })
                        gt_df.append(gt)

        
        
        
        if predictions is not None:
            # predictions = torch.cat([torch.max(predictions[:, :-1], torch.cat(preds,dim=0)[:, :-1]),
            #                         torch.min(predictions[:, -1:], torch.cat(preds,dim=0)[:, -1:])],dim=-1)
            
            predictions+= torch.cat(preds,dim=0)#*weight
        else:
            predictions = torch.cat(preds,dim=0)#*weight
            
#         if predictions is not None:
# #             predictions = torch.max(predictions,torch.cat(preds,dim=0))
#             predictions+= torch.cat(preds,dim=0)*weight
#         else:
#             predictions = torch.cat(preds,dim=0)*weight
#             predictions+= torch.cat(preds,dim=0)*weight
#         print(predictions.shape)
        print(checkpoint)
    predictions = predictions.softmax(-1)
    s,i = predictions.max(-1)
    pred_df = pd.DataFrame({"document":doc_ids,
                                "token" : tokens,
                                "tokens":tokens_v,
                                "label" : i.numpy() ,
                                "score" : s.numpy() ,
#                                  "o_score":predictions[:,-1].numpy()
                                })
    
    # ==== Loop Inference =========== #
    del valid_dataset
    del val_loader
    del net
    # del s,i
    del predictions

    gc.collect()
    

    gt_df = pd.concat(gt_df,axis=0).reset_index(drop=True)
    gt_df = gt_df[gt_df.label!=7].reset_index(drop=True)
    gt_df['labels'] = gt_df['label'].astype(str)+'-'+gt_df['I'].astype(str)
    gt_df["label_gt"] = gt_df["labels"].map(ID_TYPE).fillna(0).astype(int)
    gt_df['row_id'] = np.arange(len(gt_df))

    
    
    return pred_df , gt_df



def inference_steps(df,folder,bs=1,folds=[0],device=0,max_len=4096,kaggle=False):
        
    # ==== Loading Args =========== #
    f = open(f'{folder}/params.json')
    args = json.load(f)
    args = SimpleNamespace(**args)
    args.val_loader['batch_size'] = bs
    args.model['pretrained_tokenizer'] = f"{folder}/tokenizer"
    args.model['model_params']['config_path'] = f"{folder}/config.pth"
    args.model['pretrained_weights'] = None
    args.model["model_params"]['pretrained_path'] = None
    args.model["model_params"]['max_len'] = max_len
    args.data['params_valid'] = {"add_text_prob":0,
                                        "replace_text_prob":0,
                                        "use_re":False
                                        }
    
    args.device = device
    f.close()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # ==== Loading dataset =========== #
    tokenizer = AutoTokenizer.from_pretrained(args.model["model_params"]['model_name'])
    valid_dataset = eval(args.dataset)(df,tokenizer,**args.data["params_valid"])
    
    
    
    # ==== Loading checkpoints =========== #
    checkpoints = [x.as_posix() for x in (Path(folder)).glob("*.pth") if f"config" not in x.as_posix()]
    checkpoints = [ x for x in checkpoints if any([f"fold_{fold}" in x for fold in folds])]
    
    weights = [1/len(checkpoints)]*len(checkpoints)
    
    
    # ==== Loop Inference =========== #
    doc_ids = []
    tokens = []
    tokens_v = []
    predictions = None
    gt_df = []
    for j,(checkpoint,weight) in enumerate(zip(checkpoints,weights)):
        
        net = FeedbackModel(**args.model["model_params"])
        net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
        net = net.to(device)
        net.eval()
        
        collator = CustomCollator(tokenizer,net)
        val_loader = DataLoader(valid_dataset,**args.val_loader,collate_fn=collator)
    

        
        preds = []
        with torch.no_grad():
            for data in tqdm(val_loader):
                data = to_gpu(data, device)
                
                pred = net(data)['pred']
                preds.append(pred.detach().cpu().to(torch.float32))
# #                 pred  = pred.softmax(-1)
                
                
                if j==0 and not kaggle:
                
                    doc_ids+=[data['text_id']]*pred.shape[0]
                    tokens+=np.arange(pred.shape[0]).tolist()
                    tokens_v += data['tokens']
                    data = to_np(data)
                    gt = pd.DataFrame({
                                    "document":data['text_id'],
                                    "token":np.arange(pred.shape[0]),
                                    "label":data["gt_spans"][:,1],
                                    "I":data["gt_spans"][:,2],
                                    })
                    gt_df.append(gt)

        
        
        
        if predictions is not None:
            # predictions = torch.cat([torch.max(predictions[:, :-1], torch.cat(preds,dim=0)[:, :-1]),
            #                         torch.min(predictions[:, -1:], torch.cat(preds,dim=0)[:, -1:])],dim=-1)
            
            predictions+= torch.cat(preds,dim=0)*weight
        else:
            predictions = torch.cat(preds,dim=0)#*weight
            
#         if predictions is not None:
# #             predictions = torch.max(predictions,torch.cat(preds,dim=0))
#             predictions+= torch.cat(preds,dim=0)*weight
#         else:
#             predictions = torch.cat(preds,dim=0)*weight
#             predictions+= torch.cat(preds,dim=0)*weight
#         print(predictions.shape)
        # print(checkpoint)
    predictions = predictions.softmax(-1)
    s,i = predictions.max(-1)
    pred_df = pd.DataFrame({"document":doc_ids,
                                "token" : tokens,
                                "tokens":tokens_v,
                                "label" : i.numpy() ,
                                "score" : s.numpy() ,
#                                  "o_score":predictions[:,-1].numpy()
                                })
    
    # ==== Loop Inference =========== #
    del valid_dataset
    del val_loader
    del net
    # del s,i
    del predictions

    gc.collect()
    
    if not kaggle:
        gt_df = pd.concat(gt_df,axis=0).reset_index(drop=True)
        gt_df = gt_df[gt_df.label!=7].reset_index(drop=True)
        gt_df['labels'] = gt_df['label'].astype(str)+'-'+gt_df['I'].astype(str)
        gt_df["label_gt"] = gt_df["labels"].map(ID_TYPE).fillna(0).astype(int)
        gt_df['row_id'] = np.arange(len(gt_df))

        return pred_df , gt_df
    else:
        return pred_df

