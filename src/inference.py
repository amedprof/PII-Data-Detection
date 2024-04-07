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



    def inference_steps(df,folder,bs=1,folds=[0],device=0,max_len=4096):
        
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
                    
                    
                    if j==0:
                    
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
                predictions = torch.cat([torch.max(predictions[:, :-1], torch.cat(preds,dim=0)[:, :-1]),
                                        torch.min(predictions[:, -1:], torch.cat(preds,dim=0)[:, -1:])],dim=-1)
                
    #             predictions+= torch.cat(preds,dim=0)*weight
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

    def make_pred_df(pred_df,threshold=0.15):
        
        pred_df["label_next_e_prev"] = pred_df.groupby('document')['label'].transform(lambda x: (x.shift(1)==x.shift(-1))*1)
        pred_df["label_next"] = pred_df.groupby('document')['label'].transform(lambda x: x.shift(1))
        pred_df["label_next_e_prev"] = ((pred_df["label_next_e_prev"]==1) & (pred_df["label_next"]==6))*1
        pred_df["score_next"] = pred_df.groupby('document')['score'].transform(lambda x: x.shift(1))
        pred_df.loc[pred_df["label_next_e_prev"]==1,"label"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"label_next"]
        pred_df.loc[pred_df["label_next_e_prev"]==1,"score"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"score_next"]
        
        if threshold>0:
            pred_df = pred_df[(pred_df.label!=7) & ((pred_df.score>threshold))].reset_index(drop=True)
        
    #     pred_df['token_size'] = pred_df['tokens'].transform(len)
    #     pred_df = pred_df[~((pred_df.label==0) & ((pred_df.token_size<=1)))].reset_index(drop=True)
        
        pred_df["I"] = ((pred_df.groupby('document')['label'].transform(lambda x:x.diff())==0) & (pred_df.groupby('document')['token'].transform(lambda x:x.diff())==1))*1
        pred_df['labels'] = pred_df['label'].astype(str)+'-'+pred_df['I'].astype(str)
        pred_df["label_pred"] = pred_df["labels"].map(ID_TYPE).fillna(0).astype(int)
        pred_df['row_id'] = np.arange(len(pred_df))
        
        pred_df['label'] = pred_df['labels'].map(ID_NAME)
        return pred_df

    pred = []
    gt = []
    FOLD_NAME = "fold_msk_5_seed_42"
    folder = str(CHECKPOINT_PATH/Path(fr'{FOLD_NAME}/{cfg.model_name}/{cfg.exp_name}')) 

    if cfg.blend==1:
        pred_df,gt_df = inference_steps(df,folder,bs=1,folds=[0,1,2,3,4,5,6,7,8,9,10])
        # pred_df.to_csv(Path(folder)/f'pii-200-ms-blend.csv',index=False)
        pred_df = make_pred_df(pred_df,threshold=0.15)
        gt_df['label'] = gt_df['label'].map(LABEL2TYPE)
        s = compute_metrics_new(pred_df, gt_df)
        print(s)
        print("\n")

    elif cfg.blend==2:
        print('blending per fold')
        pred = []
        pdf = []
        gt = []
        folders = [str(CHECKPOINT_PATH/Path(f'{FOLD_NAME}/{x}'))  for x in cfg.folders]
        for FOLD in [0,1,2,3,4]:
            pred_df_dv3,gt_df = inference_blendings(df[df[FOLD_NAME]==FOLD],folders,bs=1,folds=[FOLD],selected_device=cfg.device,max_len=cfg.max_len) #[df[FOLD_NAME]==FOLD]
            
            pdf.append(pred_df_dv3)
            gt_df['label'] = gt_df['labels'].map(ID_NAME)
            pred_df_dv3 = make_pred_df(pred_df_dv3,threshold=0.15)
            s = compute_metrics(pred_df_dv3, gt_df)
            
            print(f"Fold {FOLD}")
            print(s)
            print("\n")

            pred.append(pred_df_dv3)
            
            gt.append(gt_df)
        
        pred = pd.concat(pred).reset_index(drop=True)
        gt = pd.concat(gt).reset_index(drop=True)

        s = compute_metrics(pred, gt)
        print(f"OOF")
        print(s)

        documents = df.document.unique()
        df_score = []
        for doc in tqdm(documents):
            p = pred[pred.document==doc].reset_index(drop=True)
            gp = gt[gt.document==doc].reset_index(drop=True)
            
            s = compute_metrics(p, gp)
            
            d = pd.DataFrame({x:[y] for x,y in s['ents_per_type'].items()})
            d["f5_micro"] = s['f5_micro']
            d['document'] = doc
            df_score.append(d)

        df_score = pd.concat(df_score).reset_index(drop=True)
        pdf = pd.concat(pdf,axis=0)
        pdf = make_pred_df(pdf,threshold=0)
        pdf = pdf.groupby("document")['label','score'].agg(list).reset_index()
        # dfold = dfold.merge(df_score,how='left',on='document',suffixes=('','_s'))
        df = df.merge(df_score,how='left',on='document',suffixes=('','_s'))
        df = df.merge(pdf,how='left',on='document')

        df.to_csv(Path(folder)/f'oof_blend_mean.csv',index=False)

        
    else:
        pred = []
        pdf = []
        gt = []
        for FOLD in [0,1,2,3,4]:
            pred_df_dv3,gt_df = inference_steps(df[df[FOLD_NAME]==FOLD],folder,bs=1,folds=[FOLD],device=cfg.device,max_len=cfg.max_len) #[df[FOLD_NAME]==FOLD]
            
            pdf.append(pred_df_dv3)
            gt_df['label'] = gt_df['labels'].map(ID_NAME)
            pred_df_dv3 = make_pred_df(pred_df_dv3,threshold=0.15)
            s = compute_metrics(pred_df_dv3, gt_df)
            
            print(f"Fold {FOLD}")
            print(s)
            print("\n")

            pred.append(pred_df_dv3)
            
            gt.append(gt_df)
        
        pred = pd.concat(pred).reset_index(drop=True)
        gt = pd.concat(gt).reset_index(drop=True)

        s = compute_metrics(pred, gt)
        print(f"OOF")
        print(s)

        documents = df.document.unique()
        df_score = []
        for doc in tqdm(documents):
            p = pred[pred.document==doc].reset_index(drop=True)
            gp = gt[gt.document==doc].reset_index(drop=True)
            
            s = compute_metrics(p, gp)
            
            d = pd.DataFrame({x:[y] for x,y in s['ents_per_type'].items()})
            d["f5_micro"] = s['f5_micro']
            d['document'] = doc
            df_score.append(d)

        df_score = pd.concat(df_score).reset_index(drop=True)
        pdf = pd.concat(pdf,axis=0)
        pdf = make_pred_df(pdf,threshold=0)
        pdf = pdf.groupby("document")['label','score'].agg(list).reset_index()
        # dfold = dfold.merge(df_score,how='left',on='document',suffixes=('','_s'))
        df = df.merge(df_score,how='left',on='document',suffixes=('','_s'))
        df = df.merge(pdf,how='left',on='document')

        df.to_csv(Path(folder)/f'oof.csv',index=False)

