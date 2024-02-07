import re
import os
import gc
import math
import time
import json
import random
import numpy as np
import pandas as pd
import wandb

from pathlib import Path

import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig
 
from data.data_utils import to_gpu,to_np
from data.dataset import FeedbackDataset,CustomCollator,ID_TYPE,ID_NAME,LABEL2TYPE,TYPE2LABEL
from torch.utils.data import DataLoader

from model_zoo.models import FeedbackModel,span_nms,aggregate_tokens_to_words
from metrics_loss.metrics import score_feedback,score,pii_fbeta_score_v2,compute_metrics
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup

# from sklearn.metrics import log_loss 
from tqdm.auto import tqdm

from utils.utils import count_parameters
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def average_checkpoints(input_ckpts, output_ckpt):
    assert len(input_ckpts) >= 1
    data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
    swa_n = 1
    for ckpt in input_ckpts[1:]:
        new_data = torch.load(ckpt, map_location='cpu')['state_dict']
        swa_n += 1
        for k, v in new_data.items():
            if v.dtype != torch.float32:
                print(k)
            else:
                data[k] += (new_data[k] - data[k]) / swa_n
    torch.save(dict(state_dict=data), output_ckpt)


class EMA():
    """credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
# ------------------------------------------ Loss function ------------------------------------------- #
import torch 

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=0.0001,
        adv_eps=0.001,
        start_epoch=1,
        adv_step=1,
        scaler=None,
        args = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.args = args
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu") 
        self.criterion = eval(args.model['loss']['loss_name'])(**args.model['loss']['loss_params']).to(device)
    
    def attack_backward(self,data,epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            adv_loss,_ = training_step(self.args,self.model,data,self.criterion)            
            self.optimizer.zero_grad()
            if self.scaler is None:
                adv_loss.backward()
            else:
                self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

# ------------------------------------------ Seed Everything ------------------------------------------- #

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ------------------------------------------  ------------------------------------------- #
# ------------------------------------------ Auto SAVE ------------------------------------------- #

class AutoSave:
  def __init__(self, top_k=3,metric_track="mae_val",mode="min", root=None):
    
    self.top_k = top_k
    self.logs = []
    self.metric = metric_track
    self.mode = -1 if mode=='min' else 1
    self.root = Path(root)
    assert self.root.exists()

    self.top_models = []
    self.top_metrics = []
    self.texte_log = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(self.mode*metric)

    self.top_metrics.insert(rank+1, self.mode*metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)


    self.logs.append(metrics)
    self.save(model, rank, metrics)


  def save(self, model,rank, metrics):
    val_text = " "
    order_prt = ["fold","epoch",'step','train_loss','valid_loss',self.metric] if self.metric!="valid_loss" else ["fold","epoch",'step','train_loss','val_loss']
    for k,v in metrics.items():
        if k in order_prt:
            if k in ["fold","epoch",'step']:
                val_text+=f"_{k}={v:.0f} "  if k!="fold" else f"{k}={v:.0f} "
            else:
                val_text+=f"_{k}={v:.4f} "

    name = val_text.strip()
    name = name+".pth"
    name = name.replace('=',"_")
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r


# # ----------------- Opt/Sched --------------------- #
def get_optim_sched(model,args,plot=False):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.optimizer["params"]['weight_decay'],
        "lr": args.optimizer["params"]['lr'],
                                    },
                                    {
                                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                        "weight_decay": 0.0,
                                        "lr": args.optimizer["params"]['lr'],
                                    }]

    if args.optimizer['name']=="optim.AdamW":
        optimizer = eval(args.optimizer['name'])(model.parameters(),lr=args.optimizer["params"]['lr'])
    else:
        optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

    # if 'scheduler' in args:
    if args.scheduler['name'] == 'poly':

        params = args.scheduler['params']

        power = params['power']
        lr_end = params['lr_end']

        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
            
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:
        max_lr = args.optimizer['params']['lr']
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        scheduler = eval(args.scheduler['name'])(optimizer,max_lr=max_lr,
                                                 epochs=args.trainer['epochs'],
                                                 steps_per_epoch=training_steps,
                                                 pct_start = args.scheduler['warmup']
                                                 )
        
    elif args.scheduler['name'] in ['optim.lr_scheduler.CosineAnnealingLR']:
        scheduler = eval(args.scheduler['name'])(optimizer,**args.scheduler['params'])
    

    # Plot the lr curve
    if plot:
        lrs = []
        for i in range(training_steps):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        plt.figure(figsize=(10,5))
        plt.plot(range(training_steps),lrs, marker='o')
        plt.xlabel('epoch'); plt.ylabel('learnig rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    return optimizer,scheduler


# # ----------------- One Step --------------------- #
# # ----------------- One Step --------------------- #
def training_step(args,model,data,criterion,criteriony):
    model.train()
    device = model.backbone.device
    data = to_gpu(data, device)

    if args.trainer['use_amp']:
        with amp.autocast(args.trainer['use_amp']):
            pred = model(data)

            # print(y.shape)
            # print(data["gt_spans"][:1,-1].shape)
            
            mask = (data["word_boxes"]!=-100)[:,0]  
            # print(sum(mask))        
            lossx = criterion(pred["pred"][mask],data["gt_spans"][mask,1])
            if "y" in pred.keys():
                lossy = criteriony(pred['y'][mask],data["gt_spans"][mask,-1].to(torch.float32))
                loss = lossx+lossy
            else:
                loss = lossx

            log_vars = dict(
                train_loss=loss.item(),
                train_lossx = lossx.item()
                            )
            if "y" in pred.keys():
                log_vars["train_lossy"] = lossy.item()

    else:
        pred = model(data)

        # print(y.shape)
        # print(data["gt_spans"][:1,-1].shape)
        
        mask = (data["word_boxes"]!=-100)[:,0]  
        # print(sum(mask))        
        lossx = criterion(pred["pred"][mask],data["gt_spans"][mask,1])
        if "y" in pred.keys():
            lossy = criteriony(pred['y'][mask],data["gt_spans"][mask,-1].to(torch.float32))
            loss = lossx+lossy
        else:
            loss = lossx

        log_vars = dict(
            train_loss=loss.item(),
            train_lossx = lossx.item()
                        )
        if "y" in pred.keys():
            log_vars["train_lossy"] = lossy.item()
    return loss,log_vars

def evaluation_step(args,model,val_loader,criterion,criteriony):

    device = model.backbone.device
    model.eval()

    losses = []
    gt_df = []
    pred_df = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = to_gpu(data, device)
            pred = model(data)

            mask = (data["word_boxes"]!=-100)[:,0]  
            lossx = criterion(pred["pred"][mask],data["gt_spans"][mask,1])

            if "y" in pred.keys():
                lossy = criteriony(pred['y'][mask],data["gt_spans"][mask,-1].to(torch.float32))
                loss = lossx+lossy
                losses.append([loss.item(),lossx.item(),lossy.item()])
            else:
                loss = lossx
                losses.append([loss.item(),lossx.item()])

            pred = pred['pred']
            pred  = pred.softmax(-1)
            s,i = pred.max(-1)
            p_df = pd.DataFrame({"document":data['text_id'],
                                 "token" : np.arange(pred.shape[0]),
                                 "label" : i.detach().cpu().numpy() ,
                                 "score" : s.detach().cpu().numpy() 
                                 })
            pred_df.append(p_df)
            data = to_np(data)

            gt = pd.DataFrame({
                                "document":data['text_id'],
                                "token":np.arange(pred.shape[0]),
                                "label":data["gt_spans"][:,1],
                                "I":data["gt_spans"][:,2],
                            })
            gt_df.append(gt)
            

    pred_df = pd.concat(pred_df,axis=0).reset_index(drop=True)

    pred_df["label_next_e_prev"] = pred_df.groupby('document')['label'].transform(lambda x: (x.shift(1)==x.shift(-1))*1)
    pred_df["label_next"] = pred_df.groupby('document')['label'].transform(lambda x: x.shift(1))
    pred_df["label_next_e_prev"] = ((pred_df["label_next_e_prev"]==1) & (pred_df["label_next"]==6))*1
    pred_df["score_next"] = pred_df.groupby('document')['score'].transform(lambda x: x.shift(1))
    pred_df.loc[pred_df["label_next_e_prev"]==1,"label"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"label_next"]
    pred_df.loc[pred_df["label_next_e_prev"]==1,"score"] = pred_df.loc[pred_df["label_next_e_prev"]==1,"score_next"]


    pred_df = pred_df[(pred_df.label!=7) & ((pred_df.score>0.15))].reset_index(drop=True)
    pred_df["I"] = ((pred_df.groupby('document')['label'].transform(lambda x:x.diff())==0) & (pred_df.groupby('document')['token'].transform(lambda x:x.diff())==1))*1
    pred_df['labels'] = pred_df['label'].astype(str)+'-'+pred_df['I'].astype(str)
    pred_df["label_pred"] = pred_df["labels"].map(ID_TYPE).fillna(0).astype(int)
    pred_df['row_id'] = np.arange(len(pred_df))


    # pred_df = pred_df[(pred_df.label!=7) & (pred_df.score>0.5)].reset_index(drop=True)
    # pred_df["I"] = ((pred_df.groupby('document')['label'].transform(lambda x:x.diff())==0) & (pred_df.groupby('document')['token'].transform(lambda x:x.diff())==1))*1
    # pred_df['labels'] = pred_df['label'].astype(str)+'-'+pred_df['I'].astype(str)
    # pred_df["label_pred"] = pred_df["labels"].map(ID_TYPE).fillna(0).astype(int)
    # pred_df['row_id'] = np.arange(len(pred_df))

    gt_df = pd.concat(gt_df,axis=0).reset_index(drop=True)
    gt_df = gt_df[gt_df.label!=7].reset_index(drop=True)
    gt_df['labels'] = gt_df['label'].astype(str)+'-'+gt_df['I'].astype(str)
    gt_df["label_gt"] = gt_df["labels"].map(ID_TYPE).fillna(0).astype(int)
    gt_df['row_id'] = np.arange(len(gt_df))


    # try:
    pred_df['label'] = pred_df['labels'].map(ID_NAME)
    gt_df['label'] = gt_df['labels'].map(ID_NAME)
    scores = compute_metrics(pred_df, gt_df) #score(gt_df, pred_df, row_id_column_name = "row_id", beta = 5)#score_feedback(pred_df, gt_df,return_class_scores=False)
    # s = pii_fbeta_score_v2(pred_df, gt_df, beta=5)
    # except:
        # micro_f1,macro_f1 = 0,{}

    # macro_f1 = micro_f1
    losses = np.mean(losses,axis=0)

    log_vars = dict(
        valid_loss=losses[0],
        valid_lossx=losses[1],
        valid_lossy=losses[-1],
        f5_prec = scores['f5_prec'],
        f5_rec = scores['f5_rec'],
        f5_micro = scores['f5_micro'],
        # f5_micro_new = micro_f1_new,
        # score = s
    )
    log_vars.update(scores['ents_per_type'])
    return log_vars
# ------------------------------------------ ------------------------------------------- #
def inference_step(args,df):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device>=0 else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model["model_params"]['model_name'])    
    valid_dataset = eval(args.dataset)(df,tokenizer,**args.data["params_valid"])

    model = FeedbackModel(**args.model["model_params"]).to(device)
    # print(model)
    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")

    model.load_state_dict(torch.load(args.pretrained_weights, map_location=lambda storage, loc: storage))
    model = model.to(device)
  
    collator = CustomCollator(tokenizer,model)
    val_loader = DataLoader(valid_dataset,**args.val_loader,collate_fn=collator)

    model.eval()
    pred_df = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = to_gpu(data, device)
            
            pred = model(data)
            pred  = pred.softmax(-1)
            s,i = pred.max(-1)
            p_df = pd.DataFrame({"document":data['text_id'],
                                 "token" : np.arange(pred.shape[0]),
                                 "label" : i.detach().cpu().numpy() ,
                                 "score" : s.detach().cpu().numpy() 
                                 })
            pred_df.append(p_df)

    pred_df = pd.concat(pred_df,axis=0).reset_index(drop=True)

    return pred_df
# #----------------------------------- Training Steps -------------------------------------------------#

def fit_net(
        model,
        train_dataset,
        valid_dataset,
        args,
        fold,
        wandb
    ):
    device = model.backbone.device

    loss_params = args.model['loss']['loss_params']
    names = [ x for x in LABEL2TYPE  if x in train_dataset.df.columns.tolist()]
    weight = torch.Tensor([1/train_dataset.df[name].sum() for name in names])
    loss_params.update({"weight":weight})

    criterion = eval(args.model['loss']['loss_name'])(**loss_params).to(device)
    criteriony = eval(args.model['lossy']['loss_name'])(**args.model['lossy']['loss_params']).to(device)
    # criterion = eval(args.model['loss'])(**args.model['loss_params']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = CustomCollator(tokenizer,model)
    train_loader = DataLoader(train_dataset,**args.train_loader,collate_fn=collator)
    val_loader = DataLoader(valid_dataset,**args.val_loader,collate_fn=collator)

    args.len_train_loader = len(train_loader)
    args.dataset_size = len(train_dataset)

    mode_ = -1 if args.callbacks["mode"]=='max' else 1
    best_epoch = mode_*np.inf
    best = mode_*np.inf

    es = args.callbacks['es']
    es_step = 0
    patience = args.callbacks['patience']

    if args.callbacks["save"]:
        Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    saver = AutoSave(root=args.checkpoints_path,metric_track=args.callbacks['metric_track'],top_k=args.callbacks['top_k'],mode=args.callbacks['mode'])

  
    if args.trainer['use_amp'] and ("cuda" in str(device)):
        scaler = amp.GradScaler(enabled=args.trainer['use_amp'])
        print("Using Amp")
    else:
        scaler = None

    optimizer,scheduler = get_optim_sched(model,args)
    end_steps = [x for x in np.arange(args.trainer['epochs'])][-args.callbacks['save_last_k']:]

    if args.trainer['use_awp']>0:
        print(f"Using awp")
        awp = AWP(model,
                    optimizer,
                    #   adv_lr=args.adv_lr,
                    #   adv_eps=args.adv_eps,
                    #   start_epoch=args.num_train_steps/args.epochs,
                    scaler=scaler,
                    args=args
                )
    
      #------- EMA -----------------------------------------------------------------------------#
    if args.trainer['ema_decay_rate']>0:
        decay_rate = args.trainer['ema_decay_rate']  # torch.exp(torch.log(cfg.train_params.ema_prev_epoch_weight) / num_update_steps_per_epoch)
        ema = EMA(model, decay=decay_rate)
        ema.register()
        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")

    for epoch in range(args.trainer['epochs']):
        # Init
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        
        nb_step_per_epoch = args.len_train_loader
        step_val = int(np.round(nb_step_per_epoch*args.callbacks['epoch_pct_eval']))
        nstep_val = int(1/args.callbacks['epoch_pct_eval'])

        if args.callbacks['epoch_eval_dist']=="uniforme":
            evaluation_steps = [(nb_step_per_epoch//2)+x for x in np.arange(0,nb_step_per_epoch//2,nb_step_per_epoch//(2*nstep_val))][1:]
        else:
            evaluation_steps = [x for x in np.arange(nb_step_per_epoch) if (x + 1) % step_val == 0][1:]

        
        pbar = tqdm(train_loader)
        for step,data in enumerate(pbar):
            if step==epoch and step==0:
                print(" ".join(train_dataset.tokenizer.convert_ids_to_tokens(data['input_ids'][0])))

            loss,trc= training_step(args,model,data,criterion,criteriony)

            pbar.set_postfix(trc)
            if step==0:
                # Init Metrics
                trn_metric = {}
                for k in trc.keys():
                    trn_metric[k] = 0

            # Sum of metrics
            for k in trc.keys():
                trn_metric[k]+= trc[k]

 
            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
                if args.trainer['use_awp']:
                    awp.attack_backward(data,epoch) 

                # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                                                        parameters=model.parameters(), max_norm=args.trainer['max_norm']
                                                    )

                scaler.step(optimizer)
                scaler.update()
                

            else:
                loss.backward()
                if args.trainer['use_awp']:
                    awp.attack_backward(data,epoch)

                # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                                                    parameters=model.parameters(), max_norm=args.trainer['max_norm']
                                                )

                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            
            if args.trainer['ema_decay_rate']>0:
                ema.update()

            # Evaluation
            if (((step + 1) in evaluation_steps) or (step + 1 == nb_step_per_epoch)) and (epoch>=args.callbacks["start_eval_epoch"]):
                
                model.eval()
                # apply ema if it is used
                if args.trainer['ema_decay_rate']>0:
                    ema.apply_shadow()

                metric_val = evaluation_step(args,model,val_loader,criterion,criteriony)

                # if es:
                if args.callbacks['mode']=='min':
                    if (metric_val[args.callbacks['metric_track']]<best):
                        best = metric_val[args.callbacks['metric_track']]
                else:
                    if (metric_val[args.callbacks['metric_track']]>best):
                        best = metric_val[args.callbacks['metric_track']]

                metrics = {
                    "fold":fold,
                    "epoch": epoch+1,
                    "step": int(step),
                    "global_step":step+(epoch*nb_step_per_epoch),
                    "best":best
                    
                }

                met_train = trn_metric.copy()
                # AVG of metrics
                for k in trc.keys():
                    met_train[k]= met_train[k]/(step+1)

                metrics.update(metric_val)
                metrics.update(met_train)
                saver.log(model, metrics)
        
                elapsed_time = time.time() - start_time
                elapsed_time = elapsed_time * args.callbacks['verbose_eval']

                lr = scheduler.get_lr()[0]
                
                val_text = " "
                for k,v in metric_val.items():
                    val_text+=f" {k}={v:.4f} "

                trn_text = " "
                for k,v in met_train.items():
                    trn_text+=f" {k}={v:.4f} "

                metrics.update({"lr":lr})

                if args.callbacks['use_wnb']:
                    wandb.log(metrics)

                
                texte = f"Epoch {epoch + 1}.{int(np.ceil((step+1)/step_val))}/{args.trainer['epochs']} lr={lr:.6f} t={elapsed_time:.0f}s "
                # texte = texte+trn_text+val_text
                print("="*len(texte))
                print(texte+trn_text)
                print("="*len(texte))

                print("="*len(texte))
                print(texte+val_text)
                print("="*len(texte))
                metric_val = metric_val[args.callbacks['metric_track']]

        # Saving
        if ((((epoch) in end_steps)) and (args.callbacks['save_last_k']>0)) or ((epoch+1)==args.trainer['epochs']):
            name = f"fold_{fold}_epoch_{epoch}.pth"
            direc = Path(args.checkpoints_path).joinpath(name)
            torch.save(model.state_dict(), direc.as_posix())  

        if args.trainer['ema_decay_rate']>0:
            ema.restore()
        # if es:
        if args.callbacks['mode']=='min':
            if (best<best_epoch):
                best_epoch = best
                es_step = 0
            else:
                es_step+=1
                print(f"es step {es_step}  best is {best}")
        else:
            if (best>best_epoch):
                best_epoch = best
                es_step = 0
            else:
                es_step+=1
                print(f"es step {es_step}  best is {best}")

        if (es_step>patience) and es:
            break

    torch.cuda.empty_cache()


def train_one_fold(args,
                      train_df,
                      valid_df,
                      fold,
                      wandb):    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device>=0 else "cpu")
    print(f"Training with {device}")
    args.pretrained_path = Path(args.model["model_params"]['pretrained_path']) if args.model["model_params"]['pretrained_path'] else None

    if args.pretrained_path:
        checkpoints = [x.as_posix() for x in Path(args.pretrained_path).glob("*.pth") if any([f'fold_{fold}' in str(x),
                                                                                                f'fold{fold}' in str(x),f'fold={fold}' in str(x)])]
        checkpointg = [x.as_posix() for x in Path(args.pretrained_path).glob("*.pth") if f'global' in str(x)]
        if len(checkpoints):
            pretrained_path = random.choice(checkpoints)
        elif len(checkpointg):
            pretrained_path = random.choice(checkpointg)
        else:
            try:
                checkpoints = [x.as_posix() for x in Path(args.pretrained_path).glob("*.pth") if f'config' not in str(x)]
                pretrained_path = random.choice(checkpoints)
            except:
                pretrained_path = args.pretrained_path
    else:
        pretrained_path = None

    args.model["model_params"]['pretrained_path'] = pretrained_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = eval(args.dataset)(train_df,tokenizer,**args.data["params_train"])
    
    valid_dataset = eval(args.dataset)(valid_df,tokenizer,**args.data["params_valid"])

    model = FeedbackModel(**args.model["model_params"]).to(device)
    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")
    if pretrained_path:
        print("Using Pretrained Weights")
        print(pretrained_path)
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))
            model = model.to(device)
        except:
            print('Loading failed')
    model.zero_grad()    

    fit_net(
        model,
        train_dataset,
        valid_dataset,
        args,
        fold,
        wandb
    )
    
def kfold(args,df):
    


    if args.model['pretrained_tokenizer']:
        tokenizer = AutoTokenizer.from_pretrained(args.model['pretrained_tokenizer'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model["model_params"]['model_name'])

    if args.model['pretrained_tokenizer']:
        pass
    else:
        tokenizer.save_pretrained(Path(args.checkpoints_path)/'tokenizer/')
        config = AutoConfig.from_pretrained(args.model["model_params"]['model_name'])
        torch.save(config, Path(args.checkpoints_path)/'config.pth')

    k = len((df[df[args.kfold_name]!=-1][args.kfold_name].unique()))
    folds = df[df[args.kfold_name]!=-1][args.kfold_name].unique()
    print(f"----------- {args.kfold_name} ---------")
    
    config = {"model":args.model,'dataset':args.dataset,"exp_name":args.exp_name}
            
    config.update({"data":args.data})
    config.update({"optimizer":args.optimizer})
    config.update({'scheduler':args.scheduler})
    config.update({"train_loader":args.train_loader})
    config.update({"val_loader":args.val_loader})
    config.update({"trainer":args.trainer})
    config.update({"callbacks":args.callbacks})
    
    with open(args.checkpoints_path+'/params.json', 'w') as f:
        json.dump(config, f)

    
                
    for i in args.selected_folds:
        if i in folds:
            print(f"\n-------------   Fold {i+1} / {k}  -------------\n")
            if args.trainer['sample']:
                train_df = df.head(100).reset_index(drop=True)
                valid_df = df.head(500).reset_index(drop=True)

            elif args.trainer['train_all_data']:
                train_df = df.reset_index(drop=True)#.sample(100)
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True)#.sample(100)

            else:
                train_df = df[~(df[args.kfold_name].isin([i]))].reset_index(drop=True)
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True)

            print(f"Training size {len(train_df)}")
            print(f"Validation size {len(valid_df)}")

            if args.callbacks['use_wnb']:
                wandb.init(project=args.project_name,
                        name=f'{args.exp_name}_fold_{i+1}',
                        group=f'{args.exp_name}',
                        #  job_type="training", 
                        config=config,
                        )
            
            train_one_fold(args,
                                train_df,
                                valid_df,
                                i,
                                wandb
                                )