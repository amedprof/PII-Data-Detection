import argparse
import yaml 
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

from train_utils import kfold

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from sklearn.model_selection import GroupKFold,StratifiedGroupKFold,KFold,StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

DATA_PATH = Path(r"/database/kaggle/PII/data")
CHECKPOINT_PATH = Path(r"/database/kaggle/PII/checkpoint")


from datetime import date

TODAY = date.today()
TODAY = TODAY.strftime('%Y-%m-%d')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_version(start=0):
    if not hasattr(get_version, 'counter'):
        get_version.counter = start
    value = get_version.counter
    get_version.counter += 1
    return value

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("--device", type=int, default='0', required=False)   
    parser.add_argument("--rep", type=int, default=1, required=False) 
    parser.add_argument("--exp", type=str, default='v0', required=False) 
    parser.add_argument("--model_name", type=str, default=None, required=False) 
    parser.add_argument("--external_data", type=str, default=None, required=False) 
    parser.add_argument("--bs", type=int, default=None, required=False) 
    parser.add_argument("--epochs", type=int, default=None, required=False) 
    parser.add_argument("--lr", type=float, default=None, required=False) 
    parser.add_argument("--seed", type=int, default=-1, required=False) 
    parser.add_argument("--use_amp", type=int, default=None, required=False) 
    parser.add_argument("--scheduler", type=str, default=None, required=False) 
    parser.add_argument("--folds", nargs='+',type=int, default=None, required=False) 
    parser.add_argument("--max_len",type=int, default=None, required=False) 
    parser.add_argument("--pct_eval", type=float, default=None, required=False) 
    parser_args, _ = parser.parse_known_args(sys.argv)
    return parser.parse_args()

if __name__ == "__main__":
    
    cfg = parse_args()

    with open(cfg.config, 'r') as f:
        args = yaml.safe_load(f)

    args = SimpleNamespace(**args)
    args.device = cfg.device
    args.seed = cfg.seed
    exp = cfg.exp
    Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    if args.seed<0:
        args.seed = random.choice(np.arange(3500))
    seed_everything(args.seed)
    
    df = pd.read_json(DATA_PATH/'train.json')

    # if cfg.external_data:
    #     print("Using external data")
    #     dx = pd.read_json(DATA_PATH/f'{cfg.external_data}.json')
    #     # dx[name] = -1
    #     df = pd.concat([df,dx],axis=0).reset_index(drop=True)

    df['has_label'] = (df['labels'].transform(lambda x:len([i for i in x if i!="O" ]))>0)*1

    LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
    for name in LABEL2TYPE[:-1]:
        df[name] = ((df['labels'].transform(lambda x:len([i for i in x if i.split('-')[-1]==name ])))>0)*1

    seeds = [42]
    folds_names = []
    for K in [5]:  
        for seed in seeds:
            mskf = MultilabelStratifiedKFold(n_splits=K,shuffle=True,random_state=seed)
            name = f"fold_msk_{K}_seed_{seed}"
            df[name] = -1
            for fold, (trn_, val_) in enumerate(mskf.split(df,df[list(LABEL2TYPE)[:-1]])):
                df.loc[val_, name] = fold
            # valid_df[name] = 0
    
    if cfg.external_data:
        print("Using external data")
        dx = pd.read_json(DATA_PATH/f'{cfg.external_data}.json')
        LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
        for name in LABEL2TYPE[:-1]:
            dx[name] = ((dx['labels'].transform(lambda x:len([i for i in x if i.split('-')[-1]==name ])))>0)*1

        seeds = [42]
        folds_names = []
        for K in [5]:  
            for seed in seeds:
                mskf = MultilabelStratifiedKFold(n_splits=K,shuffle=True,random_state=seed)
                name = f"fold_msk_{K}_seed_{seed}"
                dx[name] = -1
                for fold, (trn_, val_) in enumerate(mskf.split(dx,dx[list(LABEL2TYPE)[:-1]])):
                    dx.loc[val_, name] = fold

        # dx[name] = -1
        df = pd.concat([df,dx],axis=0).reset_index(drop=True)

    print(df.groupby(name)[list(LABEL2TYPE)[:-1]].sum())
    # data_path = Path(r"/database/kaggle/Identify Contrails/data")
    # df = pd.concat([train_df,valid_df],axis=0).reset_index(drop=True)
    # df['file_name'] = str(data_path/"3c_all_images/")+'/'+df['record_id'].astype(str)+'.npy'

    # del train_df,valid_df
    # gc.collect()

    if cfg.model_name is not None:
        args.model["model_params"]["model_name"] = cfg.model_name

    if cfg.bs is not None:
        args.train_loader["batch_size"] = cfg.bs

    if cfg.epochs is not None:
        args.trainer["epochs"] = cfg.epochs
    if cfg.lr is not None:
        args.optimizer["params"]["lr"] = cfg.lr
   
    if cfg.scheduler is not None:
        args.scheduler["name"] = cfg.scheduler

    if cfg.use_amp is not None:
        args.trainer["use_amp"] = cfg.use_amp
    
    if cfg.pct_eval is not None:
        args.callbacks["epoch_pct_eval"] = cfg.pct_eval

    if cfg.folds is not None:
        args.selected_folds = cfg.folds
    
    if cfg.max_len is not None:
        args.model["model_params"]['max_len'] = cfg.max_len
    

    args.model_name =  args.model["model_params"]["model_name"]
    # config_name = cfg.config.split('/')[-1].split('.')[0]
    args.name = args.model["model_params"]["model_name"].split('/')[1] if '/' in args.model["model_params"]["model_name"] else args.model["model_params"]["model_name"]
    encoder_name = args.model["model_params"]["model_name"]
    args.exp_name = f"{TODAY}--{exp}"  
    args.checkpoints_path = str(CHECKPOINT_PATH/Path(fr'{args.kfold_name}/{args.name}/{args.exp_name}')) 
    # args.model['pretrained_weights'] = str(Path(r"/database/kaggle/MLM/Shovel Ready/checkpoint/fold/deberta-v3-large/microsoft/deberta-v3-large--2023-08-26--ep_10_bs_1_v1"))
    args.model['pretrained_weights'] = None #str(Path(r"/database/kaggle/Shovel Ready/checkpoint/fold/deberta-large/2023-08-30--awp_v3"))
    args.model["model_params"]['pretrained_path'] = args.model['pretrained_weights'] 
    Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    args.selected_folds = args.selected_folds*cfg.rep

    kfold(args,df)

