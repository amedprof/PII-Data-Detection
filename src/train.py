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
    parser.add_argument("--bs", type=int, default=None, required=False) 
    parser.add_argument("--epochs", type=int, default=None, required=False) 
    parser.add_argument("--lr", type=float, default=None, required=False) 
    parser.add_argument("--seed", type=int, default=-1, required=False) 
    parser.add_argument("--use_amp", type=int, default=None, required=False) 
    parser.add_argument("--scheduler", type=str, default=None, required=False) 
    parser.add_argument("--folds", nargs='+',type=int, default=None, required=False) 
    parser.add_argument("--max_length",type=int, default=None, required=False) 
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
    df['has_label'] = (df['labels'].transform(lambda x:len([i for i in x if i!="O" ]))>0)*1
    seeds = [42]
    folds_names = []
    for K in [5]:  
        for seed in seeds:
            mskf = StratifiedKFold(n_splits=K,shuffle=True,random_state=seed)
            name = f"fold_sk_{K}_seed_{seed}"
            df[name] = -1
            for fold, (trn_, val_) in enumerate(mskf.split(df,df['has_label'])):
                df.loc[val_, name] = fold
            # valid_df[name] = 0
    
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

    if cfg.folds is not None:
        args.selected_folds = cfg.folds
    
    if cfg.max_length is not None:
        args.data["params_valid"]['max_length'] = cfg.max_length
        args.data["params_train"]['max_length'] = cfg.max_length
    

    args.model_name =  args.model["model_params"]["model_name"]
    # config_name = cfg.config.split('/')[-1].split('.')[0]
    args.name = args.model["model_params"]["model_name"].split('/')[1] if '/' in args.model["model_params"]["model_name"] else args.model["model_params"]["model_name"]
    encoder_name = args.model["model_params"]["model_name"]
    args.exp_name = f"{TODAY}--{exp}"  
    args.checkpoints_path = str(CHECKPOINT_PATH/Path(fr'{args.kfold_name}/{args.name}/{args.exp_name}')) 
    # args.model['pretrained_weights'] = str(Path(r"/database/kaggle/MLM/Shovel Ready/checkpoint/fold/deberta-v3-large/microsoft/deberta-v3-large--2023-08-26--ep_10_bs_1_v1"))
    args.model['pretrained_weights'] = str(Path(r"/database/kaggle/Shovel Ready/checkpoint/fold/deberta-large/2023-08-30--awp_v3"))
    args.model["model_params"]['pretrained_path'] = args.model['pretrained_weights'] 
    Path(args.checkpoints_path).mkdir(parents=True,exist_ok=True)

    args.selected_folds = args.selected_folds*cfg.rep

    kfold(args,df)

