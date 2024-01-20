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

from train_utils import evaluate_save

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path(r"/database/kaggle/Shovel Ready/data")
CHECKPOINT_PATH = Path(r"/database/kaggle/Shovel Ready/checkpoint")


from datetime import date

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--folder", help="folder filename")
    parser.add_argument("--device", type=int, default='0', required=False)   
    parser.add_argument("--weights", type=str, default=None, required=False) 
    parser.add_argument("--bs", type=int, default=None, required=False) 

    parser_args, _ = parser.parse_known_args(sys.argv)
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()

    f = open(CHECKPOINT_PATH/f'{cfg.folder}/params.json')
    args = json.load(f)
    args = SimpleNamespace(**args)

    args.device = cfg.device
    args.folder = cfg.folder

    
    df = pd.read_csv(DATA_PATH/'persuade_corpus.csv')
    LABEL2EFFEC = ('Adequate', 'Effective', 'Ineffective')
    EFFEC2LABEL = {t: l for l, t in enumerate(LABEL2EFFEC)}

    df_valid = df.loc[df['competition_set'] == "test"].reset_index(drop=True)
    df_valid = df_valid.loc[df_valid['discourse_type'] != "Unannotated"]
    df_valid = df_valid[df_valid.discourse_effectiveness.isin(LABEL2EFFEC)].reset_index(drop=True)
    df_valid = df_valid[df_valid.test_split_feedback_1=='Public']
    df_valid = df_valid[df_valid.discourse_effectiveness.isin(LABEL2EFFEC)]
    df_valid['fold'] = 0

    df = df.loc[df['discourse_type'] != "Unannotated"]
    df.discourse_effectiveness = df.discourse_effectiveness.fillna('Adequate')

    args.model['pretrained_weights'] = str(CHECKPOINT_PATH/cfg.folder/cfg.weights)
    args.pretrained_weights = str(CHECKPOINT_PATH/cfg.folder/cfg.weights)
    args.model_name =  args.model["model_params"]["model_name"]
    if cfg.bs is not None:
        args.val_loader["batch_size"] = cfg.bs
    
    args.folder = CHECKPOINT_PATH/f'{cfg.folder}'
    evaluate_save(args,df)