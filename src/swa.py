import argparse
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
import torch
from tqdm.auto import tqdm

CHECKPOINT_PATH = Path(r"/database/kaggle/")

def average_checkpoints(input_ckpts, output_ckpt):
    assert len(input_ckpts) >= 1
    data = torch.load(input_ckpts[0], map_location='cpu')
    swa_n = 1
    for ckpt in tqdm(input_ckpts[1:]):
        new_data = torch.load(ckpt, map_location='cpu')
        swa_n += 1
        for k, v in new_data.items():
            if v.dtype == torch.int64:
                continue
            data[k] += (new_data[k] - data[k]) / swa_n
    torch.save(data, output_ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cp", type=str, default=None, required=True) 
    args = parser.parse_args()

    ckpts = [x.as_posix() for x in Path(CHECKPOINT_PATH/args.cp).glob("*.pth") if f'config' not in str(x)]
    average_checkpoints(ckpts, osp.join(str(CHECKPOINT_PATH/args.cp), 'swa.pth'))


if __name__ == '__main__':
    main()