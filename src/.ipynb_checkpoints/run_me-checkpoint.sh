python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 20 --bs 16 --exp 2stge_v26 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 20 --bs 16 --exp 2stge_v27 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 30 --bs 16 --exp 2stge_v28 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 16 --exp 2stge_v29 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 48

python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 20 --bs 32 --exp 2stge_v22 --img_size 256 --lr 1e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 20 --bs 32 --exp 2stge_v23 --img_size 256 --lr 3e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 30 --bs 32 --exp 2stge_v24 --img_size 256 --lr 3e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 32 --exp 2stge_v25 --img_size 256 --lr 1e-4 --decoder smp.Unet --seed 48

python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder inceptionv4 --epochs 20 --bs 64 --exp 2stge_v2 --img_size 256 --lr 1e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder inceptionv4 --epochs 20 --bs 64 --exp 2stge_v3 --img_size 256 --lr 3e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest50d --epochs 30 --bs 64 --exp 2stge_v4 --img_size 256 --lr 3e-4 --decoder smp.Unet --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder resnet34 --epochs 10 --bs 64 --exp 2stge_v5 --img_size 256 --lr 1e-4 --decoder smp.Unet --seed 48

python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder resnet34 --epochs 20 --bs 32 --exp 2stge_v6 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder timm-resnest26d --epochs 20 --bs 32 --exp 2stge_v7 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder inceptionv4 --epochs 30 --bs 32 --exp 2stge_v8 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 48
python train.py --config ../configs/std_img_256_8c_f.yaml --rep 1 --device 0 --encoder mit_b3 --epochs 10 --bs 16 --exp 2stge_v9 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 48






python train.py --config '../configs/deberta_v3_large.yaml' --device 0 --model_name microsoft/deberta-v3-xsmall --bs 1 --epochs 3 --max_len 4096 --folds 0
python predict.py --model_name 'deberta-v3-large' --device 0 --max_len 4096 --exp_name 2024-04-03--dv3l_cp_nbrod_add05_rep_05_v1
python train.py --config '../configs/deberta_v3_large.yaml' --device 0 --model_name microsoft/deberta-xlarge --bs 1 --epochs 7 --max_len 4096 --folds 0 1 2 3 4 --replace_text_prob 0.0 --add_text_prob 0.0 --external_data mixtral-8x7b-v1 --pct_eval 0.5



# python train.py --config ../configs/std_img_256.yaml --rep 2 --device 0 --encoder timm-resnest14d
# python train.py --config ../configs/std_img_256.yaml --rep 2 --device 0 --encoder timm-resnest26d
# python train.py --config ../configs/std_img_256.yaml --rep 2 --device 0 --encoder timm-resnest50d
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp seed1 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 48
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp seed2 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 2022
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp seed3 --img_size 256 --lr 3e-4 --decoder smp.UnetPlusPlus --seed 1710

# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp lr2seed1 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 48
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp lr2seed2 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 2022
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 1 --encoder timm-resnest200e --epochs 10 --bs 64 --exp lr2seed3 --img_size 256 --lr 1e-4 --decoder smp.UnetPlusPlus --seed 1710


# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 32 --exp 1e4 --img_size 256 --lr 1e-4
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 32 --exp 2e4 --img_size 256 --lr 2e-4
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 32 --exp 4e4 --img_size 256 --lr 4e-4
# python train.py --config ../configs/std_img_256.yaml --rep 1 --device 0 --encoder timm-resnest200e --epochs 10 --bs 32 --exp 8e4 --img_size 256 --lr 8e-4

# This is all ResNeSt + Res2Ne(X)t encoders