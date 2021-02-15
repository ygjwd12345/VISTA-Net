
nohup python -u  train.py --dataset pcontext --model encnet --attentiongraph \
--aux --se-loss --backbone resnet101 --checkname attentiongraph_res101_pcontext_v2 \
--batch-size 2 --checkpoint-path ./ > run.log &
python train.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --checkname attentiongraph_res101_pcontext_v2 --epoch 150 --batch-size 2 --rank 1 \
    --checkpoint-path ./

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset cityscape \
    --model encnet  --aux --se-loss \
    --backbone resnet101 --checkname attentiongraph_res101_pcontext_v2  \
    --checkpoint-path ./ --base-size 1024 --crop-size 720

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset ade20k \
    --model encnet  --aux --se-loss --pretrained \
    --backbone resnet101 --batch-size 16   \
    --checkpoint-path ./


CUDA_VISIBLE_DEVICES=4,5 python train.py --dataset pascal_aug \
    --model encnet  --aux --se-loss   --lr 0.001\
    --backbone resnet101 --checkname pcontext_v2  --batch-size 16 \
    --checkpoint-path ./
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset pascal_aug \
    --model encnet  --aux --se-loss   --lr 0.001 --attentiongraph --rank 9\
    --backbone resnet101 --checkname pcontext_v2  --batch-size 16 \
    --checkpoint-path ./

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset pcontext \
    --model encnet  --aux --se-loss   --lr 0.001 --attentiongraph --rank 9\
    --backbone resnet101  --batch-size 16 \
    --checkpoint-path ./

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset ade20k \
    --model encnet  --aux --se-loss   \
    --backbone resnet101   --pretrained\
    --checkpoint-path ./