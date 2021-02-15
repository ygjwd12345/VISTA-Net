
CUDA_VISIBLE_DEVICES=4 python test.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --resume ./pcontext/attentiongraph_res101_pcontext_rank_9/model_best.pth.tar --split val --mode testval  --batch-size 1

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --resume ./pcontext/attentiongraph_res101_pcontext_rank_9/model_best.pth.tar --split val --mode testval  --batch-size 1


CUDA_VISIBLE_DEVICES=1 python test.py --dataset cityscape \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --resume ./cityscape/attentiongraph_res101_pcontext_v2_rank_5_2020-08-10/model_best.pth.tar --split val --mode testval --ms --batch-size 1
CUDA_VISIBLE_DEVICES=4 python test.py --dataset ade20k \
    --model encnet  --aux --se-loss \
    --backbone resnet101 --resume ./ade20k/res101_pcontext_v2_rank_1_2020-08-13/model_best.pth.tar --split val --mode testval --ms --batch-size 1
CUDA_VISIBLE_DEVICES=4 python test.py --dataset cityscape \
    --model deeplab \
    --backbone resnet101 --resume ./cityscape/res101_pcontext_v2_rank_1_2020-08-16/model_best.pth.tar --split val --mode testval --ms --batch-size 1


CUDA_VISIBLE_DEVICES=4 python test.py --dataset coco \
    --model encnet --aux --se-loss \
    --backbone resnet101 --resume ./coco/encnet_res101_pcontext_v2_rank_0_2020-08-17/model_best.pth.tar --split test --mode testval --ms --batch-size 1

CUDA_VISIBLE_DEVICES=4 python test.py --dataset pascal_aug \
    --model encnet --aux --se-loss \
    --backbone resnet101 --resume ./pascal_aug/encnet_pcontext_v2_rank_0_2020-08-18/model_best.pth.tar  --mode val --ms --batch-size 1
CUDA_VISIBLE_DEVICES=4 python test.py --dataset pascal_aug \
    --model encnet --aux --se-loss --attentiongraph\
    --backbone resnet101 --resume ./pascal_aug/encnet_pcontext_v2_rank_9_2020-08-23/model_best.pth.tar  --mode val --ms --batch-size 1