#test [single-scale]
#CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
#    --model encnet --jpu --aux --se-loss \
#    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext/model_best.pth.tar --split val --mode testval
#
#test [multi-scale][resnet-50]
#CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
#    --model encnet --jpu --aux --se-loss \
#    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext/model_best.pth.tar --split val --mode testval --ms

#test [multi-scale][resnet-101]
#CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
#    --model encnet --jpu --aux --se-loss \
#    --backbone resnet101 --resume runs/pcontext/encnet/encnet_res101_pcontext/model_best.pth.tar --split val --mode test --ms

#test [multi-scale][resnet-101]
#CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
#    --model fcn \
#    --backbone resnet101 --resume runs/pcontext/fcn/fcn_res101_pcontext/model_best.pth.tar --split val --mode test --ms

#test [multi-scale][resnet-101]
srun python test.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --resume /scratch/shared/nfs1/danxu/checkpoints/GatedAttention/pcontext/encnet/attentiongraph_res101_pcontext_v3model_best.pth.tar --split val --mode test --ms
