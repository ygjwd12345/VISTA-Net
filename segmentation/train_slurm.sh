#srun python train.py --dataset pcontext --dataroot /datasets/Carla \
#    --model encnet --checkpoint_path /scratch/shared/nfs1/danxu/checkpoints/GatedAttention --attentiongraph --aux --se-loss \
#    --backbone resnet101 --checkname attentiongraph_res101_pcontext

#FCN head
srun python train.py --dataset pcontext --dataroot /datasets/Carla \
    --model encnet --checkpoint_path /scratch/shared/nfs1/danxu/checkpoints/GatedAttention/ --attentiongraph --aux --se-loss \
    --backbone resnet101 --checkname attentiongraph_res101_pcontext_v3
