# semantic segmentation

## Prepare pascal context dataset

First way is running `./scripts/prepare_pcontext.py`.

Seconde way is just download from [google drive](https://drive.google.com/open?id=13TLw6TR22K8CwUOOLEvPyOJ9SnjUg0Tx) (about 2.42GB).

Third it also auto down load and prepocess but expend huge time.
## Prepare pascal dataset
 running `./scripts/prepare_pascal.py`.
 
## Prepare cityscape dataset
 running `./scripts/prepare_citys.py`.
## Train



 for mult GPU, we recommand is `at least 4 GPU at least 24 GB`. at least `2 GPU at least 11GB`
 
 ```bash
 CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --checkname attentiongraph_res101_pcontext_v2
 ```
## Test
 
 ```bash
 python test.py --dataset pcontext \
    --model encnet --attentiongraph --aux --se-loss \
    --backbone resnet101 --resume ./pcontext/attentiongraph_res101_pcontext_v2/model_best.pth.tar --split val --mode testval --ms
 ```