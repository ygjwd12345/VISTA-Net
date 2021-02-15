# Depth preditcion
This repository contains PyTorch implementations of VISTA-Net for depth prediction.
The original code is from [bts](https://github.com/cogaplex-bts/bts). 
## Environment
First of all, you can use Dockerfile to built environment or pull from docker hub
```bash
docker pull ygjwd12345/pga:latest
docker run -it --rm --gpus all -v /path/to/VISTA-Net:/home pga
```
or use requirement.txt
```bash
pip install -r requirements.txt
```
## NYU dataset
```bash
cd ./pytorch
mkdir dataset
mkdir dataset/nyu_depth_v2
python ../utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ./dataset/nyu_depth_v2/sync.zip
cd dataset
cd nyu_depth_v2
unzip sync.zip
```
##  KITTI dataset
```bash
cd dataset
mkdir kitti_dataset
cd kitti_dataset
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip data_depth_annotated.zip
cd train
mv * ../
rm train
cd val
mv * ../
rm val
rm data_depth_annotated.zip
unzip '*.zip'
```
## Run
### train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_nyu.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python bts_main.py arguments_train_eigen.txt
```

### test
```bash
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_nyu.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_nyu_v2_pytorch_att/raw/ --gt_path ../../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
CUDA_VISIBLE_DEVICES=1 python bts_test.py arguments_test_eigen.txt
python ../utils/eval_with_pngs.py --pred_path vis_att_bts_eigen_v2_pytorch_att/raw/ --gt_path ./dataset/kitti_dataset/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
