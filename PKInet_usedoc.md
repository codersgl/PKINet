```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

```bash
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
```

```bash
git clone https://gh-proxy.org/https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

```bash
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
python demo/image_demo.py demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
```

```bash
git clone https://gh-proxy.org/https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet.git
cd PKINet/
mkdir data
cd data
mkdir DOTA
```

```bash
pip install shapely
pip install mmengine
pip install -e git+https://hk.gh-proxy.org/https://github.com/open-mmlab/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
pip install -e git+https://hk.gh-proxy.org/https://github.com/QUVA-Lab/e2cnn.git#egg=e2cnn
pip install -r requirements.txt
pip install -v -e .
```



```bash
oss cp oss://datasets/DOTA.zip /hy-tmp/projects/PKINet/data/DOTA
cd data/DOTA
unzip DOTA.zip
```

将images重命名为images1

```
cd test
unzip images1/"part*.zip"
```

```bash
cd train
unzip images1/"part*.zip"
cd labelTxt-v1.0
unzip labelTxt.zip
cd labelTxt-v1.5/
unzip DOTA-v1.5_train.zip
```

```bash
cd val
unzip images1/part1.zip
cd labelTxt-v1.0
unzip labelTxt.zip
cd labelTxt-v1.5/
unzip DOTA-v1.5_val.zip
```

```python
cd data
mkdir DOTAv15
cp -r DOTA/train/images DOTAv15/train/
mv DOTA/train/labelTxt-v1.5 DOTAv15/train/
cp -r DOTA/val/images DOTAv15/val/
mv DOTA/val/labelTxt-v1.5 DOTAv15/val/
cp -r DOTA/test/images DOTAv15/test/
```



修改/hy-tmp/projects/PKINet/tools/data/dota/split/split_configs下的json文件中的`img_dirs`和`ann_dirs`为数据路径

执行

```bash
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```

修改/hy-tmp/projects/PKINet/configs/_base_/datasets/dotav1.py中的data_root，train_dic, val_dic, test_dic为实际路径，

添加预训练权重

## DOTAv1-T

### 训练：

```bash
tmux new -s train
conda activate openmmlab
chmod +x ./tools/dist_train.sh
./tools/dist_train.sh configs/pkinet/pkinet-t_fpn_o-rcnn-dotav1-ss_le90.py 4
# 退出tmux
Ctrl + B   然后按   D
# 重新进入
tmux attach -t train
# 查看所有会话
tmux ls
# 关闭 tmux 会话（并终止训练）
tmux kill-session -t train
```

### 测试

```bash
tmux new -s test
conda activate openmmlab
chmod +x ./tools/dist_test.sh
./tools/dist_test.sh configs/pkinet/pkinet-t_fpn_o-rcnn-dotav1-ss_le90.py work_dirs/pkinet-t_fpn_o-rcnn-dotav1-ss_le90/best_mAP_epoch_21.pth 4 --format-only --eval-options submission_dir=work_dirs/Task1_results
# 退出tmux
Ctrl + B   然后按   D
# 重新进入
tmux attach -t test
# 查看所有会话
tmux ls
# 关闭 tmux 会话（并终止训练）
tmux kill-session -t test
```

## DOTAv1-S

```bash
tmux new -s train
conda activate openmmlab
chmod +x ./tools/dist_train.sh
./tools/dist_train.sh configs/pkinet/pkinet-s_fpn_o-rcnn-dotav1-ss_le90.py 4
# 退出tmux
Ctrl + B   然后按   D
# 重新进入
tmux attach -t train
# 查看所有会话
tmux ls
# 关闭 tmux 会话（并终止训练）
tmux kill-session -t train
```

```bash
tmux new -s test
conda activate openmmlab
chmod +x ./tools/dist_test.sh
./tools/dist_test.sh configs/pkinet/pkinet-s_fpn_o-rcnn-dotav1-ss_le90.py work_dirs/pkinet-s_fpn_o-rcnn-dotav1-ss_le90/best_mAP_epoch_30.pth 4 --format-only --eval-options submission_dir=work_dirs/Task1_results_v1_s
# 退出tmux
Ctrl + B   然后按   D
# 重新进入
tmux attach -t test
# 查看所有会话
tmux ls
# 关闭 tmux 会话（并终止训练）
tmux kill-session -t test
```

## DOTAv15-S

```bash
tmux new -s train
conda activate openmmlab
chmod +x ./tools/dist_train.sh
./tools/dist_train.sh configs/pkinet/pkinet-s_fpn_o-rcnn-dotav15-ss_le90.py 4
# 退出tmux
Ctrl + B   然后按   D
# 重新进入
tmux attach -t train
# 查看所有会话
tmux ls
# 关闭 tmux 会话（并终止训练）
tmux kill-session -t train
```

```python
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
```

```bash
chmod +x ./tools/dist_test.sh
./tools/dist_test.sh configs/pkinet/pkinet-s_fpn_o-rcnn-dotav15-ss_le90.py work_dirs/pkinet-s_fpn_o-rcnn-dotav15-ss_le90/best_mAP_epoch_30.pth 4 --format-only --eval-options submission_dir=work_dirs/Task1_results_v15_s
```

