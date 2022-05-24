wsw
---------------------
nvidia-smi.exe
============================================
[从头训练]
GPU = 24G
python tools/train.py -f exps/example/custom/yolox_s_custom_wsw.py -d 1 -b 18 --fp16 -o -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_20211223_2.pth

python tools/train.py -f exps/example/custom/yolox_s_custom_wsw.py -d 1 -b 18 --fp16 -o -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth

python tools/train.py -f exps/example/custom/yolox_s_custom_wsw.py -d 1 -b 18 --fp16 -o -c weights/yolox_s.pth

===========================================================================================================================

[继续训练]
###
GPU=24G
python tools/train.py -f exps/example/custom/yolox_s_custom_wsw.py -d 1 -b 18 --fp16 -o -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_20211223_1.pth
###
python tools/train.py -f exps/example/custom/yolox_s_custom_wsw.py -d 1 -b 18 --fp16 -o -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth
-------------------------------------------
============================================
[查看训练结果]

cd YOLOX_outputs/yolox_s_custom_wsw
cd D:/AI/YOLOX-main/YOLOX_outputs/yolox_s_custom_wsw

tensorboard --logdir=tb/
---------------------------------------------
==================================================
[测试][本地]
python tools/demo_online.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path assets/wsw/pictures --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu

python tools/demo_online_2xml.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path C:/Users/chuxi/Desktop/test/ --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu

python tools/demo.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path assets/wsw/pictures --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu

python tools/demo_online.py webcam -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --conf 0.3 --nms 0.65 --tsize 640

python tools/demo_online_CSGO.py video -f exps/example/custom/yolox_nano_custom_CSGO.py -c YOLOX_outputs/yolox_nano_custom_CSGO/yolox_nano_custom_CSGO_20220117.pth --conf 0.3 --nms 0.65 --tsize 640 --device gpu

python tools/demo_online_CSGO.py video -f exps/example/custom/nano.py -c weights/yolox_nano.pth --conf 0.3 --nms 0.65 --tsize 640 --device gpu
---------------------------------------------------------
[预标注]
‘记得改标注文件保存地址’
python tools/demo_online_2xml.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path D:/AI/YOLOX-main/datasets/WSW/wsw_yuantu/yauntu/Augly_JPEGImages/sum/ --fuse --conf 0.2 --nms 0.55 --tsize 640 --device gpu

python tools/demo_online_2xml.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path C:/Users/chuxi/Desktop/test/ --conf 0.3 --nms 0.65 --tsize 640 --device gpu
python tools/demo_online_2xml.py image -f exps/example/custom/yolox_s_custom_wsw.py -c YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_RTX3090_20220105.pth --path D:/AI/YOLOX-main/datasets/WSW/wsw_yuantu/yauntu/00previous_datasets/picture_xml_20220427/2/ --conf 0.3 --nms 0.65 --tsize 640 --device gpu


--
--------------------------------------------------------

[加速]

tar xzvf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz
----------------------------------------------------------------------------
vim ~/.bashrc
export LD_LIBRARY_PATH=/home/ps/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6/TensorRT-7.0.0.11/lib:${LD_LIBRARY_PATH}

cd TensorRT-7.0.0.11/python
pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl

cd TensorRT-7.0.0.11/graphsurgeon
pip install  graphsurgeon-0.4.1-py2.py3-none-any.whl

python
import tensorrt
tensorrt.__version__

git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install

import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))

=================================================
=================================================
export LD_LIBRARY_PATH=/user-data/TensorRT_ubuntu/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH

==================================================

python tools/trt.py -f /user-data/YOLOX_wsw/exps/example/custom/yolox_s_custom_wsw.py -c /user-data/YOLOX_wsw/YOLOX_outputs/yolox_s_custom_wsw/yolox_s_custom_wsw_8class_20211223_2.pth


python tools/demo.py image -f /user-data/YOLOX_wsw/exps/example/custom/yolox_s_custom_wsw.py -c /user-data/YOLOX_wsw/YOLOX_outputs/yolox_s_custom_wsw/model_trt.pth --trt --path /user-data/YOLOX_wsw/assets/wsw/pictures --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu

python tools/demo.py video -f /user-data/YOLOX_wsw/exps/example/custom/yolox_s_custom_wsw.py -c /user-data/YOLOX_wsw/YOLOX_outputs/yolox_s_custom_wsw/model_trt.pth --trt --path /user-data/YOLOX_wsw/assets/wsw/video --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu


===============================================
