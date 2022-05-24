this is a pre-label program based on yolox backbone!

someting detail in https://github.com/Megvii-BaseDetection/YOLOX

VOC datasets used in this program 

how to run this program:

<1>

python setup.py install

<2>

python tools/demo_online_2xml.py image -f exps/example/custom/yolox_s_custom_wsw.py -c your-weight-path --path your-prelabel-pictures --conf 0.3 --nms 0.65 --tsize 640 --device gpu

notice:

<1>in tools/demo_online_2xml.py line 161, give your xml save path 
