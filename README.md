# FRCNN-on-ResNet-50-detection
Faster-RCNN on ResNet-50 detection model
* To build prerequisites, run:
```
python3 setup.py develop
```
* Run <tt>inference.py</tt> script to test **FRCNN on ResNet-50** detection model:
```
python3 inference.py
```
Default image for classification is <tt>data/cat.jpg</tt>. To specify the image use <tt>--path_img</tt> key: 
```
python3 inference.py --path_img data/croco.jpg              
```
