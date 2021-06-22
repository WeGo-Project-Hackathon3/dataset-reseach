* train google colab custom dataset

```from google.colab import drive```

```drive.mount('/content/drive/', force_remount=True)```

\# clone darknet repo

```!git clone https://github.com/AlexeyAB/darknet```

\# change makefile to have GPU and OPENCV enabled

```python
%cd darknet

!sed -i 's/OPENCV=0/OPENCV=1/' Makefile

!sed -i 's/GPU=0/GPU=1/' Makefile

!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```



```python
import tensorflow as tf
# 2.5.0
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```

```python
# make darknet (builds darknet so that you can then use the darknet executable file to run or train object detectors)
!make
```

```python
#make backup folder
#/content/MyDrive/yolov4/backup
```



```python
# make train/test folder
mkdir /content/darknet/data/obj
mkdir /content/darknet/data/test
```

```python
# copy preprocess img data to darknet
# move img zip data to obj, test folder, unzip
!cp /content/drive/*.zip ./
! mv ./test.zip ./data/test
# mv otherzip file(trainset)
# Error Occurred, unzip/ unzip each file under obj or test(jpg-txt pair)
! mv ./train.zip ./data/obj
! unzip /content/darknet/data/test/test.zip -d data/test
! unzip /content/darknet/data/train/train.zip -d data/train
```

```python
# download cfg to google drive and change its name
!cp cfg/yolov4-custom.cfg /content/drive/yolov4/yolov4-obj.cfg
# changename yolov4-obj_1.cfg
```

* obj.names and obj.data

   

  obj.names - edit txt word nums your classes

  ex) person

  ​	  dog

  obj.data - ㅡmatch path

  classes  =1

  train = data/train.txt

  valid = data/test.txt

  names = data/obj.names

  backup = /mydrive/yolov4/backup

* edit cfg file

  I recommend having **batch = 64** and **subdivisions = 16** for ultimate results. If you run into any issues then up subdivisions to 32.

  Make the rest of the changes to the cfg based on how many classes you are training your detector on.

  **Note:** I set my **max_batches = 6000**, **steps = 4800, 5400**, I changed the **classes = 1** in the three YOLO layers and **filters = 18** in the three convolutional layers before the YOLO layers.

  How to Configure Your Variables:

  width = 416

  height = 416 **(these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)**

  max_batches = (# of classes) * 2000 **(but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)**

  steps = (80% of max_batches), (90% of max_batches) **(so if your max_batches = 10000, then steps = 8000, 9000)**

  filters = (# of classes + 5) * 3 **(so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)**

  **Optional:** If you run into memory issues or find the training taking a super long time. In each of the three yolo layers in the cfg, change one line from random = 1 to **random = 0** to speed up training but slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

* gathering train.txt, test.txt

 last configuration files needed before we can begin to train our custom detector are the train.txt and test.txt files which hold the relative paths to all our training images and valdidation images.

Luckily I have created scripts that eaily generate these two files withe proper paths to all images.

The scripts can be accessed from the [Github Repo](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)

Just download the two files to your local machine and upload them to your Google Drive so we can use them in the Colab Notebook.

```python
# upload the generate_train.py and generate_test.py script to cloud VM from Google Drive
!cp /mydrive/yolov4/generate_train.py ./
!cp /mydrive/yolov4/generate_test.py ./
!python generate_train.py
!python generate_test.py
# check txt file 
```

```python
# download darknet yolov4-tiny pretrained weight conv29
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

```python
# train your custom detector! (uncomment %%capture below if you run into memory issues or your Colab is crashing)
# %%capture
# train from pretrained weigth(*.weight)  >> -clear
!./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
```

* input code avoid colab runtime error ( google search)

```python
# map data
!./darknet detector map data/obj.data cfg/yolov4-tiny-onlyai.cfg /content/drive/MyDrive/yolov4/backup/yolov4-tiny-onlyai_best.weights
```

```python
!./darknet detector test <path to obj.data> <path to custom config> <weights path> <image path> -thresh 0.5
imShow('predictions.jpg')
```



* **YOLOv4 Using Tensorflow (tf, .pb model)**

  파라미터는 사이트를 참조하여 자신의 custom weights 와 class 에 맞춰서 하면 된다.

  (학습한 class를 작성한 .names 파일을 classes 폴더에 넣고, core 폴더의 [config.py](http://config.py/) 에 .names 파일을 14line에 지정함.)

  ```
  # yolov4-tiny
  python save_model.py --weights ./data/yolov4-tiny-50000.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny
  # Run yolov4-tiny tensorflow model
  python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --images ./data/images/kite.jpg --tiny
  ```





* reference 

https://www.youtube.com/watch?v=mmj3nxGT2YQ

https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing
