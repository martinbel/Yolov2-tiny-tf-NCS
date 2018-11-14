# Yolov2 tiny tensorflow for Movidius Neural Compute Stick (NCS)

This is a Tensorflow/NCSdk implementation of Yolov2. It was inpired by simo23 repo, see: https://github.com/simo23/tinyYOLOv2.

Credits to him for the functions to load the .weights file and functions to process the outcome of the Network.

I made this repo to make a tensorflow implementation of Yolov2 possible for the Movidius NCS, because until now I've only seen caffe implementations of Yolo for the NCS.

### How to use it on the NCS
- clone the repo on a linux 16.04 system
- install ncsdk2
- run the following command in the terminal:
```python
python3 Script.py
make compile
python3 run.py -m [video/image] [-i PathToImage] [-v PathToVideo] [-v 0] #Last one is for using your webcam
```
- and done! The output of the network will look something like this:

![alt text](https://github.com/bastiaanv/Yolov2-tiny-tf-NCS/blob/master/images/test.jpg "YOLOv2-tiny output")

### How to use it normally / in debug mode
- clone the repo
- run the following script with python3:
```python
Script.py
```
- and done! The output should be the same as on the NCS

### Freeze graph
cd model
python3 ../freeze_graph.py --input_graph=NN.pb \
           --input_binary=true \
           --input_checkpoint=NN.ckpt \
           --output_graph=NN_frozen.pb \
           --output_node_name=Output;

### Start docker and add folder with models
docker start -a -i ncsdk
docker run -v /mnt/yolo_retrain:/mnt/yolo_retrain -i -t ncsdk

### Compile
source /opt/movidius/virtualenv-python/bin/activate
mvNCCompile -s 12 NN_frozen.pb -in=Input -on=Output




### Note!
The processing power of the NCS is not as good as 4 titan X (pascal) GPU's, ofcourse. That is why you will get a processing time of ~6Hz / ~170ms, instead of 220Hz like advertised on the Yolov2 page. The biggest lost in preformance is made in conv7 and conv8:

![alt text](https://github.com/bastiaanv/Yolov2-tiny-tf-NCS/blob/master/images/Preformance%20lost.png "Preformance lost")

If anybody can help me improve this lost, it would be much appreciated!! Any other sugestions and questions are welcome as well! You can contact me on this repo or at: verhaar.bastiaan@gmail.com
