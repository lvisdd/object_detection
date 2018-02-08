## Tensorflow Object Detection API Sample For Movie and Webcam

For more details, see [Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md)

# Dependencies

* Windows 10 64Bit
* Python 3.5/3.6
* Tensorflow 1.4/CUDA 8.0/cuDNN v6
* Tensorflow 1.5/CUDA 9.0/cuDNN v7

# Create an environment

``` dos
## For Python 3.6
chcp 65001
conda create --name=objectdetectionenv python=3.6

## For Python 3.5
conda create --name=objectdetectionenv python=3.5

## Activate an environment
activate objectdetectionenv
```

# Install dependencies

``` dos
pip install tensorflow-gpu
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install opencv-python
```

# Cloning TensorFlow Models

``` dos
mkdir c:\work
cd c:\work
git clone https://github.com/tensorflow/models.git
cd c:\work\models\research
```

# Protobuf Compilation

``` dos
## https://repo1.maven.org/maven2/com/google/protobuf/protoc/3.4.0/
## protoc-3.4.0-windows-x86_64.exe -> c:\tools\bin\protoc.exe

## c:\tools\bin\protoc.exe object_detection/protos/*.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/anchor_generator.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/argmax_matcher.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/bipartite_matcher.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/box_coder.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/box_predictor.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/eval.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/faster_rcnn.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/faster_rcnn_box_coder.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/grid_anchor_generator.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/hyperparams.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/image_resizer.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/input_reader.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/keypoint_box_coder.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/losses.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/matcher.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/mean_stddev_box_coder.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/model.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/optimizer.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/pipeline.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/post_processing.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/preprocessor.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/region_similarity_calculator.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/square_box_coder.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/ssd.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/ssd_anchor_generator.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/string_int_label_map.proto --python_out=.
c:\tools\bin\protoc.exe object_detection/protos/train.proto --python_out=.
```

# Running

``` dos
cd object_detection

## For Movie
## ./test.mp4 -> [Your Movie Filename]
python object_detection_for_movie.py

## For Webcam
python object_detection_for_webcam.py
```
