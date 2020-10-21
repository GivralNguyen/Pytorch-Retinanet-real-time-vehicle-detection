#Installation
Requirements:
PyTorch 1.0 from a nightly release. It will not work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
torchvision from master
cocoapi
yacs
matplotlib
GCC >= 4.9
OpenCV
CUDA >= 9.0
Option 1: Step-by-step installation
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark -y
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
Windows 10
open a cmd and change to desired installation directory
from now on will be refered as INSTALL_DIR
conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
## Important : check the cuda version installed on your computer by running the command in the cmd :
nvcc -- version
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

git clone https://github.com/cocodataset/cocoapi.git

    #To prevent installation error do the following after commiting cocooapi :
    #using file explorer  naviagate to cocoapi\PythonAPI\setup.py and change line 14 from:
    #extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    #to
    #extra_compile_args={'gcc': ['/Qstd=c99']},
    #Based on  https://github.com/cocodataset/cocoapi/issues/51

cd cocoapi/PythonAPI
python setup.py build_ext install

# navigate back to INSTALL_DIR
cd ..
cd ..
# install apex

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
# navigate back to INSTALL_DIR
cd ..
# install PyTorch Detection

git clone https://github.com/Idolized22/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
#How to inference on a single image:
##1. change config_file path in demo/run.py to the path of retinanet_Mobilenetv2-96-FPN_1x.yaml
##2. go to retinanet_MobileNetV2-96-FPN_1x.yaml. Change OUTPUT_DIR to the path of 320x320_5class_fpn8000.
##3. Run run.py 
