FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN apt-get update
RUN apt-get install -y git libsm6 libxext6 libxrender-dev swig wget
RUN pip install --upgrade pip

RUN git clone https://github.com/alexus37/tf-pose-estimation.git
WORKDIR /tf/tf-pose-estimation
RUN pip install cython
RUN pip install -e .
WORKDIR /tf/tf-pose-estimation/tf_pose/pafprocess
RUN swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

WORKDIR /tf/tf-pose-estimation/models/graph/cmu
RUN bash download.sh

WORKDIR /tf
RUN pip install opencv-python websockets