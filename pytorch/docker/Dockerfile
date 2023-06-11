FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

LABEL maintainer "Mark - Auto generated"

ENV DEBIAN_FRONTEND=noninteractive

# Install standard tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils software-properties-common dirmngr build-essential \
    gpg-agent lsb-release git curl unzip wget gedit nano cmake \
    pkg-config checkinstall yasm libopenblas-dev

# install python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-tk python3-setuptools

# Update pip
RUN pip3 --no-cache-dir install --upgrade \
    pip setuptools

# Install basal python packages
RUN pip3 install sklearn gensim umap-learn pandas matplotlib

# OpenCV 4.4.0
RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev

# Get source from github
RUN git clone -b 4.4.0 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          .. && \
    make -j"$(nproc)" && \
    make install

# Install pytorch
#RUN pip3 install torch torchvision
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch-lightning
RUN pip3 install pytorch-lightning pytorch-lightning-bolts wandb gym tensorboard albumentations

RUN rm -rf /var/lib/apt/lists/*

COPY user.sh /usr/local/bin/
CMD bash -C '/usr/local/bin/user.sh'
