FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

# COPY . .

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update

RUN apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y python3.10

RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0  gnupg g++ libusb-1.0-0 wget

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && python3.10 -m pip --version

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install torch torchvision torchaudio

RUN pip install onemetric azureml-sdk azureml-mlflow mlflow tqdm

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

RUN apt install -y libpython3.10-dev

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && \
    pip install -r requirements.txt && pip install -e .

RUN mkdir weights && cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# docker build --no-cache -f dino/Dockerfile -t dino_base . > build_base.log 2>&1