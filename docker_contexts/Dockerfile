# FROM ubuntu:20.04

# WORKDIR /opt/nuclio

# ENV TZ=Europe/Kiev
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt update

# RUN apt install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y python3.9 python3.9-distutils libpython3.9-dev

# RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0  gnupg g++ libusb-1.0-0 wget

# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && python3.9 -m pip --version

# RUN ln -s /usr/bin/python3.9 /usr/local/bin/python

# RUN pip install torch torchvision torchaudio tqdm

# RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && \
#     pip install -r requirements.txt && pip install .

# RUN mkdir weights && cd weights && \
#     wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

FROM cvat.pytorch.idea.dino.base

RUN rm /usr/bin/python3 && ln -s python3.9 /usr/bin/python3

# docker build --no-cache -f Dockerfile -t cvat.pytorch.idea.dino.base . > build.log 2>&1

