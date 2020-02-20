FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="mitsuo"

RUN pip install -q keras seaborn torch torchvision tensorboardX torchsummary
ENV NB_USER mitsuo
ARG HOST_UID
ENV NB_UID ${HOST_UID}
RUN useradd -m -G sudo -u $NB_UID $NB_USER && \
   echo "${NB_USER}:password" | chpasswd && \
   echo 'Defaults visiblepw' >> /etc/sudoers && \
   echo 'mitsuo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $NB_USER
