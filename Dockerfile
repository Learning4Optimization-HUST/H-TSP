ARG BASE_IMAGE=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM ${BASE_IMAGE} as base
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100"
###################
# python packages #
###################
RUN conda install --revision 0 &&\
    conda install python==3.8 -y &&\
    conda install pytorch=1.10.0 -c pytorch -c nvidia
RUN conda install ruamel.yaml flake8 yapf conda pip -y &&\
    conda install pyg -c pyg -c conda-forge -y &&\
    conda install -c dglteam dgl-cuda11.0 -y &&\
    ${PIP_INSTALL} pytorch-lightning==1.5.2 numba wandb hydra-core line-profiler tensorboardX scikit-learn revtorch tensor-sensor[torch] -i https://mirrors.ustc.edu.cn/pypi/web/simple
##################
## clean cache ###
##################
RUN conda clean --all -y && rm -rf /tmp/* /root/.cache/pip
RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL wget tar git
RUN wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz && \
    tar xvfz LKH-3.0.6.tgz && \
    cd LKH-3.0.6; make && \
    cp LKH /usr/local/bin &&\
    ${PIP_INSTALL} tsplib95 lkh word2vec &&\
    rm -rf /workspace/* /root/.cache/pip

FROM ${BASE_IMAGE} as dev
ENV LANG C.UTF-8
################################
# Install apt-get Requirements #
################################
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    apt-utils build-essential ca-certificates dpkg-dev pkg-config software-properties-common \
    cifs-utils openssh-server nfs-common net-tools iputils-ping iproute2 locales htop tzdata
# separate into several parts due to network issue
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    tar wget git swig vim curl tmux zip unzip rar unrar sudo zsh cmake &&\
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && rm -rf /var/lib/apt/lists/*

################
# Set Timezone #
################
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -f /etc/localtime && \
    rm -rf /usr/share/zoneinfo/UTC && \
    dpkg-reconfigure --frontend=noninteractive tzdata

COPY --from=base /opt/conda /opt/conda
COPY --from=base /usr/local/bin/LKH /usr/local/bin/LKH