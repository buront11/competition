FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV PYTHON_VERSION 3.8.5
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
# timezone setting
ENV TZ=Asia/Tokyo 

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN python --version

WORKDIR /competition
COPY requirements.txt /competition
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs