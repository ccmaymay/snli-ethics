FROM centos:7

RUN yum update -y && yum clean all # cache bust 20170301

RUN yum install -y \
        atlas \
        atlas-devel \
        autoconf \
        automake \
        make \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        git \
        numpy \
        python \
        python-devel \
        scipy \
        tar \
        unzip \
        wget

RUN curl https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade setuptools && \
    pip install --upgrade setuptools && \
    pip install --upgrade tox

RUN useradd -m -U -s /bin/bash snli && \
    passwd -l snli
ADD . /home/snli/snli-ethics
RUN chown -R snli:snli /home/snli

USER snli
WORKDIR /home/snli/snli-ethics
