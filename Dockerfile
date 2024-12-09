FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG http_proxy
ARG https_proxy

RUN apt-get update

RUN apt-get install -y \
    locales \
    build-essential \
    git \
    git-lfs \
    vim \
    cmake \
    pkg-config \
    zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget \
    liblzma-dev libsqlite3-dev libbz2-dev

RUN apt-get clean

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

ENV PYENV_ROOT=/root/.pyenv
ENV PATH="$PYENV_ROOT/bin/:$PATH"

RUN /usr/bin/echo -e '#!/bin/bash\neval "$(pyenv init -)"\neval "$(pyenv virtualenv-init -)"\ncd /moe_peft\nbash' | tee /opt/init.sh \
    && chmod +x /opt/init.sh \
    && /usr/bin/echo -e 'export PYENV_ROOT=/root/.pyenv' >> ~/.bashrc \
    && /usr/bin/echo -e 'export PATH=/root/.pyenv/bin:$PATH' >> ~/.bashrc \
    && /usr/bin/echo -e 'eval "$(pyenv init -)"' >> ~/.bashrc \
    && /usr/bin/echo -e 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc \
    && git clone https://github.com/pyenv/pyenv.git /root/.pyenv \
    && git clone https://github.com/pyenv/pyenv-virtualenv.git /root/.pyenv/plugins/pyenv-virtualenv \
    && cd /root/.pyenv && src/configure && make -C src \
    && eval "$(pyenv init -)" \
    && eval "$(pyenv virtualenv-init -)"

RUN . ~/.bashrc \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && git clone https://github.com/TUDB-Labs/MoE-PEFT /moe_peft \
    && cd /moe_peft \
    && pyenv virtualenv $PYTHON_VERSION moe_peft \
    && pyenv local moe_peft \
    && pip install -r ./requirements.txt --upgrade --no-compile --no-cache-dir

WORKDIR /moe_peft

CMD ["/bin/bash", "/opt/init.sh"]