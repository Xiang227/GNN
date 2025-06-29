FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   python3.10 \
                   python3-pip \
                   libgl1 \
                   zip \
                   wget \
                   python3.12-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir \
        torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
		--extra-index-url https://download.pytorch.org/whl/cu124 && \
    python3.12 -m pip install --no-cache-dir \
		torch_geometric \
		pyg_lib \
		torch_scatter \
		torch_sparse \
		torch_cluster \
		torch_spline_conv \
		torchdiffeq \
		-f https://data.pyg.org/whl/torch-2.5.0+cu124.html && \
	python3.12 -m pip install --no-cache-dir \
        accelerate \
        diffusers \
        huggingface-hub \
        scipy \
        transformers \
        matplotlib \
		pandas \
		sentencepiece \
		wandb \
		ogb \
		pykeops \
		numba \
		ray[tune] \
		optuna \
		hyperopt \
		tensorboard

CMD ["/bin/bash"]