FROM rapidsai/rapidsai:0.18-cuda10.2-runtime-ubuntu18.04-py3.7

RUN apt-get update \
     && apt-get install --yes  --no-install-recommends \
     cuda-toolkit-10-2 \
     gcc \
     ssh-client \
     && apt-get clean

RUN ln -s /usr/local/cuda/bin/nvcc /opt/conda/envs/rapids/bin/ \
     && ln -s /usr/local/cuda/bin/nvcc.profile /opt/conda/envs/rapids/bin/ \
     && ln -s  /usr/local/cuda/nvvm /opt/conda/envs/rapids/
ENV CPATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$CPATH

RUN  source activate rapids \
     && pip install opencv-python \
     albumentations \
     tqdm \
     joblib \
     torch==1.7.0 \
     torchvision \
     efficientnet-pytorch \
     pretrainedmodels \
     segmentation-models-pytorch \
     pytorch-lightning \
     hydra-core \
     hydra_colorlog \
     google-cloud-storage \
     seaborn \
     plotly \
     nilearn \
     h5py \
     wandb \
     jupyterlab \
     jupyterlab_code_formatter \
     black \
     mypy \
     lightgbm \
     flake8 \
     timm \
     xgboost \
     pydicom \
     rising \
     threadpoolctl \
     mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html

# RUN source activate rapids \
#      && git clone https://github.com/open-mmlab/mmdetection.git \
#      && cd mmdetection \
#      && pip install -r requirements/build.txt \
#      && pip install -v -e .

RUN source activate rapids \
     && pip install transformers

RUN source activate rapids \
     && pip install kaggle \
     && apt install -y libopenmpi-dev \
     && pip install deepspeed mpi4py fairscale

RUN source activate rapids \
     && pip install --upgrade timm


# RUN source activate rapids \
#      && wget https://github.com/NVIDIA/TRTorch/releases/download/v0.2.0/trtorch-0.2.0-cp37-cp37m-linux_x86_64.whl \
#      && pip install trtorch-0.2.0-cp37-cp37m-linux_x86_64.whl nvidia-pyindex \
#      && pip install nvidia-tensorrt \
#      && rm trtorch-0.2.0-cp37-cp37m-linux_x86_64.whl

# RUN source activate rapids \
#      && pip install tensorflow-hub tensorflow-gpu

RUN apt install -y software-properties-common \
     && add-apt-repository ppa:ubuntu-toolchain-r/test \
     && apt update \
     && apt install --only-upgrade -y libstdc++6 \
     && apt clean
RUN source activate rapids \ 
     && pip install seqeval

RUN source activate rapids \ 
     && pip install fasttext
WORKDIR /root/workdir
