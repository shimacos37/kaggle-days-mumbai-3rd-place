docker run --rm -it \
  -v $PWD/:/root/workdir/ \
  -v $HOME/.config/:/root/.config \
  -v $HOME/.netrc/:/root/.netrc \
  -v $HOME/.cache/:/root/.cache \
  -v $HOME/.git/:/root/.git \
  -e SLURM_LOCALID=0 \
  --runtime=nvidia \
  --shm-size=600gb \
  --ipc=host \
  --security-opt seccomp=unconfined \
  kaggle:cuda11 \
  python main_nn.py \
          data.n_fold=0 \
          data.is_train=True \
          store.model_name=$(basename $0 .sh) \
          train.epoch=50 \
          train.batch_size=4 \
          train.warm_start=False \
          test.batch_size=8 

docker run --rm -it \
  -v $PWD/:/root/workdir/ \
  -v $HOME/.config/:/root/.config \
  -v $HOME/.netrc/:/root/.netrc \
  -v $HOME/.cache/:/root/.cache \
  -v $HOME/.git/:/root/.git \
  -e SLURM_LOCALID=0 \
  --runtime=nvidia \
  --shm-size=600gb \
  --ipc=host \
  --security-opt seccomp=unconfined \
  kaggle:cuda11 \
  python main_nn.py \
          data.n_fold=0 \
          data.is_train=False \
          store.model_name=$(basename $0 .sh) \
          train.epoch=50 \
          train.batch_size=4 \
          train.warm_start=True \
          test.batch_size=8 