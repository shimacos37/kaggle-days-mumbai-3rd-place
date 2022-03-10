# kaggle-days-mumbai-3rd-place

## shimacos part

- build envirionment

```bash
cd shimacos
docker build . -f images/Dockerfile.cuda11 -t kaggle:cuda11
```

- preprocess

```bash
cd shimacos
python scripts/make_fold.py
```

- preprocess

```train
cd shimacos
sh bin/exp001_xlm_roberta_large.sh
```
