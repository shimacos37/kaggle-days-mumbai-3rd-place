# kaggle-days-mumbai-3rd-place

## sakami part

Put the competition data `sakami/input/`.

```shell
cd sakami/
python feature_classification.py
python lm_classification.py
```

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

- train

```bash
cd shimacos
sh bin/exp001_xlm_roberta_large.sh
```
