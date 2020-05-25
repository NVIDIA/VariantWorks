# VariantWorks
DL based variant calling toolkit

## Installation

```
conda install -c bioconda tabix
pip install -e .
ln -nfs $(readlink -f hooks/pre-push) .git/hooks/pre-push
```

## Test
```
cd test
pytest -s .
```
