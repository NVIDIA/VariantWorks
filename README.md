# VariantWorks
VariantWorks is a framework for enabling development of deep learning based variant callers. It provides a
range of libraries spanning encoders, I/O modules and reference architectures that simplfy the boostrapping
of deep learning driven pipelines.

## Documentation
Detailed developer documentation is available at https://clara-parabricks.github.io/VariantWorks .

## Requirements

1.  Python 3.7+
2.  NVIDIA GPU (Pascal+ architecture)

## Installation
### Source
Download from GitHub and install locally

```
git clone --recursive https://github.com/clara-parabricks/VariantWorks.git
cd VariantWorks
pip install -r python-style-requirements.txt
pip install -r requirements.txt
pip install -e .
# Install pre-push hooks to run tests
ln -nfs $(readlink -f hooks/pre-push) .git/hooks/pre-push
```

## Unit Tests
```
cd tests
pytest -s .
```
