# CMMT Classify

## Install

```[bash]
conda create -n car1 python=3.10.15
conda activate car1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy pandas argparse Pillow
```

## Training

```[bash]
python train.py --version 4
```
