# Scene Recognition with a CNN

A convolutional neural network trained to classify images into one of **100 scene categories** from the [MIT Mini Places](http://places2.csail.mit.edu/) dataset. The model uses no pretrained weights or outside data and reaches about **42% validation accuracy**.

## Model

The network (`MyConv`) is a small CNN with about **415K parameters**. It has two parts:

**Feature extractor:** Four convolutional blocks, and each applies:

1. a 3×3 convolution,
2. batch normalization,
3. a ReLU activation,
4. 2×2 max pooling, which halves the image size.

The number of channels grows from 32 to 64 to 128 to 256. The spatial size shrinks from 128×128 to 8×8.

**Classifier:** Global average pooling keeps the model small and turns each of the 256 feature maps into a single number. Dropout (p=0.5) is applied, then a fully connected layer outputs scores for the 100 classes.

## Training Setup

| Setting | Value |
|---|---|
| Input size | 128 × 128, normalized with ImageNet mean/std |
| Loss | Cross-entropy with label smoothing (0.1) |
| Optimizer | Adam (learning rate 0.001, weight decay 0.0005) |
| Learning-rate schedule | Cosine annealing over all epochs |
| Epochs | 40 |
| Batch size | 64 |
| Precision | Mixed precision (`torch.amp`) for faster training and lower memory use |

After every epoch, the script reports loss and accuracy on both the training and validation sets. The best epoch overall is saved as the final checkpoint to `model.ckpt`.

## Results

Accuracy reached 42% on the validation split.

## Getting Started

### 1. Set up the environment

```bash
conda create --name pytorch python=3.9
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tqdm pillow
```

### 2. Get the data

```bash
wget https://web.cs.ucla.edu/~smo3/data.tar.gz
tar -xzf data.tar.gz
```

Then move the extracted `train/`, `val/` and `test/` folders into `data/images/`.

### 3. Train

```bash
python scene_classification.py
```

This trains for 40 epochs and saves the best model to `model.ckpt`.

### 4. Predict on the test set

```bash
python scene_classification.py --test
```

This loads `model.ckpt` and writes `predictions.csv`. Each line holds an image path and its predicted class index.
