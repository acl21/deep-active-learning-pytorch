# Getting Started

## Environment Setup

Clone the repository:

```
git clone https://github.com/acl21/deep-active-learning-pytorch
```

Install dependencies:

```
pip install -r requirements.txt
```

### Config File (Very Important)
```
# Folder name where best model logs etc are saved. "auto" creates a timestamp based folder
EXP_NAME: 'SOME_RANDOM_NAME'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
RNG_SEED: 1
# GPU ID you want to execute the process on (this isn't working as of now, use the commands shown in this file below instead)
GPU_ID: '3'
DATASET:
  NAME: CIFAR10 # or CIFAR100, MNIST, SVHN, TinyImageNet
  ROOT_DIR: 'data' # Relative path where data should be downloaded
  # Specifies the proportion of data in train set that should be considered as the validation data
  VAL_RATIO: 0.1
  # Data augmentation methods - 'simclr', 'randaug', 'horizontalflip'
  AUG_METHOD: 'horizontalflip' 
MODEL:
  # Model type. 
  # Choose from vgg style ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]
  # or from resnet style ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
  # 'wide_resnet50_2', 'wide_resnet101_2']
  TYPE: resnet18
  NUM_CLASSES: 10
OPTIM:
  TYPE: 'sgd' # or 'adam'
  BASE_LR: 0.1
  # Learning rate policy select from {'cos', 'exp', 'steps'}
  LR_POLICY: steps
  # Steps for 'steps' policy (in epochs)
  STEPS: [0] #[0, 30, 60, 90]
  # Training Epochs
  MAX_EPOCH: 1
  # Momentum
  MOMENTUM: 0.9
  # Nesterov Momentum
  NESTEROV: False
  # L2 regularization
  WEIGHT_DECAY: 0.0005
  # Exponential decay factor
  GAMMA: 0.1
TRAIN:
  SPLIT: train
  # Training mini-batch size
  BATCH_SIZE: 256
  # Image size
  IM_SIZE: 32
  IM_CHANNELS = 3
  # Evaluate model on test data every eval period epochs
  EVAL_PERIOD: 1
TEST:
  SPLIT: test
  # Testing mini-batch size
  BATCH_SIZE: 200
  # Image size
  IM_SIZE: 32
  # Saved model to use for testing (useful when running `tools/test_model.py`)
  MODEL_PATH: ''
DATA_LOADER:
  NUM_WORKERS: 4
CUDNN:
  BENCHMARK: True
ACTIVE_LEARNING:
  # Active sampling budget (at each episode)
  BUDGET_SIZE: 5000
  # Active sampling method
  SAMPLING_FN: 'dbal' # 'random', 'uncertainty', 'entropy', 'margin', 'bald', 'vaal', 'coreset', 'ensemble_var_R'
  # Initial labeled pool ratio (% of total train set that should be labeled before AL begins)
  INIT_L_RATIO: 0.1
  # Max AL episodes
  MAX_ITER: 1
  DROPOUT_ITERATIONS: 10 # Used by DBAL
# Useful when running `tools/ensemble_al.py` or `tools/ensemble_train.py`
ENSEMBLE: 
  NUM_MODELS: 3
  MODEL_TYPE: ['resnet18']
```

Please refer to `pycls/core/config.py` to configure your experiments at a deeper level. 

### Active Learning
Once the config file is configured appropriately, perform active learning with the following command. 

```
CUDA_VISIBLE_DEVICES=0 python tools/train_al.py \
    --cfg configs/cifar10/al/RESNET18_DBAL.yaml
```

### Ensemble Active Learning 

Watch out for the ensemble options in the config file.

```
CUDA_VISIBLE_DEVICES=0 python tools/ensemble_al.py \
    --cfg configs/cifar10/al/RESNET18_ENSEMBLE.yaml
```

### Passive Learning

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg configs/cifar10/train/RESNET18.yaml
```

### Ensemble Passive Learning

Watch out for the ensemble options in the config file.

```
CUDA_VISIBLE_DEVICES=0 python tools/ensemble_train.py \
    --cfg configs/cifar10/train/RESNET18_ENSEMBLE.yaml
```

### Specific Model Evaluation

This is useful if you want to evaluate a particular saved model. 

```
CUDA_VISIBLE_DEVICES=0 python tools/test_model.py \
    --cfg configs/cifar10/evaluate/RESNET18.yaml
```

Refer to the corresponding yaml files for more clarity. 