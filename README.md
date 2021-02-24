# Deep Active Learning Toolkit for Image Classification in PyTorch

This is a code base for deep active learning for image classification written in [PyTorch](https://pytorch.org/). 

## Introduction

The goal of this repository is to provide a simple and flexible codebase for deep active learning. It is designed to support rapid implementation and evaluation of research ideas. We also provide a large collection of baseline results ([Model Zoo](MODEL_ZOO.md)).

The codebase currently only supports single-machine training. 
<!-- The codebase supports efficient single-machine multi-gpu training, powered by the PyTorch distributed package, and provides implementations of standard models including [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946), and [RegNet](https://arxiv.org/abs/2003.13678). -->

## Using the toolkit

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief installation instructions and basic usage examples.

## Model Zoo

We provide a large set of baseline results as proof of repository's efficiency.

## Active Learning Methods Supported
* Least Confidence
* Margin
* Entropy
* Deep Bayersian AL
* Coreset (greedy)
* Ensemble Variation Ratio
* Variational Adversarial Active Learning


## Citing this repository

If you find this repo helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing an appropriate subset of the following papers:

```
@article{daltoolkit,
    Author = {Akshay L Chandra and Vineeth N Balasubramanian},
    Title = {Deep Active Learning Toolkit for Image Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2021}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](docs/CODE_OF_CONDUCT.md) for more info.
