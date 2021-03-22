# Deep Active Learning Toolkit for Image Classification in PyTorch

This is a code base for deep active learning for image classification written in [PyTorch](https://pytorch.org/). I want to emphasize that this toolkit is merely a lightweight derivative of the toolkit originally shared with me via email by Prateek Munjal _et al._, the authors of the paper _"Towards Robust and Reproducible Active Learning using Neural Networks"_, paper available [here](https://arxiv.org/abs/2002.09564).  

## Introduction

The goal of this repository is to provide a simple and flexible codebase for deep active learning. It is designed to support rapid implementation and evaluation of research ideas. We also provide a large collection of baseline results (coming soon).

The codebase currently only supports single-machine single-gpu training. We will soon scale it to single-machine multi-gpu training, powered by the PyTorch distributed package.
<!-- The codebase supports efficient single-machine multi-gpu training, powered by the PyTorch distributed package, and provides implementations of standard models including [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946), and [RegNet](https://arxiv.org/abs/2003.13678). -->

## Using the toolkit

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief installation instructions and basic usage examples.

## Active Learning Methods Supported
* Uncertainty Sampling
  * Least Confidence
  * Min-Margin
  * Max-Entropy
  * Deep Bayesian Active Learning (DBAL) [1]
  * Bayesian Active Learning by Disagreement (BALD) [1]
* Diversity Sampling 
  * Coreset (greedy) [2]
  * Variational Adversarial Active Learning (VAAL) [3]
* Query-by-Committee Sampling
  * Ensemble Variation Ratio (Ens-varR) [4]


## Datasets Supported
* [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [TinyImageNet](https://www.kaggle.com/c/tiny-imagenet) (Download the zip file [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip))


## Model Zoo

We provide a large set of baseline results as proof of repository's efficiency. (coming soon)


## Citing this repository

If you find this repo helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing us and the owners of the original toolkit:

```
@article{deepaltoolkit,
    Author = {Akshay L Chandra and Vineeth N Balasubramanian},
    Title = {Deep Active Learning Toolkit for Image Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2021}
}


@article{Munjal2020TowardsRA,
  title={Towards Robust and Reproducible Active Learning Using Neural Networks},
  author={Prateek Munjal and N. Hayat and Munawar Hayat and J. Sourati and S. Khan},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.09564}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## References

[1] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1183–1192. JMLR. org, 2017.

[2] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[3] Sinha, Samarth et al. Variational Adversarial Active Learning. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 5971-5980.

[4] William H. Beluch, Tim Genewein, Andreas Nürnberger, and Jan M. Köhler. The power of ensembles for active learning in image classification. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9368–9377, 2018.
