# Beyond Debiasing: Actively Steering Feature Selection via Loss Regularization

* [Overview](#overview)
* [Installation](#installation)
* [Usage in Python](#usage-in-python)
* [Repository organization](#repository)
* [License and support](#license-support)

## Overview <a name="overview"></a>

This repository provides code to use the method presented in our DAGM GCPR 2023 paper [**"Beyond Debiasing: Actively Steering Feature Selection via Loss Regularization"**](https://pub.inf-cv.uni-jena.de/pdf/Blunk23:FS). If you want to get started, take a look at our [example network](regression_network.py) and the corresponding [jupyter notebook](feature_steering_example.ipynb).

If you are only interested in the implementation of the feature steering part of the loss, you can find it in `feat_steering_loss(...)` of [regression_network.py](regression_network.py).

<div align="center">
    <img src="https://git.inf-cv.uni-jena.de/blunk/beyond-debiasing/raw/main/teaser.png" alt="By measuring the feature usage, we can steer the model towards (not) using features that are specifically (un-)desired." width="35%"/>
</div>

Our method generalizes from debiasing to the **encouragement and discouragement of arbitrary features**. That is, it not only aims at removing the influence of undesired features / biases but also at increasing the influence of features that are known to be well-established from domain knowledge.

If you use our method, please cite:

    @inproceedings{Blunk23:FS,
    author = {Jan Blunk and Niklas Penzel and Paul Bodesheim and Joachim Denzler},
    booktitle = {DAGM German Conference on Pattern Recognition (DAGM-GCPR)},
    title = {Beyond Debiasing: Actively Steering Feature Selection via Loss Regularization},
    year = {2023},
    }


## Installation <a name="installation"></a>

**Install with pip, Python and PyTorch 2.0+**

    git clone https://git.inf-cv.uni-jena.de/blunk/beyond-debiasing.git
    cd beyond-debiasing
    pip install -r requirements.txt

First, create an environment with pip and Python first (Anaconda environment / Python virtual environment). We recommend to install [PyTorch with CUDA support](https://pytorch.org/get-started/locally/). Then, you can install all subsequent packages via pip as described above.


## Usage in Python <a name="usage-in-python"></a>

Since our method relies on loss regularization, it is very simple to add to your own networks - you only need to modify your loss function. To help with that, we provide an [exemplary network](regression_network.py) and a [jupyter notebook](feature_steering_example.ipynb) with example code.

You can find the implementation of the feature steering part of the loss in `feat_steering_loss(...)` of [regression_network.py](regression_network.py), which is where all the magic of our method takes place.

## Repository <a name="repository"></a>

* Installation:
    * [`requirements.txt`](requirements.txt): List of required packages for installation with pip
* Feature attribution:
    * [`contextual_decomposition.py`](contextual_decomposition.py): Wrapper for contextual decomposition
    * [`mixed_cmi_estimator.py`](mixed_cmi_estimator.py): Python port of the CMIh estimator of the conditional 
* Redundant regression dataset:
    * [`algebra.py`](algebra.py): Generation of random orthogonal matrices
    * [`make_regression.py`](make_regression.py): An adapted version of scikit-learns make_regression(...), where the coefficients are standard-uniform
    * [`regression_dataset.py`](regression_dataset.py): Generation of the redundant regression dataset
    * [`dataset_utils.py`](dataset_utils.py): Creation of torch dataset from numpy arrays
    * [`tensor_utils.py`](tensor_utils.py): Some helpful functions for dealing with tensors
* Example:
    * [`feature_steering_example.ipynb`](feature_steering_example.ipynb): Example for generating the dataset, creating and training the network with detailed comments  
    * [`regression_network.py`](regression_network.py): Neural network (PyTorch) used in the example notebook

With [`mixed_cmi_estimator.py`](mixed_cmi_estimator.py) this repository includes a Python implementation of the hybrid CMI estimator CMIh presented by [Zan et al.](https://doi.org/10.3390/e24091234) The authors' original R implementation can be found [here](https://github.com/leizan/CMIh2022).


## License and Support <a name="license-support"></a>
This repository is released under *CC BY 4.0* license, which allows both academic and commercial use. If you need any support, please open an issue or contact [Jan Blunk](https://inf-cv.uni-jena.de/home/group/blunk/).

