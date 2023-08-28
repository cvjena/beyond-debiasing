"""
This file provides a way to use contextual decomposition in the loss function.

The implementation requires and is partly based on the implementation provided
by Singh et al. for their paper "Hierarchical interpretations for neural network
predictions":

    @inproceedings{
       singh2019hierarchical,
       title={Hierarchical interpretations for neural network predictions},
       author={Chandan Singh and W. James Murdoch and Bin Yu},
       booktitle={International Conference on Learning Representations},
       year={2019},
       url={https://openreview.net/forum?id=SkEqro0ctQ},
    }

The original implementation can be found here:
    https://github.com/csinva/hierarchical-dnn-interpretations

It was published under the following license:

    MIT License

    Copyright (c) 2019 Chandan Singh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import acd
import numpy as np
import torch


def get_cd_1d_by_modules(model, modules, inputs, feat_of_interest, device="cpu"):
    # Device.
    inputs = inputs.to(device)

    # Prepare mask.
    # The mask answers the following question: For which dimensions do you want
    # to get their feature attribution compared to all other features?
    n_dim = list(inputs.size())[1]
    mask = np.zeros(n_dim, dtype=np.int32)
    mask[feat_of_interest] = 1

    # Set up relevant/irrelevant based on mask.
    # Starting here, we start to follow the CD implementation from
    # https://github.com/csinva/hierarchical-dnn-interpretations/blob/master/acd/scores/cd.py.
    im_torch = inputs
    mask = torch.FloatTensor(mask).to(device)
    relevant = mask * im_torch
    irrelevant = (1 - mask) * im_torch
    relevant = relevant.to(device)
    irrelevant = irrelevant.to(device)

    relevant, irrelevant = acd.cd_generic(modules, relevant, irrelevant)

    return relevant, irrelevant


def get_cd_1d(model, inputs, feat_of_interest, device="cpu"):
    """Calculates contextual decomposition scores for the given model.

    The contextual decomposition performs feature attribution by decomposing
    the output of the model into two parts: The contribution of the feature(s)
    of interest and the contribution of all other features.
    Therefore, you have to specify which features are of interest. In a 1d
    scenario you are typically interested in the influence of a single
    feature compared to all other features, but this method also allows you
    to specify a list of features that, together, form the features of
    interest.

    Interpretation of the generated scores:
    The output is (scores_feat, scores_other) with both being a one-dimensional
    tensor. Since this method works with batched data, that means that for
    each input sample two floating point scores are generated: the contribution
    of the feature(s) of interest and the contribution of all other features.
        Prediction of the Network = score of the features of interest
                                    + score of the other features

    :param model: PyTorch-Model to generate the CD scores for.
    :param inputs: Batched inputs to the model. Typically 2-dimensional tensor
    containing the inputs for a single batch.
    :param feat_of_interest: Integer or list of integers. Define which
    dimensions of the input are part of the feature(s) of interest.
    :param device: Device used by PyTorch (cuda / cpu).
    :return: Tuple (scores_feat, scores_other). These are the scores for each
    of the batched inputs. Here, scores_feat[i] + scores_other[i]=prediction[i].
    Note that the feature scores are determined in a per-batch manner. Therefore,
    the resulting feature scores are vectors.
    :rtype: Tupel of one-dimensional tensors.

    """

    # Set model in evaluation mode.
    prev_training_status = model.training
    model.eval()

    # Prepare mask.
    # The mask answers the following question: For which dimensions do you want
    # to get their feature attribution compared to all other features?
    n_dim = list(inputs.size())[1]
    mask = np.zeros(n_dim, dtype=np.int32)
    mask[feat_of_interest] = 1

    # Contextual decomposition.
    # We receive the contribution of the feature(s) of interest compared to all
    # other features.
    # The output is a tensor with a length >= 1, because we are considering batches.
    # That is, for each element of the batch we get the contribution of the
    # feature(s) of interest.
    scores_feat, scores_other = acd.cd(inputs, model=model, mask=mask, device=device)

    # Reset evaluation mode if necessary.
    if prev_training_status:
        model.train()

    return (torch.flatten(scores_feat), torch.flatten(scores_other))
