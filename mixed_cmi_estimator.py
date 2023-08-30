"""
This is document contains a python port of the hybrid CMI estimator CMIh proposed
by Zan et al.:

    Zan, Lei; Meynaoui, Anouar; Assaad, Charles K.; Devijver, Emilie; Gaussier,
    Eric (2022): A Conditional Mutual Information Estimator for Mixed Data and
    an Associated Conditional Independence Test. In: Entropy (Basel, Switzerland)
    24 (9). DOI: 10.3390/e24091234.

The original R implementation can be found here:
    https://github.com/leizan/CMIh2022

It was published under the following license:

    MIT License

    Copyright (c) 2022 leizan

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

import torch
import math
from torch.special import digamma
from torchmetrics.functional import pairwise_manhattan_distance

from tensor_utils import (
    intersection,
    setdiff,
    unsqueeze_to_1d,
    unsqueeze_to_2d,
    concatenate_1d,
    cbind,
)


def get_dist_array(data: torch.Tensor) -> torch.Tensor:
    """Returns pairwise distances for all columns of the input matrix measured
    with the manhattan distance.

    Args:
        data (torch.Tensor): Input matrix. Variables are represented by COLUMNS,
        observations by ROWS.

    Returns:
        torch.Tensor: Pairwise (manhattan) distances.
    """
    # Works for 1D and 2d data.

    # Rotate if only a list so that it becomes a two-dimensional vector.
    if data.size()[0] == 1:
        data = data.transpose(dim0=0, dim1=1)
    N = data.size()[0]
    nDim = data.size()[1]
    disArray = torch.zeros(size=(N, N, nDim))

    # Attention: Our m is one smaller than the m of the original implementation
    for m in range(nDim):  # 1:nDim
        # Get m-th column of data.
        dataDim = unsqueeze_to_2d(data[:, m])

        # Calculate pairwise manhattan distance of columns.
        disArray[:, :, m] = pairwise_manhattan_distance(dataDim)
    return disArray


def get_epsilon_distance(k, disArray):
    """Based on a tensor of pairwise distances per observation (tensor is
    three-dimensional, quadratic, symmetric).
    """

    if disArray.size()[0] == 1:
        disArray = disArray.transpose(dim0=0, dim1=1)
    N = disArray.size()[0]
    epsilonDisArray = torch.zeros(size=[N])

    for point_i in range(N):
        coord_dists = disArray[:, point_i, :]
        # Compute maximum element per row.
        # torch.max returns (values, indices).
        dists, _ = torch.max(coord_dists, dim=1)
        # torch.sort returns (sorted, indices).
        ordered_dists, _ = torch.sort(dists)
        # The original R implementation uses k+1 to access ordered_dists. However,
        # python's indexing starts with 0 instead of 1. Therefore, the index k+1 in R
        # corresponds to index k in python.
        epsilonDisArray[point_i] = 2 * ordered_dists[k]

    return epsilonDisArray


def find_inter_cluster(eleInEachClass):
    """Takes a list of scalars / lists of scalars and returns the intersection."""

    interCluster = eleInEachClass[0]
    for m in range(1, len(eleInEachClass)):
        interCluster = intersection(interCluster, eleInEachClass[m])

    # Output if zero if there is no intersection (= empty list).
    if interCluster.nelement() == 0:
        interCluster = torch.tensor(0.0)

    return interCluster


def con_entro_estimator(data, k, dN):
    """Estimates entropy of quantitative data."""

    # If only one row, count the number of columns (= length of the vector).
    if data.size()[0] == 1:
        N = data.size()[1]
    else:
        N = data.size()[0]
    if N == 1:
        return 0

    # Get distances.
    distArray = get_dist_array(data)
    epsilonDis = get_epsilon_distance(k, distArray)

    if 0 in epsilonDis:
        epsilonDis = epsilonDis[epsilonDis != 0]
        N = epsilonDis.nelement()
        if N == 0:
            return 0
        entropy = (
            -digamma(torch.tensor(k))
            + digamma(torch.tensor(N))
            + (dN * torch.sum(torch.log(epsilonDis))) / N
        )
        return entropy

    # The maximum norm is used, so Cd=1, log(Cd)=0
    entropy = (
        -digamma(torch.tensor(k))
        + digamma(torch.tensor(N))
        + (dN * torch.sum(torch.log(epsilonDis))) / N
    )
    return entropy


def mixed_entro_estimator(
    data: torch.Tensor, dimCon: torch.Tensor, dimDis: torch.Tensor, k: int = 0.1
) -> torch.Tensor:
    """Estimates the entropy of the mixed variables with the given dimensions.

    The data is a matrix where COLUMNS represent variables and ROWS represent
    observations. dimCon and dimDis contain the indices of which variables are
    quantitative and which are qualitative.
    Variables are represented by COLUMNS, observations by ROWS.

    Args:
        data (torch.Tensor): Data with variables represented as columns and
            observations represented as rows.
        dimCon (torch.Tensor): Indices of data for the columns corresponding to
            quantitative variables.
        dimDis (torch.Tensor): Indices of data for the columns corresponding to
            qualitative variables.
        k (int, optional): Neighborhood size is calculated as
            max(1, round(k*#all_neighbors)). k should be < 1 and defaults to 0.1.

    Returns:
        torch.Tensor: Entropy estimate.
    """

    # Get number of quantitative variables and number of observations.
    dN = len(dimCon)
    N = data.size()[0]

    # Split data into quantitative and qualitative data.
    entroCon = 0
    entroDis = 0
    if len(dimCon) != 0:
        dataCon = torch.index_select(data, 1, dimCon)
    if len(dimDis) != 0:
        dataDis = torch.index_select(data, 1, dimDis)
    if len(dimCon) == 0 and len(dimDis) == 0:
        # Input data is null.
        pass

    # Calculate the entropie for the extracted data.
    # If the data is purely continuous!
    if len(dimDis) == 0 and len(dimCon) != 0:
        entroCon = con_entro_estimator(dataCon, max(1, round(k * N)), dN)

    if len(dimDis) != 0:
        # Histogram creation.
        # We create a histogram that shows how often each combination of all
        # possible combinations of the input variables X, Y, Z occurs. That is,
        # we gather the unique values of X, Y and Z separately and then calculate
        # all possible combinations of these values.
        # Afterwards, we count the occurence of each of these combinations
        # and divide all bins of the histogram by the total number of bins
        # (= combinations of values from X, Y and Z).

        # Unique elements -> Histogram bins.
        classByDimList = [
            torch.unique(column) for column in dataDis.transpose(dim0=0, dim1=1)
        ]

        # Create all possible combinations of the unique values of the columns
        # of dataDis (we have already removed duplicate rows).
        # Make sure that classList is a two-dimensional tensor, even for dataDis
        # having just one column.
        classList = torch.cartesian_prod(*classByDimList)
        classList = unsqueeze_to_2d(classList)

        # Create histogram entries. That is, we count the occurence of each
        # of the histogram bins / possible row combinations from dataDis.
        indexInClass = [0] * classList.size()[0]
        for i in range(len(indexInClass)):
            classElement = classList[i, :]

            if classElement.size()[0] == 1:
                # Get indices where classElement is a row in dataDis.
                # See: https://stackoverflow.com/questions/59705001/torch-find-indices-of-matching-rows-in-2-2d-tensors.
                classElement = unsqueeze_to_2d(classElement)
                indexInClass[i] = torch.where((classElement == dataDis).all(dim=1))[0]
            else:
                eleInEachClass = [0] * classElement.size()[0]
                for m in range(len(eleInEachClass)):
                    eleInEachClass[m] = torch.where(
                        (
                            unsqueeze_to_2d(classElement[m])
                            == unsqueeze_to_2d(dataDis[:, m])
                        ).all(dim=1)
                    )[0]
                indexInClass[i] = find_inter_cluster(eleInEachClass)

        # Remove the empty bins and reverse order (for some reason the authors
        # of the original paper reverse the order here).
        indexInClass = [bin for bin in indexInClass if not len(bin.size()) == 0]
        indexInClass.reverse()

        # Entropy calculation.
        # We calculate the entropy by treating each entry of the histogram
        # as probabilities and use the following formular:
        #       H=-\sum(p*log p)
        # To treat the entries of the histogram as probabilities, we normalize
        # with the total number of observations.
        probBins = [hist_bin.nelement() / N for hist_bin in indexInClass]
        for i in probBins:
            entroDis = entroDis - i * math.log(i)
        entroDis = torch.tensor(entroDis)

    if len(dimDis) != 0 and len(dimCon) != 0:
        # Unlike in the case of no qualitative variables, in the case of both
        # qualitative and quantitative variables the calculation of the
        # quantitative entropy depends on the previous calculation of the
        # qualitative entropy.
        # Since we want the result to be differentiable w.r.t. the quantitative
        # variables, we have to make sure that this calculation is differentiable.
        # Everything that interacts with dataCon has to be differentiable.
        for i in range(len(probBins)):
            data = unsqueeze_to_2d(dataCon)[indexInClass[i], :]
            # Neighborhood size is not named k, because that parameter already exists.
            neighborhood_size = max(1, round(k * indexInClass[i].nelement()))
            entroCon = (
                entroCon
                + con_entro_estimator(data=data, k=neighborhood_size, dN=dN)
                * probBins[i]
            )

    # Put quantitative and qualitative entropy together.
    res = entroCon + entroDis

    return res


def _mixed_cmi_model(
    data: torch.Tensor,
    xind: torch.Tensor,
    yind: torch.Tensor,
    zind: torch.Tensor,
    is_categorical: torch.Tensor,
    k: int = 0.1,
) -> torch.Tensor:
    """Estimates the CMI from qualitative and quantitative data.

    The conditional mutual independence I(X;Y|Z) is -> 0 if X and Y are
    dissimilar and -> inf if they are similar.

    Method:
    Zan, Lei; Meynaoui, Anouar; Assaad, Charles K.; Devijver, Emilie; Gaussier,
    Eric (2022): A Conditional Mutual Information Estimator for Mixed Data and
    an Associated Conditional Independence Test. In: Entropy (Basel, Switzerland)
    24 (9). DOI: 10.3390/e24091234.
    The implementation follows their R implementation licensed unter MIT license:
        https://github.com/leizan/CMIh2022/blob/main/method.R

    Args:
        data (torch.Tensor): Observations of the variables x, y, and z. Variables are
            represented by COLUMNS, observations by ROWS.
        xind (torch.Tensor): One-dimensional tensor that contains a list of the indices
            corresponding to the columns of data containing the observations of the
            variable x.
        yind (torch.Tensor): One-dimensional tensor that contains a list of the indices
            corresponding to the columns of data containing the observations of the
            variable y.
        zind (torch.Tensor): One-dimensional tensor that contains a list of the indices
            corresponding to the columns of data containing the observations of the
            variable z.
        is_categorical (torch.Tensor): One-dimensional tensor that contains a list of
            the indices corresponding to the columns of data that contain qualitative
            (=categorical) data. All other columns are expected to contain quantitative
            data.
        k (int, optional): Neighborhood size for kNN. Calculated as
            Neighborhood size = max(1, round(k * #All neighbors)). k should be < 1
            and defaults to 0.1.

    Returns:
        torch.Tensor: Estimate of the CMI I(X;Y|Z).
    """

    # data: Variables are represented by COLUMNS, obervations by ROWS.
    #
    # xind: Tensor with indices of the columns that contain the observations of the variable x.
    # yind: Tensor with indices of the variable x.
    # zind: Tensor with indices of the variable x.
    #
    # isCat: List that contains the indices of all columns of data that contain
    # qualitative (=categorical) data. All other columns contain quantitative values.

    # All input variables describing indices should be 1D-Tensors of type int.
    xind, yind, zind, is_categorical = [
        torch.tensor(var) if isinstance(var, int) or isinstance(var, list) else var
        for var in (xind, yind, zind, is_categorical)
    ]
    xind, yind, zind, is_categorical = [
        var.unsqueeze(dim=0) if len(var.size()) == 0 else var
        for var in (xind, yind, zind, is_categorical)
    ]
    xind, yind, zind, is_categorical = [
        var.to(torch.int) for var in (xind, yind, zind, is_categorical)
    ]

    # Move tensors to correct device.
    xind, yind, zind, is_categorical = [
        var.to(data.get_device()) for var in (xind, yind, zind, is_categorical)
    ]

    # setdiff: setdiff(A, B) returns all elements of A that are not in B
    # (without repetitions).
    # ...Con = Indices of data for the quantitative columns of X
    # ...Dis = Indices of data for the qualitative columns of X
    xDimCon = setdiff(xind, is_categorical)
    xDimDis = setdiff(xind, xDimCon)
    yDimCon = setdiff(yind, is_categorical)
    yDimDis = setdiff(yind, yDimCon)
    zDimCon = setdiff(zind, is_categorical)
    zDimDis = setdiff(zind, zDimCon)
    xDimCon, xDimDis, yDimCon, yDimDis, zDimCon, zDimDis = unsqueeze_to_1d(
        xDimCon, xDimDis, yDimCon, yDimDis, zDimCon, zDimDis
    )

    conXYZ = concatenate_1d(xDimCon, yDimCon, zDimCon)
    disXYZ = concatenate_1d(xDimDis, yDimDis, zDimDis)
    hXYZ = mixed_entro_estimator(data, conXYZ, disXYZ, k=k)

    conXZ = concatenate_1d(xDimCon, zDimCon)
    disXZ = concatenate_1d(xDimDis, zDimDis)
    hXZ = mixed_entro_estimator(data, conXZ, disXZ, k=k)

    conYZ = concatenate_1d(yDimCon, zDimCon)
    disYZ = concatenate_1d(yDimDis, zDimDis)
    hYZ = mixed_entro_estimator(data, conYZ, disYZ, k=k)

    conZ = zDimCon
    disZ = zDimDis
    hZ = mixed_entro_estimator(data, conZ, disZ, k=k)

    cmi = hXZ + hYZ - hXYZ - hZ

    return cmi


def mixed_cmi_model(
    feature: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
    feature_is_categorical: bool,
    target_is_categorical: bool,
) -> torch.Tensor:
    """Estimates the CMI for a set of a feature of interest, model output and
    desired output.

    Estimates the conditional mutual information CMI(feature, output | target).
    Here, both the feature and the target are allowed to be both qualitative and
    quantitative variables. The model output has to be quantitative. All input
    tensors are only allowed to be one-dimensional tensors and describe batched
    observations.

    The conditional mutual information I(X;Y|Z) is -> 0 if X and Y are
    dissimilar and -> inf if they are similar. Please note, that the returned CMI
    is only differentiable with respect to non-categorical variables, i.e., if
    you specify one of the three input variables as categorical you cannot
    differentiate with respect to it afterward. That is, because in this case a
    histogram is created which cannot be done in a differentiable manner.

    Method:
    Zan, Lei; Meynaoui, Anouar; Assaad, Charles K.; Devijver, Emilie; Gaussier,
    Eric (2022): A Conditional Mutual Information Estimator for Mixed Data and
    an Associated Conditional Independence Test. In: Entropy (Basel, Switzerland)
    24 (9). DOI: 10.3390/e24091234.
    The implementation follows their R implementation and was adapted for
    differentiability w.r.t. quantitative variables:
        https://github.com/leizan/CMIh2022/blob/main/method.R

    Args:
        feature (torch.Tensor): One-dimensional feature vector.
        output (torch.Tensor): One-dimensional model output vector.
        target (torch.Tensor): One-dimensional target vector.
        feature_is_categorical (bool): Whether the feature is a categorical variable.
        target_is_categorical (bool): Whether the target is a categorical variable.

    Raises:
        ValueError: Raised if one of feature, output and target is not a
            one-dimensional tensor.

    Returns:
        torch.Tensor: Estimate of CMI(feature, output | target).
    """

    if any([z.dim() != 1 for z in (feature, output, target)]):
        raise ValueError("All input tensors have to be one-dimensional!")

    # Merge into one 2D array.
    # Variables are represented by columns.
    data = cbind(feature, output, target)

    # Prepare indices and which variables are categorical.
    is_categorical = []
    if feature_is_categorical:
        is_categorical.append(0)
    if target_is_categorical:
        is_categorical.append(2)
    xind, yind, zind = [0], [1], [2]

    return _mixed_cmi_model(data, xind, yind, zind, is_categorical)
