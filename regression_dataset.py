import os
from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from algebra import random_orthogonal_matrix
from make_regression import make_regression
from dataset_utils import get_dataset_from_arrays


def make_regression_dataset(
    high_dim_transform: bool = True,
    n_features_low_dim: int = 4,
    n_uninformative_low_dim: int = 0,
    n_high_dim: int = 128,
    noise_on_high_dim_snrdb: float = None,
    noise_on_output: float = 0.0,
    n_train: int = 50000,
    n_test: int = 10000,
    n_validation: int = 10000,
    normalize: bool = False,
    seed: int = None,
    batch_size: int = 10,
    log_coefs: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, NDArray]:
    """Creates a redundant regression dataset based on the specified parameters.

    The dataset generation process follows the one described in our paper. First,
    a low-dimensional linear regression problem is generated. Its input variables
    are standard normal-distributed and the coefficients of the regression task
    follow a standard-uniform distribution. The bias of the regression problem
    is set to zero. Since a linear combination of normal-distributed random variables
    is again normal-distributed, the target variable of the regresssion is also
    normal-distributed. The uninformative variables are distributed similarly
    to the input variables.
    Afterward, the input variables of the regression problem generated in the
    previous step are transformed into a higher-dimensional feature space if
    high_dim_transform is specified. That is, redundancy is introduced.

    Args:
        high_dim_transform (bool, optional): Whether a high-dimensional transformation
            shall be performed. In case no high-dimensional transformation is
            desired, all inputs like n_high_dim, ... are ignored and instead the
            identity transformation is used. noise_on_high_dim_snrdb is still
            used to add noise after the identity transformation. Defaults to True.
        n_features_low_dim (int, optional): Number of informative low-dimensional
            variables. Defaults to 4.
        n_uninformative_low_dim (int, optional): Number of uninformative
            low-dimensional variables. Defaults to 0.
        n_high_dim (int, optional): Number of features after transformation into
            higher-dimensional feature space. Defaults to 128.
        noise_on_high_dim_snrdb (float, optional): Additive gaussian noise is
            applied to the regression input variables after transformation into
            a higher dimension. Here, the variance of each input variable is
            determined and the noise is added so that the SNR corresponds
            to the given value (in dB). A reasonable value would be for instance
            10 or 40. Defaults to None.
        noise_on_output (float, optional): Standard deviation of the gaussian
            noise (zero mean) applied to the output (before normalization).
            Defaults to 0.0.
        n_train (int, optional): Number of samples generated for the training
            dataset. Defaults to 50000.
        n_test (int, optional): Number of samples generated for the test
            dataset. Defaults to 10000.
        n_validation (int, optional): Number of samples generated for the
            validation dataset. Defaults to 10000.
        normalize (bool, optional): Whether the target variable should be
            normalized to mean zero and unit variance. Normalization is based on
            statistics calculated over all samples generated in total. Defaults
            to False.
        seed (int, optional): If specified, the seed to generate a reproducible
            dataset. Defaults to None.
        batch_size (int, optional): Batch size of the created dataset. Defaults
            to 10.
        log_coefs (bool, optional): Whether the coefficients used to generate
            the regression problem should be logged into a separate file. If
            specified, logged to "lowdim_regression_coefficients.list" in the
            current working directory. Defaults to False.

    Raises:
        ValueError: If high_dim_transform is False, the n_high_dim should be None.
        ValueError: noise_on_high_dim_snrdb has to be strictly positive. For no
            noise, specify None.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, NDArray]:  Tuple containing the
            dataloader for training, validation and test dataset and the matrix
            used to perform the dimensionality expansion.
    """

    # Generate random rergession problem.
    n_samples = n_train + n_validation + n_test
    features, output, coefficients = _regression_dataset(
        n_features=n_features_low_dim,
        n_uninformative=n_uninformative_low_dim,
        n_samples=n_samples,
        noise_on_output=noise_on_output,
        seed=seed,
    )
    if log_coefs:
        f = open(os.path.join(os.getcwd(), "lowdim_regression_coefficients.list"), "w")
        f.writelines(coefficients)
        f.close()

    # Normalization.
    # Normalize outputs to mean zero and unit variance.
    if normalize:
        output = output - np.mean(output)
        output = output / np.std(output)

    # Expand to high dimensional problem.
    # If not desired, we apply the identity transformation.
    if high_dim_transform:
        features, transformation_matrix = _inverse_pca_dataset(
            features, n_high_dim, seed=seed
        )
    else:
        if not n_high_dim is None:
            raise ValueError(
                "When no dimensionality expansion is performed, the number of \
                    high dimensional features should not be set."
            )
        transformation_matrix = np.identity(n=n_features_low_dim)

    # Add noise if specified.
    if not noise_on_high_dim_snrdb is None:
        # Calculation of Noise SNR from Signal SNR in dB:
        #           SNR = E(S^2) / E(N^2)
        #       <-> E(N^2) = E(S^2) / SNR
        # Our noise is mean-zero:
        #       <-> Noise Variance = E(S^2) / SNR
        # With SNR = 10^(SNR in dB / 10):
        #       <-> Noise Variance = E(S^2) / (10^(SNR in dB / 10))
        #
        if not noise_on_high_dim_snrdb > 0:
            raise ValueError(
                "SNR has to be strictly positive. Remember, that a SNR of zero \
                    equals infitely large noise. For no noise specify 'None'."
            )
        signal_second_moments = np.mean(features**2, axis=0)
        noise_variances = signal_second_moments / (10 ** (noise_on_high_dim_snrdb / 10))
        noise_stds = np.sqrt(noise_variances)

        np.random.seed(seed=seed)
        features += features + np.random.normal(
            loc=0.0, scale=noise_stds, size=features.shape
        )

    # Divide into test, train, validation.
    (
        _,
        train_dataloader,
        _,
        test_dataloader,
        _,
        validation_dataloader,
    ) = get_dataset_from_arrays(
        train_features=features[:n_train],
        train_outputs=output[:n_train],
        test_features=features[n_train : n_train + n_test],
        test_outputs=output[n_train : n_train + n_test],
        validation_features=features[n_train + n_test :],
        validation_outputs=output[n_train + n_test :],
        batch_size=batch_size,
    )

    return (
        train_dataloader,
        test_dataloader,
        validation_dataloader,
        transformation_matrix,
    )


def _regression_dataset(
    n_features: int,
    n_samples: int,
    n_uninformative: int = 0,
    noise_on_output: float = 0.0,
    seed: int = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generates a random regression dataset with n_features parameters.

    n_features describes the total number of features. If n_uninformative > 0,
    not all of these features are relevant for the generated regression problem.
    The regression coefficients are drawn standard-uniformly.

    Args:
        n_features (int): Number of coefficients of the regression problem /
            dimensions of the input.
        n_samples (int): Number of samples generated for the regression problem.
        n_uninformative (int, optional): Number of noise variables (uninformative
            variables). The uninformative variables are distributed similarly to
            the coefficients of the regression problem. Defaults to 0.
        noise_on_output (float, optional): std of gaussian noise applied to the
            output. Defaults to 0.0.
        seed (int, optional): If specified, the seed to generate a reproducible
            regression dataset. Defaults to None.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Coefficients of the regression problem
            and generated samples. Tuple of (inputs of the generated samples,
            outputs of the generated samples, coefficients of the regression problem
            used to generate the samples).
    """

    # n_samples:        Number of observations to estimate coefficients
    # n_features:       Number of total features, potentially not all relevant for
    #                   regression (number of dimensions of the input vector)
    # n_informative:    Number of informative features (=coefficients), that is, the
    #                   features that are used to build the linear model
    # n_targets:        Number of dimensions of the output vector
    # noise:            std of gaussian noise applied to output
    features, output, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features + n_uninformative,
        n_informative=n_features,
        n_targets=1,
        noise=noise_on_output,
        coef=True,
        random_state=seed,
    )

    return (features, output, coef)


def _inverse_pca_dataset(
    features: NDArray, n_high_dim_features: int, seed: int = None
) -> Tuple[NDArray, NDArray]:
    """Input data is interpreted as output of a PCA. We perform an "inverse PCA"
    with a random transformation matrix.

    Args:
        features (NDArray): Dataset to transform to higher dimension. Rows are
            interpreted as observations and columns as input features.
        n_high_dim_features (int): Number of output features per observation.
        seed (int, optional): If specified, the seed to generate a reproducible
            dataset. Defaults to None.

    Returns:
        Tuple[NDArray, NDArray]: Tuple of (transformed dataset, transformation matrix).
    """

    # Generate transformation to higher dimension.
    # For theoretical reasons the change of basis matrix has to have a left inverse,
    # but when doing PCA this is guaranteed even if we only care about the first
    # k eigen vectors - in this case, the first k columns.
    n_low_dim_features = features.shape[1]
    change_of_basis_matrix = random_orthogonal_matrix(n_high_dim_features, seed=seed)
    transformation_matrix = change_of_basis_matrix[:, :n_low_dim_features]

    # Apply transformation to higher dimension to the observational data.
    high_dim_features = _matrix_transform_dataset(features, transformation_matrix)

    # Standardize the high-dimensional data.
    # high_dim_features = StandardScaler().fit_transform(high_dim_features)

    return (high_dim_features, transformation_matrix)


def _matrix_transform_dataset(
    features: NDArray, transformation_matrix: NDArray
) -> NDArray:
    """Applies a given transformation matrix A to a dataset.

    The transformation matrix A is applied to the rows x of the dataset via Ax=x'.

    Args:
        features (NDArray): Dataset to apply transformation to. Rows are
            interpreted as observations and columns as input features.
        transformation_matrix (NDArray): Matrix used for transformation.

    Returns:
        NDArray: Transformed input dataset.
    """

    # features: rows are observations, columns the features.
    # (transformed features to higher dimension, matrix used for transformation)
    n_dim_features = features.shape[1]

    assert transformation_matrix.shape[1] == n_dim_features

    # Apply transformation to higher dimension to the observational data.
    # Instead of individually applying the change of basis matrix via B*x=x' we
    # can apply it to all observations at once via X*B^T=X'.
    transformed_features = features @ transformation_matrix.T

    return transformed_features
