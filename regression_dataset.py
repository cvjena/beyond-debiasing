import os
import numpy as np
from toy_data.algebra import random_orthogonal_matrix
from toy_data.sklearn_adaptions import make_regression
from dataset_utils import get_dataset_from_arrays

def make_regression_dataset(high_dim_transform=True, n_features_low_dim=4, n_uninformative_low_dim=0, n_high_dim = 128, noise_on_high_dim_snrdb=None,
                               noise_on_output=0.0, n_train=50000, n_test=10000, n_validation=10000, normalize=False, seed = None, batch_size = 10, log_coefs = False):
    """
        The input variables are standard normal distributed and the coefficients
        of the regression task follow a standard uniform distribution. The bias
        of the regression problem is set to zero.
        Since a linear combination of normal distributed random variables is
        again normal distributed, the target variable of the regresssion is also
        normal distributed.
        The random variables are distributed similarly to the input variables.

        high_dim_transform:
            Whether a high-dimensional transformation shall be performed. In case
            no high-dimensional transformation is desired, all inputs like
            n_high_dim, ... are ignored and instead the identity transformation
            is used.
            noise_on_high_dim_snrdb is still used to add noise after the identity
            transformation.
        n_low_dim:
            Number of informative low-dimensional variables.
        n_informative_low_dim:
            Number of uninformative low-dimensional variables.
        noise_on_high_dim_snrdb:
            Additive gaussian noise is applied to the regression input variables after transformation into
            a higher dimension. Here, the variance of each input variable is determined and the noise is
            added so that the SNR corresponds to the given value (in dB). A reasonable value can for instance
            be 10 or 40.
        noise_on_output: 
            Standard deviation of the gaussian noise (zero mean) applied to the output (before normalization).
    """

    # Generate random rergession problem.
    n_samples = n_train + n_validation + n_test
    features, output, coefficients = _regression_dataset(n_features=n_features_low_dim,
                                            n_uninformative=n_uninformative_low_dim,
                                            n_samples=n_samples,
                                            noise_on_output=noise_on_output,
                                            seed=seed)
    if log_coefs:
        f = open(os.path.join(os.getcwd(), "lowdim_regression_coefficients.list"),'w')
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
        features, transformation_matrix = _inverse_pca_dataset(features, n_high_dim, seed=seed)
    else:
        if not n_high_dim is None:
            raise ValueError("When no dimensionality expansion is performed, the number of high dimensional features should not be set.")
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
        if noise_on_high_dim_snrdb == 0.0:
            raise ValueError("A SNR of zero equals infite noise. For no noise specify \'None\'.")
        signal_second_moments = np.mean(features ** 2, axis=0)
        noise_variances = signal_second_moments / (10 ** (noise_on_high_dim_snrdb / 10))
        noise_stds = np.sqrt(noise_variances)

        np.random.seed(seed=seed)
        features += features + np.random.normal(loc=0.0, scale=noise_stds, size=features.shape)

    # Divide into test, train, validation.
    _, train_dataloader, _, test_dataloader, _, validation_dataloader = get_dataset_from_arrays(train_features=features[:n_train],
                                                                                                train_outputs=output[:n_train],
                                                                                                test_features=features[n_train:n_train + n_test],
                                                                                                test_outputs=output[n_train:n_train + n_test],
                                                                                                validation_features=features[n_train + n_test:],
                                                                                                validation_outputs=output[n_train + n_test:],
                                                                                                batch_size=batch_size)

    return (train_dataloader, test_dataloader, validation_dataloader, transformation_matrix)

def _regression_dataset(n_features, n_samples, n_uninformative=0, noise_on_output=0.0, seed=None):
    """Generates a random regression dataset with n_features parameters.

    :param n_features: Number of coefficients of the regression problem /
    dimensions of the input.
    :param n_uninformative: Number of noise variables (uninformative variables).
    The uninformative variables are distributed similarly to the coefficients of
    the regression problem.
    :param n_samples: Number of samples generated for the regression problem.
    :param seed: If int the seed to generate a reproducible regression dataset.
    :return: Coefficients of the regression problem and generated samples.
    :rtype: Tuple of (inputs of the generated samples, outputs of the generated
    samples, coefficients of the regression problem used to generate the samples).

    """

    # n_samples:        Number of observations to estimate coefficients
    # n_features:       Number of total features, potentially not all relevant for
    #                   regression (number of dimensions of the input vector)
    # n_informative:    Number of informative features (=coefficients), that is, the
    #                   features that are used to build the linear model
    # n_targets:        Number of dimensions of the output vector
    # noise:            std of gaussian noise applied to output
    features, output, coef = make_regression(  n_samples = n_samples,
                                                        n_features = n_features + n_uninformative,
                                                        n_informative = n_features,
                                                        n_targets = 1,
                                                        noise = noise_on_output,
                                                        coef = True,
                                                        random_state = seed)

    return (features, output, coef)

def _inverse_pca_dataset(features, n_high_dim_features, seed=None):
    """Input data is interpreted as output of a random PCA, transforms the data
    with an inverse transformation of a random PCA.

    :param features: Dataset to transform to higher dimension. Rows are
    interpreted as observations and columns as input features.
    :param n_high_dim_features: Number of output features per observation.
    :param seed: If int the seed to generate a reproducible dataset.
    :return: Transformed dataset and matrix used for transformation.
    :rtype: Tuple of (transformed dataset, transformation matrix).

    """
    # Generate transformation to higher dimension.
    # For theoretical reasons the change of basis matrix has to have a left inverse,
    # but when doing PCA this is guaranteed even if we only care about the first
    # k eigen vectors - in this case, the first k columns.
    n_low_dim_features = features.shape[1]
    change_of_basis_matrix = random_orthogonal_matrix(n_high_dim_features, seed=seed)
    transformation_matrix = change_of_basis_matrix[:,:n_low_dim_features]

    # Apply transformation to higher dimension to the observational data.
    high_dim_features = _matrix_transform_dataset(features, transformation_matrix)

    # Standardize the high-dimensional data.
    # high_dim_features = StandardScaler().fit_transform(high_dim_features)

    return (high_dim_features, transformation_matrix)

def _matrix_transform_dataset(features, transformation_matrix):
    """Applies a given transformation matrix A to a dataset.
    The transformation matrix is applied to the rows x of the dataset via Ax=x'.

    :param type features: Dataset to apply transformation to. Rows are
    interpreted as observations and columns as input features.
    :param type transformation_matrix: Matrix used for transformation.
    :return: Transformed input dataset.
    :rtype: np.array

    """

    # features: rows are observations, columns the features.
    # (transformed features to higher dimension, matrix used for transformation)
    n_dim_features = features.shape[1]

    assert(transformation_matrix.shape[1] == n_dim_features)

    # Apply transformation to higher dimension to the observational data.
    # Instead of individually applying the change of basis matrix via B*x=x' we
    # can apply it to all observations at once via X*B^T=X'.
    transformed_features = features @ transformation_matrix.T

    return transformed_features
