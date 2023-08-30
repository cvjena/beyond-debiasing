import torch
from torch.utils.data import TensorDataset, DataLoader
from numpy.typing import NDArray
from typing import Tuple, Union


def get_dataset_from_arrays(
    train_features: NDArray,
    train_outputs: NDArray,
    test_features: NDArray,
    test_outputs: NDArray,
    validation_features: NDArray = None,
    validation_outputs: NDArray = None,
    batch_size: int = 1,
) -> Union[
    Tuple[
        TensorDataset, DataLoader, TensorDataset, DataLoader, TensorDataset, DataLoader
    ],
    Tuple[TensorDataset, DataLoader, TensorDataset, DataLoader],
]:
    """Create a dataset and dataloder from each of the datasets given as numpy arrays.

    Creates dataset and dataloader for train, test and if given also validation
    dataset. Observations are represented as rows, while features are represented
    as columns. The output vectors specify the targets / desired outputs. They are
    vectors containing one value per row (observation).

    Args:
        train_features (NDArray): Features of training dataset.
        train_outputs (NDArray): Targets of training dataset.
        test_features (NDArray): Features of test dataset.
        test_outputs (NDArray): Targets of test dataset.
        validation_features (NDArray, optional): Features of validation dataset. Defaults to None.
        validation_outputs (NDArray, optional): Targets of validation dataset. Defaults to None.
        batch_size (int, optional): Batch size of the created dataset. Defaults to 1.

    Returns:
        Union[Tuple[TensorDataset, DataLoader, TensorDataset, DataLoader, TensorDataset, DataLoader], Tuple[TensorDataset, DataLoader, TensorDataset, DataLoader]]: Tuple of dataset and dataloader for training, validation and if given also validation dataset.
    """

    train_inputs = torch.tensor(train_features.tolist())
    train_targets = torch.FloatTensor(train_outputs)
    train_dataset = TensorDataset(train_inputs, train_targets)

    test_inputs = torch.tensor(test_features.tolist())
    test_targets = torch.FloatTensor(test_outputs)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    if not validation_features is None:
        validation_inputs = torch.tensor(validation_features.tolist())
        validation_targets = torch.FloatTensor(validation_outputs)
        validation_dataset = TensorDataset(validation_inputs, validation_targets)

        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        return (
            train_dataset,
            train_loader,
            test_dataset,
            test_loader,
            validation_dataset,
            validation_loader,
        )
    else:
        return (train_dataset, train_loader, test_dataset, test_loader)
