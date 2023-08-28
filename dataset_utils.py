import torch

def get_dataset_from_arrays(train_features, train_outputs, test_features, test_outputs, validation_features=None, validation_outputs=None, batch_size=1):
    """
        Both test and train dataset are numpy arrays. Observations are represented
        as rows, features as columns.
        train_targets and test_targets are vectors, containing one value per row
        (expected results).
    """

    train_inputs = torch.tensor(train_features.tolist())
    train_targets = torch.FloatTensor(train_outputs)
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)

    test_inputs = torch.tensor(test_features.tolist())
    test_targets = torch.FloatTensor(test_outputs)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    if not validation_features is None:
        validation_inputs = torch.tensor(validation_features.tolist())
        validation_targets = torch.FloatTensor(validation_outputs)
        validation_dataset = torch.utils.data.TensorDataset(validation_inputs, validation_targets)

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        return (train_dataset, train_loader, test_dataset, test_loader, validation_dataset, validation_loader)
    else:
        return (train_dataset, train_loader, test_dataset, test_loader)
