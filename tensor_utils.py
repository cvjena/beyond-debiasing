import torch

def difference(t1, t2):
    """ Returns all elements that are only in t1 or t2.
    """

    t1, t2 = t1.unique(), t2.unique()
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    diff = uniques[counts == 1]

    return diff

def intersection(t1, t2):
    """ Returns all elements that are in both t1 and t2.
    """

    t1, t2 = t1.unique(), t2.unique()
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersec = uniques[counts > 1]

    return intersec

def setdiff(t1, t2):
    """ Returns all elements of tensor t1 that are not in tensor t2.
    """

    diff = difference(t1, t2)
    diff_from_t1 = intersection(diff, t1)

    return diff_from_t1

def concatenate_1d(*tensors):
    """Concatenates the given 1d tensors.
    """

    for tensor in tensors:
        if len(tensor.size()) != 1:
            raise ValueError("Can only concatenate 1d tensors. Otherwise, use rbind / cbind.")

    return torch.cat(tensors, 0)

def cbind(*tensors):
    """Combines the given 2d tensors as columns.
    """

    # If a vector is one-dimensional, convert it to a two-dimensional column
    # vector.
    tensors = [unsqueeze_to_2d(var) if len(var.size()) == 1 else var for var in tensors]

    return torch.cat(tensors, 1)

def rbind(*tensors):
    """Combines the given 2d tensors as rows.
    """

    for tensor in tensors:
        if len(tensor.size()) < 2:
            raise ValueError("rbind only takes two-dimensional tensors as input")

    return torch.cat(tensors, 0)

def unsqueeze_to_1d(*tensors):
    """ Unsqueezes zero-dimensional tensors to one-dimensional tensors.
    """

    if len(tensors) > 1:
        return [var.unsqueeze(dim=0) if len(var.size()) == 0 else var for var in tensors]
    else:
        return tensors[0].unsqueeze(dim=0) if len(tensors[0].size()) == 0 else tensors[0]

def unsqueeze_to_2d(*tensors):
    """ Unsqueezes one-dimensional tensors two-dimensional tensors.
    """

    if len(tensors) > 1:
        return [var.unsqueeze(dim=1) if len(var.size()) == 1 else var for var in tensors]
    else:
        return tensors[0].unsqueeze(dim=1) if len(tensors[0].size()) == 1 else tensors[0]

def convert_type(torch_type, *tensors):
    """ Converts all given tensors to tensors of the given type.
    """

    if len(tensors) > 1:
        return [var.to(torch_type) for var in tensors]
    else:
        return tensors[0].to(torch_type)

def shuffle(t, mode=None):
    """ Shuffles the rows of the given tensor.
    """

    if mode == "within_columns":
        rand_indices = torch.randperm(len(t))
        t = t[rand_indices]
    elif mode == "within_rows":
        t = t.transpose(dim0=0, dim1=1)
        rand_indices = torch.randperm(len(t))
        t = t[rand_indices]
        t = t.transpose(dim0=0, dim1=1)
    else:
        rand_indices = torch.randperm(t.nelement())
        t = t.view(-1)[rand_indices].view(t.size())

    return t
