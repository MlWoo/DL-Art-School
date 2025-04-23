def safe_unsqueeze(t, dim):
    assert t.size(dim) == 1
    return t.unsqueeze(dim)


def safe_squeeze(t, dim):
    assert t.size(dim) == 1
    return t.squeeze(dim)
