from typing import Literal, Optional

import torch


# EXPLICITLY leave these empty; ensure that an attributeerror is raised if these are not initialised properly.
class nn:
    pass


class optim:
    pass


def populate(adam=True, adamw=True, linear=False, embedding: Optional[Literal["STABLE", "NORMAL"]] = "NORMAL"):
    nn.Linear = torch.nn.Linear
    nn.Embedding = torch.nn.Embedding
    optim.Adam = torch.optim.Adam  # this does nothing tbh
    optim.AdamW = torch.optim.AdamW
    #
    try:
        import bitsandbytes as bnb
    except ImportError:
        return print("WARNING: bnb was missing, not using 8bit for anything!")
    #
    if adam:
        optim.Adam = bnb.optim.Adam8bit
    if adamw:
        optim.AdamW = bnb.optim.AdamW8bit
    if linear:
        nn.Linear = bnb.nn.Linear8bitLt
    if embedding:
        nn.Embedding = bnb.nn.StableEmbedding if embedding == "STABLE" else bnb.nn.modules.Embedding
