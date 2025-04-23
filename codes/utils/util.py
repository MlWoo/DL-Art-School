import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def default(val, d):
    return val if val is not None else d


def get_network_description(network):
    """Get the string and total parameters of the network"""
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    return str(network), sum(map(lambda x: x.numel(), network.parameters()))


def print_network(net, name="some network"):
    s, n = get_network_description(net)
    net_struc_str = "{}".format(net.__class__.__name__)
    print("Network {} structure: {}, with parameters: {:,d}".format(name, net_struc_str, n))
    print(s)
