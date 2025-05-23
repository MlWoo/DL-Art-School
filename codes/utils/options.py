import logging
import os.path as osp
from collections import OrderedDict

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

loaded_options = None


def OrderedYaml():
    """yaml orderedDict support"""
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


Loader, Dumper = OrderedYaml()  # noqa: F811


class Loader(yaml.SafeLoader):
    pass


# define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


# define custom tag handler
def add(loader, node):
    seq = loader.construct_sequence(node)
    return sum([i for i in seq])


# register the tag handler
yaml.add_constructor("!join", join, Loader)
yaml.add_constructor("!add", add, Loader)


def parse(opt_path, is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)

    opt["is_train"] = is_train

    # datasets
    if "datasets" in opt.keys():
        for phase, dataset in opt["datasets"].items():
            phase = phase.split("_")[0]
            if dataset["phase"] is None:
                dataset["phase"] = phase
            is_lmdb = False
            """ LMDB is not supported at this point with the mods I've been making.
            if dataset.get('dataroot_GT', None) is not None:
                dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
                if dataset['dataroot_GT'].endswith('lmdb'):
                    is_lmdb = True
            if dataset.get('dataroot_LQ', None) is not None:
                dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
                if dataset['dataroot_LQ'].endswith('lmdb'):
                    is_lmdb = True
            """
            dataset["data_type"] = "lmdb" if is_lmdb else "img"
            if dataset["mode"].endswith("mc"):  # for memcached
                dataset["data_type"] = "mc"
                dataset["mode"] = dataset["mode"].replace("_mc", "")

    # path
    if "path" in opt.keys():
        for key, path in opt["path"].items():
            if path and key in opt["path"] and (key not in ["strict_load", "optimizer_reset", "scheduler_reset"]):
                opt["path"][key] = osp.expanduser(path)
    else:
        opt["path"] = {}
    opt["path"]["root"] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt["path"]["root"], "experiments", opt["name"])
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")

        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(opt["path"]["root"], "results", opt["name"])
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root

    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def opt_get(opt, keys, default=None):
    assert not isinstance(keys, str)  # Common mistake, better to assert.
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths"""
    logger = logging.getLogger("base")
    if opt["path"]["resume_state"]:
        if (
            opt["path"].get("pretrain_model_G", None) is not None
            or opt["path"].get("pretrain_model_D", None) is not None
        ):
            logger.warning("pretrain_model path will be ignored when resuming training.")

        # Automatically fill in the network paths for a given resume iteration.
        for k in opt["networks"].keys():
            pt_key = "pretrain_model_%s" % (k,)
            if pt_key in opt["path"].keys():
                # This is a dicey, error prone situation that has bitten me in both ways it can be handled. Opt for
                # a big, verbose error message.
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     WARNING      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!          YOU SPECIFIED A PRETRAINED MODEL PATH        !!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!          AND A RESUME STATE PATH. THERE IS NO         !!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!          GOOD WAY TO HANDLE THIS SO WE JUST IGNORE    !!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!          THE MODEL PATH!                              !!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            opt["path"][pt_key] = osp.join(opt["path"]["models"], "{}_{}.pth".format(resume_iter, k))
            logger.info("Set model [%s] to %s" % (k, opt["path"][pt_key]))
