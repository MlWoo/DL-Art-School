"""
borrow from https://github.com/open-mmlab/mmcv
"""

import importlib
import inspect

import six


class Registry:
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name, registor="general"):
        self._name = name
        self._registor = registor
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, " f"items={self._module_dict})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def registor(self):
        return self._registor

    def is_registered(self, name):
        return name in self._module_dict

    def get(self, key):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered " f"in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be a str, but got {type(name)}")

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(cfg, registry: Registry, default_args=None, default_type="type"):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop(default_type)

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    return obj_cls(**args)


def get_class(str_or_class, default_mod=None):
    if isinstance(str_or_class, six.string_types):
        parts = str_or_class.split(".")
        mod_name = ".".join(parts[:-1])
        class_name = parts[-1]
        if mod_name:
            mod = importlib.import_module(mod_name)
        elif default_mod is not None:
            mod = importlib.import_module(default_mod)
        else:
            raise ValueError("Specify a module for %s" % (str_or_class,))
        return getattr(mod, class_name)
    else:
        return str_or_class


def construct_from_kwargs(object_or_kwargs, default_mod=None, additional_parameters=None):
    if not isinstance(object_or_kwargs, dict):
        assert not additional_parameters
        return object_or_kwargs
    object_kwargs = dict(object_or_kwargs)
    class_name = object_kwargs.pop("class_name")
    klass = get_class(class_name, default_mod)
    if additional_parameters:
        object_kwargs.update(additional_parameters)
    if hasattr(klass, "config_class"):
        config = klass.config_class(**object_kwargs)
        obj = klass(config)
    else:
        obj = klass(**object_kwargs)
    return obj
