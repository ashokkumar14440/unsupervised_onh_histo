import abc
import json
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Union

import jsonschema


class ConfigFile:
    """A self-aware abstraction for JSON configuration files. The abstraction
    can be assigned any number of on-change callbacks, which can be used for
    features like autosaving when changed. By default, overwriting is turned
    off for safety. Fields are accessible both dict-style and by dot notation. It is
    possible to use arrays in the configuration, but they may only be accessed
    list-style, and not by dot notation.

    Example usages:

    Access...
    config = ConfigFile("config.json")
    dot_value = config.dict.value
    VALUE_KEY = "value"
    dict_value = config["dict"][VALUE_KEY]
    LIST_INDEX = 0
    list_value = config.list[LIST_INDEX]

    Autosaving...
    config.on_change_callbacks["autosave"] = config.AUTOSAVE_CALLBACK
    config.overwrite_on()
    config.dict.value = "new_value"
    """

    AUTOSAVE_CALLBACK = lambda x: x.save()

    # Raises OSError
    def __init__(self, config_path=None, schema_path=None):
        data = {}
        if config_path is not None:
            with open(str(config_path), mode="r") as f:
                data = json.load(f)
        schema = None
        if schema_path is not None:
            with open(str(schema_path), mode="r") as f:
                schema = json.load(f)
        if schema is not None:
            self._validate(data, schema)
        self._head = _to_config(self, data)
        self._path = config_path
        self._schema = schema
        self._allow_overwrite = False
        self._on_change_callbacks = {}

    def __contains__(self, key):
        return self._head.__contains__(key)

    def __delitem__(self, key):
        del self._head[key]

    def __eq__(self, other):
        if isinstance(other, ConfigFile):
            return self._head == other._head
        if isinstance(other, (ConfigItem, dict, list)):
            return self._head == other
        else:
            return False

    def __getattr__(self, key):
        if key in (
            "_head",
            "_path",
            "_schema",
            "_allow_overwrite",
            "_on_change_callbacks",
            "overwrite",
            "path",
            "on_change_callbacks",
            "clear",
            "overwrite_off",
            "overwrite_on",
            "save",
            "to_json",
            "_has_changed",
            "_validate",
        ):
            return super().__getattr__(key)
        if self.__contains__(key):
            return self.__getitem__(key)
        else:
            raise AttributeError()

    def __getitem__(self, key):
        return self._head[key]

    def __setattr__(self, key, value):
        # Ignoring member variables defined in __init__()
        if key in (
            "_head",
            "_path",
            "_schema",
            "_allow_overwrite",
            "_on_change_callbacks",
            "overwrite",
            "path",
            "on_change_callbacks",
            "clear",
            "overwrite_off",
            "overwrite_on",
            "save",
            "to_json",
            "_has_changed",
            "_validate",
        ):
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __iter__(self):
        return iter(self._head)

    def __len__(self):
        return len(self._head)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, key, value):
        self._head[key] = _to_config(self, value)

    def __str__(self):
        return self.to_json().__str__()

    @property
    def overwrite(self):
        return self._allow_overwrite

    @overwrite.setter
    def overwrite(self, value: bool):
        assert value in (False, True)
        self._allow_overwrite = value

    @property
    def path(self):
        return PurePath(self._path)

    @path.setter
    def path(self, value: Union[str, PurePath, Path]):
        assert isinstance(value, (str, PurePath, Path))
        self._path = value

    @property
    def on_change_callbacks(self):
        return self._on_change_callbacks

    @on_change_callbacks.setter
    def on_change_callbacks(self, value: Dict[Any, Callable[["ConfigFile"], None]]):
        self._on_change_callbacks = value

    def clear(self):
        self._head.clear()
        self._has_changed()

    def overwrite_off(self):
        self._allow_overwrite = False

    def overwrite_on(self):
        self._allow_overwrite = True

    def save(self):
        path = self._path
        if path is None:
            return
        if self._allow_overwrite or not Path(path).is_file():
            with open(str(path), mode="w") as f:
                json.dump(self.to_json(), f)

    def to_json(self):
        """Returns an object conformant with built-in json library."""
        return _from_config(self._head)

    def _has_changed(self):
        """Must be called by children when their children change."""
        if self._schema is not None:
            self._validate(self.to_json(), self._schema)
        for callback in self.on_change_callbacks.values():
            callback(self)

    @staticmethod
    def _validate(json, schema):
        jsonschema.validate(json, schema)


class ConfigItem(abc.ABC):
    """Base class for children of ConfigFile."""

    def __init__(self, parent, children):
        self._children = children
        self._parent = parent

    def __contains__(self, key):
        return self._children.__contains__(key)

    def __delitem__(self, key):
        del self._children[key]
        self._has_changed()

    @abc.abstractmethod
    def __eq__(self, other):
        assert False

    def __getitem__(self, key):
        return self._children[key]

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_json().__str__()

    def clear(self):
        self._children.clear()
        self._has_changed()

    @abc.abstractmethod
    def to_json(self):
        assert False

    def _has_changed(self):
        self._parent._has_changed()


class ConfigList(ConfigItem):
    def __init__(self, parent, value):
        children = [_to_config(self, child) for child in value]
        super().__init__(parent, children)

    def __eq__(self, other):
        if isinstance(other, ConfigList):
            return _from_config(self) == _from_config(other)
        elif isinstance(other, list):
            return _from_config(self) == other
        else:
            return False

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._children[key] = [_to_config(self, x) for x in value]
        else:
            self._children[key] = _to_config(self, value)
        self._has_changed()

    def append(self, x):
        value = _to_config(self, x)
        self._children.append(x)
        self._has_changed()

    def count(self, x):
        return self._children.count(x)

    def extend(self, iterable):
        [_to_config(self, x) for x in iterable]
        self._children.extend(iterable)
        self._has_changed()

    def index(self, x, *args):
        return self._children.index(x, *args)

    def insert(self, i, x):
        value = _to_config(self, x)
        self._children.insert(i, value)
        self._has_changed()

    def pop(self, *args):
        popped = _from_config(self._children.pop(*args))
        self._has_changed()
        return popped

    def remove(self, x):
        self._children.remove(x)
        self._has_changed()

    def reverse(self):
        self._children.reverse()
        self._has_changed()

    def sort(self, **kwargs):
        self._children.sort(**kwargs)
        self._has_changed()

    def to_json(self):
        return [_from_config(child) for child in self._children]


class ConfigDict(ConfigItem):
    def __init__(self, parent, value):
        children = {key: _to_config(self, child) for key, child in value.items()}
        super().__init__(parent, children)

    def __eq__(self, other):
        if isinstance(other, ConfigDict):
            return _from_config(self) == _from_config(other)
        elif isinstance(other, dict):
            return _from_config(self) == other
        else:
            return False

    def __getattr__(self, key):
        if key in ("_children", "_parent"):
            super().__getattr__(key)
        elif self.__contains__(key):
            return self.__getitem__(key)
        else:
            raise AttributeError()

    def __setattr__(self, key, value):
        if key in ("_children", "_parent"):
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __reversed__(self):
        return reversed(self._children)

    def __setitem__(self, key, value):
        self._children[key] = _to_config(self, value)
        self._has_changed()

    @classmethod
    def fromkeys(cls, iterable, *args):
        return dict.fromkeys(iterable, *args)

    def get(self, key, *args):
        return _from_config(self._children.get(key, *args))

    def items(self):
        return _from_config(self._children).items()

    def keys(self):
        return self._children.keys()

    def list(self):
        return list(_from_config(self._children))

    def pop(self, key, *args):
        popped = _from_config(self._children.pop(key, *args))
        self._has_changed()
        return popped

    def popitem(self):
        popped = self._children.popitem()
        popped = _from_config(tuple(popped))
        self._has_changed()
        return popped

    def setdefault(self, key, *args):
        if not args:
            args = [None]
        self._children.setdefault(key, _to_config(self, *args))

    def to_json(self):
        return {key: _from_config(child) for key, child in self._children.items()}

    def update(self, *args):
        self._children.update(_to_config(self, *args))

    def values(self):
        return self._children.values()


def _check_config_value(value):
    assert (
        isinstance(value, (list, tuple, dict))
        or isinstance(value, (int, float, str, bool))
        or isinstance(value, ConfigItem)
        or value is None
    )


def _to_config(parent, value):
    _check_config_value(value)
    if isinstance(value, dict):
        out = ConfigDict(parent=parent, value=value)
    elif isinstance(value, (list, tuple)):
        out = ConfigList(parent=parent, value=value)
    else:
        out = value
    return out


def _from_config(value):
    if isinstance(value, ConfigItem):
        return value.to_json()
    else:
        return value
