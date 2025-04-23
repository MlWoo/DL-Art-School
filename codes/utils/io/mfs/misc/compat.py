from __future__ import absolute_import, division, print_function

import six as _six


def as_bytes(bytes_or_text, encoding="utf-8"):
    """Converts either bytes or unicode to `bytes`, using utf-8 encoding for
    text.

    Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

    Returns:
    A `bytes` object.

    Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, _six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError("Expected binary or unicode string, got %r" % (bytes_or_text,))


def as_text(bytes_or_text, encoding="utf-8"):
    """Returns the given argument as a unicode string.

    Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

    Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

    Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, _six.text_type):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError("Expected binary or unicode string, got %r" % bytes_or_text)


if _six.PY2:
    as_str = as_bytes
else:
    as_str = as_text


def as_str_any(value):
    """Converts to `str` as `str(value)`, but use `as_str` for `bytes`.

    Args:
    value: A object that can be converted to `str`.

    Returns:
    A `str` object.
    """
    if isinstance(value, bytes):
        return as_str(value)
    else:
        return str(value)


def path_to_str(path):
    """Returns the file system path representation of a `PathLike` object, else
    as it is.

    Args:
    path: An object that can be converted to path representation.

    Returns:
    A `str` object.
    """
    if hasattr(path, "__fspath__"):
        path = as_str_any(path.__fspath__())
    return path
