# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exception types for TensorFlow errors."""

from __future__ import absolute_import, division, print_function

OK = 0
UNKNOWN = 1
NOT_FOUND = 2
ALREADY_EXISTS = 3
PERMISSION_DENIED = 4
UNIMPLEMENTED = 5


class OpError(Exception):
    """A generic error that is raised when TensorFlow execution fails.

    Whenever possible, the session will raise a more specific subclass
    of `OpError` from the `tf.errors` module.
    """

    def __init__(self, message, error_code):
        """Creates a new `OpError` indicating that a particular op failed.

        Args:
          message: The message string describing the failure.
          error_code: The `error_codes.Code` describing the error.
        """
        super(OpError, self).__init__()
        self._message = message
        self._error_code = error_code

    @property
    def message(self):
        """The error message that describes the error."""
        return self._message

    @property
    def error_code(self):
        """The integer error code that describes the error."""
        return self._error_code

    def __str__(self):
        return self.message


class UnknownError(OpError):
    """Unknown error.

    An example of where this error may be returned is if a Status value
    received from another address space belongs to an error-space that
    is not known to this address space. Also errors raised by APIs that
    do not return enough error information may be converted to this
    error.

    @@__init__
    """

    def __init__(self, message, error_code=UNKNOWN):
        """Creates an `UnknownError`."""
        super(UnknownError, self).__init__(message, error_code)


class NotFoundError(OpError):
    """Raised when a requested entity (e.g., a file or directory) was not
    found.

    @@__init__
    """

    def __init__(self, message):
        """Creates a `NotFoundError`."""
        super(NotFoundError, self).__init__(message, NOT_FOUND)


class AlreadyExistsError(OpError):
    """Raised when an entity that we attempted to create already exists.
    @@__init__
    """

    def __init__(self, message):
        """Creates an `AlreadyExistsError`."""
        super(AlreadyExistsError, self).__init__(message, ALREADY_EXISTS)


class PermissionDeniedError(OpError):
    """Raised when the caller does not have permission to run an operation.
    @@__init__
    """

    def __init__(self, message):
        """Creates a `PermissionDeniedError`."""
        super(PermissionDeniedError, self).__init__(message, PERMISSION_DENIED)


class UnimplementedError(OpError):
    """Raised when an operation has not been implemented.
    @@__init__
    """

    def __init__(self, node_def, op, message):
        """Creates an `UnimplementedError`."""
        super(UnimplementedError, self).__init__(message, UNIMPLEMENTED)
