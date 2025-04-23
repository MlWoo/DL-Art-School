# encoding: utf-8
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
"""A limited reimplementation of the TensorFlow FileIO API.

The TensorFlow version wraps the C++ FileSystem API.  Here we provide a
pure Python implementation, limited to the features required for
TensorBoard.  This allows running TensorBoard without depending on
TensorFlow for file operations.
"""
from __future__ import absolute_import, division, print_function

import glob as py_glob
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
from collections import namedtuple

import six

try:
    import boto3
    import botocore.exceptions

    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

try:
    import pyarrow as pa
    from pyarrow import fs as hdfs

    HDFS_ENABLED = True
except ImportError:
    HDFS_ENABLED = False


if sys.version_info < (3, 0):
    # In Python 2 FileExistsError is not defined and the
    # error manifests it as OSError.
    FileExistsError = OSError

from .misc import compat, errors

# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


# Registry of filesystems by prefix.
#
# Currently supports "s3://" URLs for S3 based on boto3 and falls
# back to local filesystem.
_REGISTERED_FILESYSTEMS = {}


def register_filesystem(prefix, filesystem):
    if ":" in prefix:
        raise ValueError("Filesystem prefix cannot contain a :")
    _REGISTERED_FILESYSTEMS[prefix] = filesystem


def get_filesystem(filename):
    """Return the registered filesystem for the given file."""
    filename = compat.as_str_any(filename)
    prefix = ""
    index = filename.find("://")
    if index >= 0:
        prefix = filename[:index]
    fs = _REGISTERED_FILESYSTEMS.get(prefix, None)
    if fs is None:
        raise ValueError("No recognized filesystem for prefix %s" % prefix)
    return fs


# Data returned from the Stat call.
StatData = namedtuple("StatData", ["length"])


class LocalFileSystem(object):
    """Provides local fileystem access."""

    def exists(self, filename):
        """Determines whether a path exists or not."""
        return os.path.exists(compat.as_bytes(filename))

    def delete(self, path, recursive=False):
        if self.isdir(path):
            if len(self.listdir(path)) > 0:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
        else:
            os.remove(path)

    def join(self, path, *paths):
        """Join paths with path delimiter."""
        return os.path.join(path, *paths)

    def tell(self, continue_from):
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("opaque_offset", 0)
        return offset

    def seek(self, offset):
        continuation_token = {"opaque_offset": offset}
        return continuation_token

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        if not exists(filename):
            raise errors.NotFoundError("Not Found: " + compat.as_text(filename))
        offset = None
        if continue_from is not None:
            offset = continue_from.get("opaque_offset", None)
        with io.open(filename, mode, encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)
            # The new offset may not be `offset + len(data)`, due to decoding
            # and newline translation.
            # So, just measure it in whatever terms the underlying stream uses.
            continuation_token = {"opaque_offset": f.tell()}
            return (data, continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file, overwriting any existing
        contents.

        Args:
          filename: string, a path
          file_content: string, the contents
          binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "wb" if binary_mode else "w")

    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
          filename: string, a path
          file_content: string, the contents to append
          binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "ab" if binary_mode else "a")

    def _write(self, filename, file_content, mode):
        encoding = None if "b" in mode else "utf8"
        with io.open(filename, mode, encoding=encoding) as f:
            compatify = compat.as_bytes if "b" in mode else compat.as_text
            f.write(compatify(file_content))

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, six.string_types):
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for matching_filename in py_glob.glob(compat.as_bytes(filename))
            ]
        else:
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for single_filename in filename
                for matching_filename in py_glob.glob(compat.as_bytes(single_filename))
            ]

    def isfile(self, filename):
        if os.path.isfile(filename):
            return True
        else:
            # print(f"Could not find file {filename}")
            return False

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        return os.path.isdir(compat.as_bytes(dirname))

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        if not self.isdir(dirname):
            raise errors.NotFoundError(f"Could not find directory {dirname}")

        entries = os.listdir(compat.as_str_any(dirname))
        entries = [compat.as_str_any(item) for item in entries]
        return entries

    def makedirs(self, path):
        """Creates a directory and all parent/intermediate directories."""
        try:
            os.makedirs(path)
        except FileExistsError:
            raise errors.AlreadyExistsError("Directory already exists")

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by .st_size as returned from
        # os.stat(), but we convert to .length
        try:
            file_length = os.stat(compat.as_bytes(filename)).st_size
        except OSError:
            raise errors.NotFoundError("Could not find file")
        return StatData(file_length)

    def copyfile(self, src_fn, dst_fn):
        try:
            shutil.copyfile(src_fn, dst_fn)
        except OSError:
            raise errors.NotFoundError(f"Could not find the source file {src_fn}")

    def move(self, src_fn, dst_fn):
        try:
            shutil.move(src_fn, dst_fn)
        except OSError:
            raise errors.NotFoundError(f"Could not find the source file {src_fn}")

    def abspath(self, path):
        return os.path.abspath(path)


class S3FileSystem(object):
    """Provides filesystem access to S3."""

    def __init__(self):
        if not boto3:
            raise ImportError("boto3 must be installed for S3 support.")
        self._s3_endpoint = os.environ.get("S3_ENDPOINT", None)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        url = compat.as_str_any(url)
        if url.startswith("s3://"):
            url = url[len("s3://") :]
        idx = url.index("/")
        bucket = url[:idx]
        path = url[(idx + 1) :]
        return bucket, path

    def exists(self, filename):
        """Determines whether a path exists or not."""
        session = boto3.Session(profile_name="b-yarn")
        client = session.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def join(self, path, *paths):
        """Join paths with a slash."""
        return "/".join((path,) + paths)

    def tell(self, continue_from):
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)
        return offset

    def seek(self, offset):
        continuation_token = {"byte_offset": offset}
        return continuation_token

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
          filename: string, a path
          binary_mode: bool, read as binary if True, otherwise text
          size: int, number of bytes or characters to read, otherwise
              read all the contents of the file (from the continuation
              marker, if present).
          continue_from: An opaque value returned from a prior invocation of
              `read(...)` marking the last read position, so that reading
              may continue from there.  Otherwise read from the beginning.

        Returns:
          A tuple of `(data, continuation_token)` where `data' provides either
          bytes read from the file (if `binary_mode == true`) or the decoded
          string representation thereof (otherwise), and `continuation_token`
          is an opaque value that can be passed to the next invocation of
          `read(...) ' in order to continue from the last read position.
        """
        s3 = boto3.resource("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        args = {}

        # For the S3 case, we use continuation tokens of the form
        # {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = ""
        if size is not None:
            # TODO(orionr): This endpoint risks splitting a multi-byte
            # character or splitting \r and \n in the case of CRLFs,
            # producing decoding errors below.
            endpoint = offset + size

        if offset != 0 or endpoint != "":
            # Asked for a range, so modify the request
            args["Range"] = "bytes={}-{}".format(offset, endpoint)

        try:
            stream = s3.Object(bucket, path).get(**args)["Body"].read()
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in ["416", "InvalidRange"]:
                if size is not None:
                    # Asked for too much, so request just to the end. Do this
                    # in a second request so we don't check length in all cases.
                    client = boto3.client("s3", endpoint_url=self._s3_endpoint)
                    obj = client.head_object(Bucket=bucket, Key=path)
                    content_length = obj["ContentLength"]
                    endpoint = min(content_length, offset + size)
                if offset == endpoint:
                    # Asked for no bytes, so just return empty
                    stream = b""
                else:
                    args["Range"] = "bytes={}-{}".format(offset, endpoint)
                    stream = s3.Object(bucket, path).get(**args)["Body"].read()
            else:
                raise
        # `stream` should contain raw bytes here (i.e., there has been neither
        # decoding nor newline translation), so the byte offset increases by
        # the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return (bytes(stream), continuation_token)
        else:
            return (stream.decode("utf-8"), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
          filename: string, a path
          file_content: string, the contents
          binary_mode: bool, write as binary if True, otherwise text
        """
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, six.binary_type):
                raise TypeError("File content type must be bytes")
        else:
            file_content = compat.as_bytes(file_content)
        client.put_object(Body=file_content, Bucket=bucket, Key=path)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        if quest_i >= 0:
            raise NotImplementedError("{} not supported by compat glob".format(filename))
        if star_i != len(filename) - 1:
            # Just return empty so we can use glob from directory watcher
            #
            # TODO: Remove and instead handle in GetLogdirSubdirectories.
            # However, we would need to handle it for all non-local registered
            # filesystems in some way.
            return []
        filename = filename[:-1]
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        p = client.get_paginator("list_objects")
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path):
            for o in r.get("Contents", []):
                key = o["Key"][len(path) :]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        p = client.get_paginator("list_objects")
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path, Delimiter="/"):
            keys.extend(o["Prefix"][len(path) : -1] for o in r.get("CommonPrefixes", []))
            for o in r.get("Contents", []):
                key = o["Key"][len(path) :]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(key)
        return keys

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        if self.exists(dirname):
            raise errors.AlreadyExistsError("Directory already exists")
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will make sure we don't override a file
        client.put_object(Body="", Bucket=bucket, Key=path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        try:
            obj = client.head_object(Bucket=bucket, Key=path)
            return StatData(obj["ContentLength"])
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                raise errors.NotFoundError("Could not find file")
            else:
                raise

    def abspath(self, path):
        return path


class HDFSFileSystem(object):
    """Provides filesystem access to S3."""

    _CLIENTS = {}
    _BUCKETS = set()

    def __init__(self):
        self.bucket = None
        self.client = None
        self.old_client = None
        # self.port = -1

    @classmethod
    def _connect(cls, bucket, port=0):
        old_client = pa.hdfs.connect(host=bucket, port=port)
        new_client = hdfs.HadoopFileSystem(host=bucket, port=port)
        old_bucket_flag = "old_" + bucket
        new_bucket_flag = "new_" + bucket
        HDFSFileSystem._CLIENTS[old_bucket_flag] = old_client
        HDFSFileSystem._CLIENTS[new_bucket_flag] = new_client
        HDFSFileSystem._BUCKETS.add(bucket)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        url = compat.as_str_any(url)
        if url.startswith("hdfs://"):
            url = url[len("hdfs://") :]
        idx = url.index("/")
        bucket = url[:idx]
        path = url[idx:]
        return bucket, path

    def connect(self, bucket, port=0, old=False):
        # port = self.ports_available.popleft()
        # self.port = port
        """
        if old:
            client = pa.hdfs.connect(host=bucket, port=port)
        else:
            client = hdfs.HadoopFileSystem(host=bucket, port=port)
        return client
        """
        """
        if self.bucket is not None:
            if self.bucket != bucket:
                print(f'Accessing bucket is {bucket}, but historical bucket is {self.bucket}. \n\
                The old connections will logout.')
                self.old_client = None
                self.client = None
        if old:
            if self.old_client is None:
                self.old_client = pa.hdfs.connect(host=bucket, port=port)
                self.bucket = bucket
            return self.old_client
        else:
            if self.client is None:
                self.client = hdfs.HadoopFileSystem(host=bucket, port=port)
                self.bucket = bucket
            return self.client
        """
        if bucket not in HDFSFileSystem._BUCKETS:
            HDFSFileSystem._connect(bucket, port=port)
        if old:
            bucket_flag = "old_" + bucket
        else:
            bucket_flag = "new_" + bucket

        if bucket_flag in HDFSFileSystem._CLIENTS.keys():
            return HDFSFileSystem._CLIENTS[bucket_flag]
        else:
            raise ValueError("Connect error")

    def exists(self, filename):
        """Determines whether a path exists or not."""
        # bucket, path = self.bucket_and_path(filename)
        # fs_ = pa.fs.HadoopFileSystem(host=bucket, port=0)
        # stat = fs_.get_file_info(filename)
        bucket, path = self.bucket_and_path(filename)
        client = self.connect(bucket=bucket)
        file_info = client.get_file_info(path)
        if file_info.type.value == file_info.type.NotFound:
            return False
        elif file_info.type.value == file_info.type.Unknown:
            print(f"Unknown type: {filename}")
            return False
        else:
            return True

    def delete(self, path, recursive=False):
        if self.exists(path):
            if self.isfile(path):
                bucket, path = self.bucket_and_path(path)
                client = self.connect(bucket=bucket)
                client.delete_file(path)
            elif self.isdir(path):
                if recursive:
                    bucket, path = self.bucket_and_path(path)
                    client = self.connect(bucket=bucket)
                    client.delete(path, recursive)
                else:
                    raise ValueError("HDFS only suport delete recursively")

    def join(self, path, *paths):
        """Join paths with a slash."""
        return "/".join((path,) + paths)

    def tell(self, continue_from):
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("h_byte_offset", 0)
        return offset

    def seek(self, offset):
        continuation_token = {"h_byte_offset": offset}
        return continuation_token

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
          filename: string, a path
          binary_mode: bool, read as binary if True, otherwise text
          size: int, number of bytes or characters to read, otherwise read all the
            contents of the file (from the continuation marker, if present).
          continue_from: An opaque value returned from a prior invocation of `read(...)`
            marking the last read position, so that reading may continue from there.
            Otherwise read from the beginning.

        Returns:
          A tuple of `(data, continuation_token)` where `data' provides either
          bytes read from the file (if `binary_mode == true`) or the decoded
          string representation thereof (otherwise), and `continuation_token`
          is an opaque value that can be passed to the next invocation of
          `read(...) ' in order to continue from the last read position.
        """
        bucket, path = self.bucket_and_path(filename)

        offset = 0
        if continue_from is not None:
            offset = continue_from.get("h_byte_offset", 0)

        client = self.connect(bucket=bucket)

        with client.open_input_file(filename) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)

            # `stream` should contain raw bytes here (i.e., there has been neither
            # decoding nor newline translation), so the byte offset increases by
            # the expected amount.
            continuation_token = {"h_byte_offset": f.tell()}

            if binary_mode:
                return (bytes(data), continuation_token)
            else:
                return (data.decode("utf-8"), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
          filename: string, a path
          file_content: string, the contents
          binary_mode: bool, write as binary if True, otherwise text
        """
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, six.binary_type):
                raise TypeError("File content type must be bytes")
        else:
            file_content = compat.as_bytes(file_content)
        bucket, path = self.bucket_and_path(filename)
        try:
            client = self.connect(bucket=bucket)
            with client.open_output_stream(filename) as f:
                f.write(file_content)
        except:  # noqa: E722
            HDFSFileSystem._BUCKETS.remove(bucket)
            client = self.connect(bucket=bucket)
            with client.open_output_stream(filename) as f:
                f.write(file_content)
        if not f.closed:
            f.close()

    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
          filename: string, a path
          file_content: string, the contents to append
          binary_mode: bool, write as binary if True, otherwise text
        """
        bucket, path = self.bucket_and_path(filename)
        client = self.connect(bucket=bucket)
        with client.open_append_stream(filename) as f:
            # Always convert to bytes for writing
            if binary_mode:
                if not isinstance(file_content, six.binary_type):
                    raise TypeError("File content type must be bytes")
            else:
                file_content = compat.as_bytes(file_content)
            f.write(file_content)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        slash_i = filename.rfind("/")
        if quest_i >= 0:
            raise NotImplementedError("{} not supported by compat glob".format(filename))

        if slash_i > 0 and star_i > slash_i:
            dirname = filename[:slash_i]
            filename_exp = filename[(slash_i + 1) :]
            filename_re = re.sub(r"\*", r"[\\s\\S]*", filename_exp)
            item_all = self.listdir(dirname)
            keys = []
            for item in item_all:
                if re.search(filename_re, item) is not None:
                    keys.append(os.path.join(dirname, item))
            return keys
        else:
            if star_i != len(filename) - 1:
                # Just return empty so we can use glob from directory watcher
                #
                # TODO: Remove and instead handle in GetLogdirSubdirectories.
                # However, we would need to handle it for all non-local registered
                # filesystems in some way.
                return []
            filename = filename[:-1]

            bucket, path = self.bucket_and_path(filename)
            dirname = os.path.dirname(path)
            item_all = self.listdir(dirname)
            keys = []
            for item in item_all:
                _, __key = self.bucket_and_path(item)
                __path = __key[len(path) :]
                if __path == path:  # Skip the base dir, which would add an empty string
                    keys.append(__path)
            return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        # if not dirname.endswith("/"):
        #    dirname += "/"  # This will now only retrieve subdir content
        bucket, path = self.bucket_and_path(dirname)
        client = self.connect(bucket=bucket)
        file_info = client.get_file_info(path)
        if file_info.type.value == file_info.type.Directory:
            return True
        else:
            return False

    def isfile(self, filename):
        """Returns whether the path is a directory or not."""
        # if not dirname.endswith("/"):
        #    dirname += "/"  # This will now only retrieve subdir content
        bucket, path = self.bucket_and_path(filename)
        client = self.connect(bucket=bucket)
        file_info = client.get_file_info(path)
        if file_info.is_file:
            return True
        else:
            return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        if not self.isdir(dirname):
            raise errors.NotFoundError("Could not find directory")
        bucket, path = self.bucket_and_path(dirname)
        old_client = self.connect(bucket=bucket, old=True)
        full_keys = old_client.ls(path)

        keys = []
        for full_path in full_keys:
            _, key = self.bucket_and_path(full_path)
            basename = os.path.basename(key)
            keys.append(basename)
        return keys

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        if self.exists(dirname):
            raise errors.AlreadyExistsError("Directory already exists")
        bucket, path = self.bucket_and_path(dirname)
        client = self.connect(bucket=bucket)
        if not path.endswith("/"):
            path += "/"  # This will make sure we don't override a file
        client.create_dir(path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length
        bucket, path = self.bucket_and_path(filename)
        client = self.connect(bucket=bucket)
        try:
            obj = client.open_input_file(filename)
            return StatData(obj.size())
        except OSError:
            raise errors.NotFoundError("Could not find directory or file")

    def copyfile(self, src_fn, dst_fn):
        bucket, path = self.bucket_and_path(src_fn)
        client = self.connect(bucket=bucket)
        try:
            client.copy_file(src_fn, dst_fn)
        except OSError:
            raise errors.NotFoundError(f"Could not find the source file {src_fn}")

    def abspath(self, path):
        return path


register_filesystem("", LocalFileSystem())
if S3_ENABLED:
    register_filesystem("s3", S3FileSystem())
if HDFS_ENABLED:
    register_filesystem("hdfs", HDFSFileSystem())


class GFile(object):
    # Only methods needed for TensorBoard are implemented.
    def __init__(self, filename, mode, flood=False, delay_flush=False):
        if mode not in ("r", "rb", "br", "w", "wb", "bw", "a", "ab", "ba"):
            raise NotImplementedError("mode {} not supported by compat GFile".format(mode))
        self.filename = compat.as_bytes(filename)
        self.fs = get_filesystem(self.filename)
        self.fs_supports_append = hasattr(self.fs, "append")
        self.buff = None
        # The buffer offset and the buffer chunk size are measured in the
        # natural units of the underlying stream, i.e. bytes for binary mode,
        # or characters in text mode.
        self.buff_chunk_size = _DEFAULT_BLOCK_SIZE
        self.buff_offset = 0
        self.continuation_token = None
        self.write_temp = None
        self.write_started = "a" in mode and self.fs.exists(filename)
        self.binary_mode = "b" in mode
        self.write_mode = "w" in mode or "a" in mode
        self.buff_file_offset = 0
        self.read_offset = 0

        self.closed = False
        # force event writer from tensorboard to delay flushing data to disk or hdfs
        frame_info = inspect.stack()
        call_file = frame_info[1][1]
        if call_file.endswith("event_file_writer.py"):
            self.delay_flush = True
            flood = False
        else:
            self.delay_flush = delay_flush
        if flood:
            if self.write_mode:
                self.file_handle = self.fs.write_file_handle(self.filename)
            else:
                self.file_handle = self.fs.read_file_handle(self.filename)
        else:
            self.file_handle = None
        self.flood = flood

    @classmethod
    def open(cls, filename, mode):
        return GFile(filename, mode)

    def __enter__(self):
        if self.flood:
            return self.file_handle
        else:
            return self

    def __exit__(self, *args):
        if self.flood:
            self.file_handle.close()
        else:
            self.close()
            self.buff = None
            self.buff_offset = 0
            self.continuation_token = None
            self.buff_file_offset = 0
            self.read_offset = 0

    def __iter__(self):
        if self.flood:
            return self.file_handle
        else:
            return self

    def _read_buffer_to_offset(self, new_buff_offset):
        old_buff_offset = self.buff_offset
        read_size = min(len(self.buff), new_buff_offset) - old_buff_offset
        self.buff_offset += read_size
        return self.buff[old_buff_offset : old_buff_offset + read_size]

    def tell(self):
        return self.read_offset

    def seek(self, offset=0):
        self.read_offset = offset
        if offset > self.fs.tell(self.continuation_token):
            self.continuation_token = self.fs.seek(offset)
        if offset < self.buff_file_offset:
            self.continuation_token = self.fs.seek(offset)

    def read(self, n=None):
        """Reads contents of file to a string.

        Args:
          n: int, number of bytes or characters to read, otherwise read all the contents of the file

        Returns:
          Subset of the contents of the file as a string or bytes.
        """
        if self.flood:
            return self.file_handle.read()
        if self.write_mode:
            raise errors.PermissionDeniedError("File not opened in read mode")
        result = None
        read_start_offset = self.tell()

        if (
            self.buff
            and (read_start_offset >= self.buff_file_offset)
            and (read_start_offset <= self.buff_file_offset + len(self.buff))
        ):
            self.buff_offset = read_start_offset - self.buff_file_offset

            if n is not None:
                chunk = self._read_buffer_to_offset(self.buff_offset + n)
                if len(chunk) == n:
                    self.read_offset += n
                    return chunk
                result = chunk
                read_n = n - len(chunk)
            else:
                # add all local buffer and update offsets
                result = self._read_buffer_to_offset(len(self.buff))
                read_n = n
        else:
            read_n = n

        self.buff_file_offset = self.fs.tell(self.continuation_token)
        # read from filesystem
        read_size = max(self.buff_chunk_size, read_n) if read_n is not None else None

        (self.buff, self.continuation_token) = self.fs.read(
            self.filename, self.binary_mode, read_size, self.continuation_token
        )
        self.buff_offset = 0
        if read_size is None:
            self.read_offset = self.fs.tell(self.continuation_token)
        else:
            self.read_offset += n

        # add from filesystem
        if read_n is not None:
            chunk = self._read_buffer_to_offset(read_n)
        else:
            # add all local buffer and update offsets
            chunk = self._read_buffer_to_offset(len(self.buff))
        result = result + chunk if result else chunk

        return result

    def readinto(self, n=None):
        result = self.read(n)
        return len(result)

    def write(self, file_content):
        """Writes string file contents to file, clearing contents of the file
        on first write and then appending on subsequent calls.

        Args:
            file_content: string, the contents
        """
        if self.flood:
            self.file_handle.write(file_content)
            return
        if not self.write_mode:
            raise errors.PermissionDeniedError("File not opened in write mode")
        if self.closed:
            raise errors.FailedPreconditionError("File already closed")

        if self.fs_supports_append and (not self.delay_flush):
            if not self.write_started:
                # write the first chunk to truncate file if it already exists
                self.fs.write(self.filename, file_content, self.binary_mode)
                self.write_started = True
            else:
                # append the later chunks
                self.fs.append(self.filename, file_content, self.binary_mode)
        else:
            # add to temp file, but wait for flush to write to final filesystem
            if self.write_temp is None:
                mode = "w+b" if self.binary_mode else "w+"
                self.write_temp = tempfile.TemporaryFile(mode)

            compatify = compat.as_bytes if self.binary_mode else compat.as_text
            self.write_temp.write(compatify(file_content))
            # flush data to disk actively.
            if self.write_temp.tell() >= _DEFAULT_BLOCK_SIZE:
                self.flush()

    def __next__(self):
        line = None
        while True:
            if not self.buff:
                # read one unit into the buffer
                line = self.read(1)
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()
            else:
                index = self.buff.find("\n", self.buff_offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index + 1 - self.buff_offset)
                    line = line + chunk if line else chunk
                    return line
                # read one unit past end of buffer
                chunk = self.read(len(self.buff) + 1 - self.buff_offset)
                line = line + chunk if line else chunk
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()

    def __in_next__(self):
        line = None
        while True:
            if not self.buff:
                # read one unit into the buffer
                chunk = self.read(1)
                line = line + chunk if line else chunk
                if line and line[-1] == "\n":
                    return line
                if not self.buff:
                    return False
            else:
                index = self.buff.find("\n", self.buff_offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index + 1 - self.buff_offset)
                else:
                    # read one unit past end of buffer
                    chunk = self.read(len(self.buff) + 1 - self.buff_offset)
                line = line + chunk if line else chunk

                if line and line[-1] == "\n":
                    return line

    def readline(self):
        return self.__in_next__()

    def readlines(self):
        lines = []
        while True:
            s = self.readline()
            if not s:
                break
            lines.append(s)
        return lines

    def next(self):
        return self.__next__()

    def flush(self):
        if self.closed:
            raise errors.FailedPreconditionError("File already closed")

        if not self.fs_supports_append:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                chunk = self.write_temp.read()
                if chunk is not None:
                    # write full contents and keep in temp file
                    self.fs.write(self.filename, chunk, self.binary_mode)
                    self.write_temp.seek(len(chunk))
        elif self.delay_flush:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                temp_pos = self.write_temp.tell()
                chunk = self.write_temp.read()
                try:
                    if chunk is not None:
                        if not self.write_started:
                            # write the first chunk to truncate file if it already exists
                            self.fs.write(self.filename, chunk, self.binary_mode)
                            self.write_started = True
                        else:
                            # append the later chunks
                            self.fs.append(self.filename, chunk, self.binary_mode)
                    self.write_temp = None
                except:  # noqa: E722
                    self.write_temp.seek(temp_pos)

    def close(self):
        self.flush()
        if self.write_temp is not None:
            self.write_temp.close()
            self.write_temp = None
            self.write_started = False
        self.closed = True
        if getattr(self.fs, "close", False):
            if callable(getattr(self.fs, "close")):
                self.fs.close()


def exists(filename):
    """Determines whether a path exists or not.

    Args:
      filename: string, a path

    Returns:
      True if the path exists, whether its a file or a directory.
      False if the path does not exist and there are no filesystem errors.

    Raises:
      errors.OpError: Propagates any errors reported by the FileSystem API.
    """
    return get_filesystem(filename).exists(filename)


def glob(filename):
    """Returns a list of files that match the given pattern(s).

    Args:
      filename: string or iterable of strings. The glob pattern(s).

    Returns:
      A list of strings containing filenames that match the given pattern(s).

    Raises:
      errors.OpError: If there are filesystem / directory listing errors.
    """
    return get_filesystem(filename).glob(filename)


def isdir(dirname):
    """Returns whether the path is a directory or not.

    Args:
      dirname: string, path to a potential directory

    Returns:
      True, if the path is a directory; False otherwise
    """
    return get_filesystem(dirname).isdir(dirname)


def isfile(dirname):
    """Returns whether the path is a directory or not.

    Args:
      dirname: string, path to a potential directory

    Returns:
      True, if the path is a directory; False otherwise
    """
    return get_filesystem(dirname).isfile(dirname)


def listdir(dirname):
    """Returns a list of entries contained within a directory.

    The list is in arbitrary order. It does not contain the special entries "."
    and "..".

    Args:
      dirname: string, path to a directory

    Returns:
      [filename1, filename2, ... filenameN] as strings

    Raises:
      errors.NotFoundError if directory doesn't exist
    """
    return get_filesystem(dirname).listdir(dirname)


def makedirs(path, exist_ok=True):
    """Creates a directory and all parent/intermediate directories.

    It succeeds if path already exists and is writable.

    Args:
      path: string, name of the directory to be created

    Raises:
      errors.AlreadyExistsError: If leaf directory already exists or
        cannot be created.
    """
    if exist_ok:
        if not exists(path):
            return get_filesystem(path).makedirs(path)
    else:
        return get_filesystem(path).makedirs(path)


def walk(top, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.

    Args:
      top: string, a Directory name
      topdown: bool, Traverse pre order if True, post order if False.
      onerror: optional handler for errors. Should be a function, it will be
        called with the error as argument. Rethrowing the error aborts the walk.

    Errors that happen while listing directories are ignored.

    Yields:
      Each yield is a 3-tuple:  the pathname of a directory, followed by lists
      of all its subdirectories and leaf files.
      (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
      as strings
    """
    top = compat.as_str_any(top)
    fs = get_filesystem(top)
    try:
        listing = listdir(top)
    except errors.NotFoundError as err:
        if onerror:
            onerror(err)
        else:
            return

    files = []
    subdirs = []
    for item in listing:
        full_path = fs.join(top, compat.as_str_any(item))
        if isdir(full_path):
            subdirs.append(item)
        else:
            files.append(item)

    here = (top, subdirs, files)

    if topdown:
        yield here

    for subdir in subdirs:
        joined_subdir = fs.join(top, compat.as_str_any(subdir))
        for subitem in walk(joined_subdir, topdown, onerror=onerror):
            yield subitem

    if not topdown:
        yield here


def stat(filename):
    """Returns file statistics for a given path.

    Args:
      filename: string, path to a file

    Returns:
      FileStatistics struct that contains information about the path

    Raises:
      errors.OpError: If the operation fails.
    """
    return get_filesystem(filename).stat(filename)


def delete(path, recursive=False):
    return get_filesystem(path).delete(path, recursive)


def move(src_fn, dst_fn):
    """Returns file statistics for a given path.

    Args:
      filename: string, path to a file

    Returns:
      FileStatistics struct that contains information about the path

    Raises:
      errors.OpError: If the operation fails.
    """
    return get_filesystem(src_fn).move(src_fn, dst_fn)


def copyfile(src_fn, dst_fn):
    """Returns file statistics for a given path.

    Args:
      filename: string, path to a file

    Returns:
      FileStatistics struct that contains information about the path

    Raises:
      errors.OpError: If the operation fails.
    """
    return get_filesystem(src_fn).copyfile(src_fn, dst_fn)


def abspath(path):
    return get_filesystem(path).abspath(path)


def islocal(path):
    """Return the registered filesystem for the given file."""
    path = compat.as_str_any(path)
    index = path.find("://")
    if index >= 0:
        return False
    else:
        return True


if __name__ == "__main__":
    if HDFS_ENABLED:
        print("HDFS_ENABLED")
    if S3_ENABLED:
        print("S3_ENABLED")
