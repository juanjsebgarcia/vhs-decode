"""Make numba's on-disk cache (``cache=True``) safe to share between concurrent processes.

numba's ``IndexDataCacheFile.save`` does an unlocked read-modify-write of the per-function
index file (``*.nbi``) and picks the lowest unused data file number (``*.N.nbc``) from the
index it read. When two processes compile *different* type signatures of the same function at
the same time (e.g. several decodes of different formats started together on a cold cache),
both can pick the same data file number. The index entry that survives can then point at a data
file holding the other signature's machine code, and every later run loads that code for the
wrong signature without any error (arguments are silently converted, e.g. float -> int64), which
produces wrong decode output. See https://github.com/numba/numba/issues/10862.

``install()`` patches numba so that:
 - saving a cache entry happens under an exclusive per-cache-directory file lock, and the data
   file is written before the index entry that refers to it;
 - a loaded cache entry whose compiled signature does not match the requested one is ignored
   (treated as a cache miss, so it is recompiled and the bad entry is overwritten). This also
   repairs caches that were already corrupted before this guard existed.
"""

import contextlib
import os

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None

LOCK_FILE_NAME = "numba_cache.lock"

_installed = False


@contextlib.contextmanager
def _cache_dir_lock(cache_path):
    """Hold an exclusive inter-process lock for the numba cache directory *cache_path*."""
    try:
        lock_file = open(os.path.join(cache_path, LOCK_FILE_NAME), "a+b")
    except OSError:
        # Read-only or otherwise unusable location; numba will fail to save anyway.
        yield
        return

    with lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            lock_file.seek(0)
            while True:
                try:
                    # LK_LOCK retries for ~10 seconds before raising
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            yield


def _locked_save(self, key, data):
    """Replacement for numba.core.caching.IndexDataCacheFile.save."""
    with _cache_dir_lock(self._cache_path):
        overloads = self._load_index()
        data_name = overloads.get(key)
        if data_name is not None:
            # Same key (same signature) already has a data file: overwrite it.
            self._save_data(data_name, data)
            return

        existing = set(overloads.values())
        number = 1
        while self._data_name(number) in existing:
            number += 1
        data_name = self._data_name(number)
        # Write the data before the index so that the index never refers to a data file
        # that has not been written yet (or still holds an older entry's code).
        self._save_data(data_name, data)
        overloads[key] = data_name
        self._save_index(overloads)


def install():
    """Install the numba cache guard. Safe to call more than once."""
    global _installed
    if _installed:
        return

    try:
        from numba.core import caching, sigutils
    except ImportError:
        return

    index_file_class = getattr(caching, "IndexDataCacheFile", None)
    cache_class = getattr(caching, "Cache", None)
    if index_file_class is None or cache_class is None:
        return
    # Private numba API: only patch if it still looks like what this was written against.
    required = ("save", "_load_index", "_save_index", "_save_data", "_data_name")
    if not all(hasattr(index_file_class, name) for name in required):
        return
    if not hasattr(cache_class, "_load_overload"):
        return

    original_load_overload = cache_class._load_overload

    def checked_load_overload(self, sig, target_context):
        data = original_load_overload(self, sig, target_context)
        loaded_signature = getattr(data, "signature", None)
        if loaded_signature is not None:
            wanted_args, _ = sigutils.normalize_signature(sig)
            if tuple(loaded_signature.args) != tuple(wanted_args):
                return None
        return data

    index_file_class.save = _locked_save
    cache_class._load_overload = checked_load_overload
    _installed = True
