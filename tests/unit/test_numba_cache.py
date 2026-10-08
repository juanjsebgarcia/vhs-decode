import os
import pickle
import subprocess
import sys
import textwrap
import threading

import pytest

numba_caching = pytest.importorskip("numba.core.caching")

from lddecode import numba_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JIT_MODULE = textwrap.dedent(
    """
    from numba import njit

    @njit(cache=True)
    def double(x):
        return x * 2
    """
)


def test_install_patches_numba():
    numba_cache.install()
    assert numba_caching.IndexDataCacheFile.save is numba_cache._locked_save


def test_concurrent_saves_keep_index_and_data_consistent(tmp_path):
    numba_cache.install()
    stamp = (1.0, 1)
    thread_count = 16
    start = threading.Barrier(thread_count)

    def save(key):
        cache_file = numba_caching.IndexDataCacheFile(str(tmp_path), "func-1.py", stamp)
        start.wait()
        cache_file.save(key, ("data for", key))

    threads = [threading.Thread(target=save, args=(key,)) for key in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    overloads = numba_caching.IndexDataCacheFile(str(tmp_path), "func-1.py", stamp)._load_index()
    assert sorted(overloads) == list(range(thread_count))
    assert len(set(overloads.values())) == thread_count
    for key, data_name in overloads.items():
        with open(tmp_path / data_name, "rb") as f:
            assert pickle.loads(f.read()) == ("data for", key)


def _run(code, tmp_path, guard):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(tmp_path))
    prelude = ""
    if guard:
        prelude = f"import sys; sys.path.insert(0, {REPO_ROOT!r}); from lddecode import numba_cache; numba_cache.install()\n"
    result = subprocess.run(
        [sys.executable, "-c", prelude + code], env=env, cwd=tmp_path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_cache_entry_for_another_signature_is_ignored(tmp_path):
    (tmp_path / "jitmod.py").write_text(JIT_MODULE)
    _run("import jitmod; jitmod.double(3); jitmod.double(2.5)", tmp_path, guard=False)

    # Corrupt the cache the way concurrent unlocked saves can: swap which data file each
    # signature's index entry points at.
    (index_path,) = (tmp_path / "cache").rglob("*.nbi")
    with open(index_path, "rb") as f:
        version = pickle.load(f)
        stamp, overloads = pickle.loads(f.read())
    assert len(overloads) == 2
    keys = list(overloads)
    overloads = {keys[0]: overloads[keys[1]], keys[1]: overloads[keys[0]]}
    with open(index_path, "wb") as f:
        pickle.dump(version, f, protocol=-1)
        f.write(pickle.dumps((stamp, overloads), protocol=-1))

    check = "import jitmod; print(jitmod.double(2.5))"
    if _run(check, tmp_path, guard=False) == "5.0":
        pytest.skip("this numba version already rejects mismatched cache entries")
    assert _run(check, tmp_path, guard=True) == "5.0"
