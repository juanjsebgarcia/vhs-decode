"""Seeking in native FLAC files (lddecode/flacseek.py) and the loaders using it.

Every seek-started read must return exactly the samples a full sequential
decode returns at that position.  The test streams are lossless encodes of
known samples, so the ground truth is the input itself.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from lddecode import flacseek
from lddecode.utils import LoadFFmpeg, LoadLDF

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_samples(count, seed=1):
    """Noisy carrier-like int16 signal: compresses, but not to nothing."""
    rng = np.random.default_rng(seed)
    t = np.arange(count)
    signal = 9000 * np.sin(t * 0.37) + rng.normal(0, 2500, count)
    return np.clip(signal, -32768, 32767).astype("<i2")


def encode_flac(samples, path, frame_size, sample_rate=40000):
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
            "-c:a", "flac", "-frame_size", str(frame_size), str(path),
        ],
        input=samples.tobytes(),
        check=True,
    )
    return path


def decode_all(path):
    with open(path, "rb") as f:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-i", "-",
             "-c:a", "pcm_s16le", "-f", "s16le", "-"],
            stdin=f, capture_output=True, check=True,
        ).stdout
    return np.frombuffer(out, "<i2")


def frame_table(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        stream = flacseek.FlacStream(fd)
        offsets, first_samples, total = flacseek.scan_frames(stream)
    finally:
        os.close(fd)
    return offsets.astype(np.int64), first_samples.astype(np.int64), total


def encode_coded_number(value):
    if value < 0x80:
        return bytes([value])
    for length, bits in ((2, 11), (3, 16), (4, 21), (5, 26), (6, 31), (7, 36)):
        if value < (1 << bits):
            break
    out = []
    for _ in range(length - 1):
        out.append(0x80 | (value & 0x3F))
        value >>= 6
    lead = (0xFF00 >> length) & 0xFF
    out.append(lead | value)
    return bytes(reversed(out))


def make_variable_blocksize(samples, path, tmp_path):
    """Splice frames of two fixed-blocksize encodes (1152 and 4096 samples)
    into one variable-blocksize stream, numbering frames by sample."""
    period = 36864  # lcm(1152, 4096)
    a = encode_flac(samples, tmp_path / "a.flac", 1152)
    b = encode_flac(samples, tmp_path / "b.flac", 4096)
    tables = {}
    for name, src in (("a", a), ("b", b)):
        data = src.read_bytes()
        offsets, first_samples, _ = frame_table(src)
        ends = list(offsets[1:]) + [len(data)]
        tables[name] = (data, dict(zip(first_samples.tolist(), zip(offsets.tolist(), ends))))

    out = bytearray()
    header_a = tables["a"][0][: int(frame_table(a)[0][0])]
    # STREAMINFO: min blocksize 1152, max 4096, total samples unchanged.
    streaminfo = bytearray(header_a[8:42])
    streaminfo[0:4] = (1152).to_bytes(2, "big") + (4096).to_bytes(2, "big")
    streaminfo[4:10] = bytes(6)  # frame sizes unknown
    out += b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo

    sample = 0
    while sample < len(samples):
        name = "a" if (sample // period) % 2 == 0 else "b"
        data, frames = tables[name]
        offset, end = frames[sample]
        frame = bytearray(data[offset:end])
        # Old header: sync(2) codes(2) coded frame number, then optional
        # blocksize/rate bytes, then CRC-8.
        coded_len = flacseek._decode_coded_number(frame, 4, 6)[1]
        rest_start = 4 + coded_len
        bs_code, rate_code = frame[2] >> 4, frame[2] & 0x0F
        extra = (1 if bs_code == 6 else 2 if bs_code == 7 else 0) + (
            1 if rate_code == 12 else 2 if rate_code in (13, 14) else 0
        )
        header = bytes([0xFF, 0xF9]) + bytes(frame[2:4]) + encode_coded_number(sample)
        header += bytes(frame[rest_start : rest_start + extra])
        header += bytes([flacseek.crc8(header)])
        body = bytes(frame[rest_start + extra + 1 : -2])
        new_frame = header + body
        new_frame += flacseek.crc16(new_frame).to_bytes(2, "big")
        out += new_frame
        sample += 1152 if name == "a" else 4096
    path.write_bytes(bytes(out))
    return path


@pytest.fixture(scope="module")
def flac_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("flacseek")
    samples = make_samples(3 * 36864 * 4 + 1234)  # last frame is short
    files = {
        "fixed4096": encode_flac(samples, tmp / "fixed4096.flac", 4096),
        "fixed1152": encode_flac(samples, tmp / "fixed1152.flac", 1152),
        "variable": make_variable_blocksize(samples, tmp / "variable.flac", tmp),
    }
    return samples, files


def targets_for(path, total):
    offsets, first_samples, _ = frame_table(path)
    rng = np.random.default_rng(5)
    picks = {0, 1, total - 1, total}
    for i in (1, 2, len(first_samples) // 2, len(first_samples) - 1):
        s = int(first_samples[i])
        picks |= {s - 1, s, s + 1}
    picks |= set(rng.integers(0, total, 12).tolist())
    return sorted(p for p in picks if 0 <= p <= total)


@pytest.mark.parametrize("name", ["fixed4096", "fixed1152", "variable"])
def test_full_decode_is_lossless(flac_files, name):
    samples, files = flac_files
    assert np.array_equal(decode_all(files[name]), samples)


@pytest.mark.parametrize("name", ["fixed4096", "fixed1152", "variable"])
def test_scan_matches_pyav_packets(flac_files, name):
    av = pytest.importorskip("av")
    _, files = flac_files
    offsets, first_samples, total = frame_table(files[name])
    pos, pts = [], []
    with av.open(str(files[name])) as container:
        for packet in container.demux(container.streams.audio[0]):
            if packet.size:
                pos.append(packet.pos)
                pts.append(packet.pts)
    assert offsets.tolist() == pos
    assert first_samples.tolist() == pts
    assert total == len(flac_files[0])


def test_scan_with_small_chunks_matches(flac_files):
    _, files = flac_files
    fd = os.open(files["fixed1152"], os.O_RDONLY)
    try:
        stream = flacseek.FlacStream(fd)
        whole = flacseek.scan_frames(stream)
        chunked = flacseek.scan_frames(stream, chunk=4099)
    finally:
        os.close(fd)
    assert np.array_equal(whole[0], chunked[0])
    assert np.array_equal(whole[1], chunked[1])
    assert whole[2] == chunked[2]


@pytest.mark.parametrize("name", ["fixed4096", "fixed1152", "variable"])
def test_locators_find_the_frame_containing_the_sample(flac_files, name, tmp_path):
    samples, files = flac_files
    path = files[name]
    offsets, first_samples, total = frame_table(path)
    index_path = flacseek.build_index(str(path), str(tmp_path / "x.idx"))

    fd = os.open(path, os.O_RDONLY)
    try:
        stream = flacseek.FlacStream(fd)
        locators = [
            flacseek.BisectLocator(stream),
            flacseek.IndexLocator(stream, index_path),
        ]
        for sample in targets_for(path, total) + [total + 10**6]:
            i = max(0, int(np.searchsorted(first_samples, sample, side="right")) - 1)
            for locator in locators:
                frame = locator.locate(sample)
                assert frame is not None, (locator.method, sample)
                assert (frame.offset, frame.first_sample) == (offsets[i], first_samples[i])
    finally:
        os.close(fd)


def test_stale_index_is_not_used(flac_files, tmp_path, monkeypatch):
    _, files = flac_files
    path = tmp_path / "copy.flac"
    shutil.copyfile(files["fixed4096"], path)
    index_path = flacseek.build_index(str(path))
    assert index_path == str(path) + ".idx"

    fd = os.open(path, os.O_RDONLY)
    try:
        assert isinstance(flacseek.open_locator(fd, str(path)), flacseek.IndexLocator)
    finally:
        os.close(fd)

    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(flacseek.StaleIndexError):
            flacseek.IndexLocator(flacseek.FlacStream(fd), index_path)
        assert isinstance(flacseek.open_locator(fd, str(path)), flacseek.BisectLocator)
        monkeypatch.setenv(flacseek.SEEK_MODE_ENV, "index")
        assert flacseek.open_locator(fd, str(path)) is None
        monkeypatch.setenv(flacseek.SEEK_MODE_ENV, "off")
        assert flacseek.open_locator(fd, str(path)) is None
    finally:
        os.close(fd)


def test_index_location_override(tmp_path, monkeypatch):
    monkeypatch.setenv(flacseek.INDEX_PATH_ENV, str(tmp_path))
    assert flacseek.index_path_for("/some/where/cap.flac") == str(tmp_path / "cap.flac.idx")
    monkeypatch.setenv(flacseek.INDEX_PATH_ENV, str(tmp_path / "named.idx"))
    assert flacseek.index_path_for("/some/where/cap.flac") == str(tmp_path / "named.idx")


def test_index_layout_is_flat_little_endian(flac_files, tmp_path):
    _, files = flac_files
    offsets, first_samples, total = frame_table(files["variable"])
    index_path = flacseek.build_index(str(files["variable"]), str(tmp_path / "v.idx"))
    raw = open(index_path, "rb").read()
    header = flacseek.INDEX_HEADER.unpack_from(raw)
    assert header[0] == flacseek.INDEX_MAGIC
    nframes = header[9]
    assert nframes == len(offsets) and header[10] == total
    base = flacseek.INDEX_HEADER.size
    assert base % 8 == 0
    assert np.array_equal(np.frombuffer(raw, "<u8", nframes, base), offsets)
    assert np.array_equal(np.frombuffer(raw, "<u8", nframes, base + 8 * nframes), first_samples)


def read_with(loader_factory, path, sample, length):
    with open(path, "rb") as f:
        loader = loader_factory(path)
        try:
            return loader.read(f, sample, length)
        finally:
            loader._close()


LOADERS = {
    "ffmpeg": lambda path: LoadFFmpeg(),
    "pyav": lambda path: LoadLDF(str(path)),
}


@pytest.mark.parametrize("mode", ["bisect", "index"])
@pytest.mark.parametrize("loader", ["ffmpeg", "pyav"])
@pytest.mark.parametrize("name", ["fixed4096", "variable"])
def test_seek_started_reads_equal_sequential_decode(
    flac_files, name, loader, mode, tmp_path, monkeypatch
):
    if loader == "pyav":
        pytest.importorskip("av")
    samples, files = flac_files
    path = files[name]
    total = len(samples)
    monkeypatch.setenv(flacseek.SEEK_MODE_ENV, mode)
    if mode == "index":
        monkeypatch.setenv(flacseek.INDEX_PATH_ENV, str(tmp_path / "s.idx"))
        flacseek.build_index(str(path))
    length = 5000
    for sample in targets_for(path, total):
        got = read_with(LOADERS[loader], path, sample, length)
        if sample + length > total:
            assert got is None
        else:
            assert np.array_equal(got, samples[sample : sample + length]), sample


@pytest.mark.parametrize("loader", ["ffmpeg", "pyav"])
def test_jumps_within_one_loader(flac_files, loader, monkeypatch):
    if loader == "pyav":
        pytest.importorskip("av")
    samples, files = flac_files
    path = files["fixed1152"]
    monkeypatch.setenv(flacseek.SEEK_MODE_ENV, "bisect")
    with open(path, "rb") as f:
        ld = LOADERS[loader](path)
        ld.seek_threshold = 20000
        ld.rewind_size = 4096
        try:
            # forward small, forward far, backward within and beyond the rewind buffer
            for sample in (0, 3000, 9000, 200000, 199000, 150000, 7, 400000, 400001):
                got = ld.read(f, sample, 3000)
                assert np.array_equal(got, samples[sample : sample + 3000]), sample
        finally:
            ld._close()


def test_resampling_loader_does_not_seek(flac_files):
    _, files = flac_files
    loader = LoadFFmpeg(output_args=["-filter:a", "asetrate=28636000.0,aresample=40000000.0"])
    with open(files["fixed4096"], "rb") as f:
        assert loader._get_locator(f) is None


@pytest.mark.parametrize("sample", [1, 4095, 4096, 200001, 423602])
def test_loadldf_container_seek_is_exact(flac_files, sample, monkeypatch):
    """Without a frame locator LoadLDF seeks the container in the stream's
    time base and positions itself by the first decoded frame's pts.  (It used
    to scale the pts by 1000, returning data from the wrong place once the
    seek landed past frame 0; that needs > 40M-sample files to show.)"""
    pytest.importorskip("av")
    samples, files = flac_files
    monkeypatch.setenv(flacseek.SEEK_MODE_ENV, "off")
    got = read_with(LOADERS["pyav"], files["fixed4096"], sample, 5000)
    assert np.array_equal(got, samples[sample : sample + 5000])


def test_loadldf_odd_blocksize(tmp_path):
    """PyAV pads the plane of an odd-sized frame; LoadLDF must not return the
    padding as a sample (it did, one stray sample per 65535-sample frame)."""
    pytest.importorskip("av")
    samples = make_samples(5 * 4095 + 77, seed=3)
    path = encode_flac(samples, tmp_path / "odd.flac", 4095)
    assert np.array_equal(read_with(LOADERS["pyav"], path, 0, len(samples)), samples)
    assert np.array_equal(read_with(LOADERS["pyav"], path, 9000, 5000), samples[9000:14000])


@pytest.mark.parametrize("loader", ["ffmpeg", "pyav"])
def test_truncated_last_frame(flac_files, loader, tmp_path, monkeypatch):
    """A capture cut off mid-frame: seeks still match a full decode, and a seek
    into the damaged frame falls back to reading from the start."""
    if loader == "pyav":
        pytest.importorskip("av")
    _, files = flac_files
    path = tmp_path / "cut.flac"
    path.write_bytes(files["fixed4096"].read_bytes()[:-500])
    full = decode_all(path)
    offsets, first_samples, _ = frame_table(path)
    flacseek.build_index(str(path))
    for mode in ("bisect", "index"):
        monkeypatch.setenv(flacseek.SEEK_MODE_ENV, mode)
        fd = os.open(path, os.O_RDONLY)
        try:
            locator = flacseek.open_locator(fd, str(path))
            assert locator.locate(int(first_samples[-1]) + 5) is None
            assert locator.locate(int(first_samples[-2]) + 5).offset == offsets[-2]
        finally:
            os.close(fd)
        # ffmpeg drops the damaged frame, so a full decode ends before it.
        assert len(full) == first_samples[-1]
        sample = int(first_samples[-2]) + 5
        got = read_with(LOADERS[loader], path, sample, 1000)
        assert np.array_equal(got, full[sample : sample + 1000]), mode
        assert read_with(LOADERS[loader], path, int(first_samples[-1]) + 5, 1000) is None
