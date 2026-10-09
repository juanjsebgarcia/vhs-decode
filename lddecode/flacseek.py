"""Sample-accurate seeking in native FLAC files.

RF captures are often stored as FLAC with no SEEKTABLE and, when written by a
streaming encoder, with a total sample count of 0 in STREAMINFO.  ffmpeg can
then only reach sample N by decoding and discarding everything before it.

FLAC frames are independent of each other, and every frame header carries the
position of its first sample (the frame number for fixed-blocksize streams,
the sample number for variable-blocksize streams).  So decoding can start at
the frame that contains sample N and produce exactly the samples a full decode
would, after discarding fewer than one frame of samples.

Two ways to find that frame are provided behind one interface
(``FlacFrameLocator.locate(sample) -> FlacFrame``):

* ``BisectLocator`` bisects byte offsets of the file, parsing frame headers on
  the fly.  It needs no preparation and costs a few dozen small reads.
* ``IndexLocator`` uses a sidecar index (``<capture>.flac.idx``) written by
  ``build_index`` (``vhs-decode-index <capture.flac>``) in one sequential pass.

Either way the frame that decoding will start from is verified before it is
used: its header must parse with the expected position, the following frame's
header must follow it, and the frame's CRC-16 must match.  When anything is
off, ``locate`` returns None and callers fall back to reading and discarding.

Environment variables:

* ``VHSD_FLAC_SEEK``: ``auto`` (default: index if a valid one exists, else
  bisect), ``index`` (index only), ``bisect`` (ignore any index) or ``off``.
* ``VHSD_FLAC_INDEX``: where the index lives.  Either the index file itself or
  a directory holding ``<capture file name>.idx``.  Default: next to the
  capture, ``<capture>.idx``.
"""

import array
import hashlib
import io
import logging
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

FLAC_MAGIC = b"fLaC"

SEEK_MODE_ENV = "VHSD_FLAC_SEEK"
INDEX_PATH_ENV = "VHSD_FLAC_INDEX"

INDEX_SUFFIX = ".idx"
INDEX_MAGIC = b"VHSDFIDX"
INDEX_VERSION = 1
INDEX_ENDIAN_MARK = 0x01020304
# magic, version, endianness mark, flags, reserved, file size, mtime_ns,
# sha256(first MiB), first frame offset, frame count, total samples,
# fixed blocksize (0 for variable-blocksize streams).  104 bytes, so the
# uint64 arrays that follow are 8-byte aligned.
INDEX_HEADER = struct.Struct("<8sIIII Qq32s QQQQ")
INDEX_FLAG_VARIABLE_BLOCKSIZE = 1
INDEX_HASH_BYTES = 1024 * 1024

# Frame header bytes 0-1: 14-bit sync code, a reserved 0 bit, then the
# blocking strategy bit (0 = fixed blocksize, 1 = variable blocksize).
SYNC_FIXED = b"\xff\xf8"
SYNC_VARIABLE = b"\xff\xf9"

# Bisect until the byte range left is this many worst-case frames long, then
# walk forward frame by frame.
WALK_FRAMES = 4


def _make_crc8_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        table.append(c)
    return table


def _make_crc16_table():
    table = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        table.append(c)
    return table


CRC8_TABLE = _make_crc8_table()
CRC16_TABLE = _make_crc16_table()


def crc8(data):
    """FLAC frame header CRC-8 (polynomial 0x07, initial value 0)."""
    c = 0
    table = CRC8_TABLE
    for b in data:
        c = table[c ^ b]
    return c


def crc16(data):
    """FLAC frame CRC-16 (polynomial 0x8005, initial value 0, MSB first)."""
    c = 0
    table = CRC16_TABLE
    for b in data:
        c = ((c << 8) & 0xFFFF) ^ table[(c >> 8) ^ b]
    return c


# Frame header code -> value tables (RFC 9639 section 9.1).
BLOCKSIZE_CODES = {
    1: 192,
    2: 576,
    3: 1152,
    4: 2304,
    5: 4608,
    **{code: 256 << (code - 8) for code in range(8, 16)},
}

SAMPLE_RATE_CODES = {
    1: 88200,
    2: 176400,
    3: 192000,
    4: 8000,
    5: 16000,
    6: 22050,
    7: 24000,
    8: 32000,
    9: 44100,
    10: 48000,
    11: 96000,
}

SAMPLE_SIZE_CODES = {1: 8, 2: 12, 4: 16, 5: 20, 6: 24, 7: 32}


@dataclass(frozen=True)
class StreamInfo:
    min_blocksize: int
    max_blocksize: int
    min_framesize: int
    max_framesize: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int


@dataclass(frozen=True)
class FrameHeader:
    first_sample: int
    blocksize: int
    length: int


@dataclass(frozen=True)
class FlacFrame:
    """A frame located in the file: its byte offset and its first sample."""

    offset: int
    first_sample: int
    blocksize: int


def _read_exact(fd, offset, length):
    """os.pread that keeps reading until `length` bytes or EOF."""
    chunks = []
    while length > 0:
        data = os.pread(fd, length, offset)
        if not data:
            break
        chunks.append(data)
        offset += len(data)
        length -= len(data)
    return b"".join(chunks)


def parse_streaminfo(body):
    min_blocksize, max_blocksize = struct.unpack(">HH", body[:4])
    min_framesize = int.from_bytes(body[4:7], "big")
    max_framesize = int.from_bytes(body[7:10], "big")
    packed = int.from_bytes(body[10:18], "big")
    return StreamInfo(
        min_blocksize=min_blocksize,
        max_blocksize=max_blocksize,
        min_framesize=min_framesize,
        max_framesize=max_framesize,
        sample_rate=packed >> 44,
        channels=((packed >> 41) & 0x7) + 1,
        bits_per_sample=((packed >> 36) & 0x1F) + 1,
        total_samples=packed & ((1 << 36) - 1),
    )


def _decode_coded_number(buf, pos, max_bytes):
    """Decode the UTF-8-like coded frame/sample number at buf[pos].

    Returns (value, length) or None if the bytes are not a valid coding."""
    if pos >= len(buf):
        return None
    first = buf[pos]
    if first < 0x80:
        return first, 1
    if first == 0xFF:
        return None
    if first == 0xFE:
        length, value = 7, 0
    elif first >= 0xFC:
        length, value = 6, first & 0x01
    elif first >= 0xF8:
        length, value = 5, first & 0x03
    elif first >= 0xF0:
        length, value = 4, first & 0x07
    elif first >= 0xE0:
        length, value = 3, first & 0x0F
    elif first >= 0xC0:
        length, value = 2, first & 0x1F
    else:
        # 10xxxxxx is a continuation byte, not a valid first byte
        return None
    if length > max_bytes or pos + length > len(buf):
        return None
    for b in buf[pos + 1 : pos + length]:
        if b & 0xC0 != 0x80:
            return None
        value = (value << 6) | (b & 0x3F)
    return value, length


class FlacStream:
    """Header-level access to a native FLAC file through an OS file descriptor.

    Raises ValueError from the constructor if the file is not a FLAC stream
    this module can seek in (any channel count parses; see `pcm_seekable`
    for what the loaders accept)."""

    def __init__(self, fd):
        self.fd = fd
        st = os.fstat(fd)
        self.file_size = st.st_size
        self.mtime_ns = st.st_mtime_ns

        head = _read_exact(fd, 0, 4)
        if head != FLAC_MAGIC:
            raise ValueError("not a native FLAC stream")

        pos = 4
        self.streaminfo_body = None
        while True:
            block_header = _read_exact(fd, pos, 4)
            if len(block_header) < 4:
                raise ValueError("truncated FLAC metadata")
            is_last = block_header[0] & 0x80
            block_type = block_header[0] & 0x7F
            block_length = int.from_bytes(block_header[1:4], "big")
            if block_type == 0:
                self.streaminfo_body = _read_exact(fd, pos + 4, block_length)
                if len(self.streaminfo_body) != 34:
                    raise ValueError("bad STREAMINFO block")
            elif block_type == 127:
                raise ValueError("invalid FLAC metadata block type")
            pos += 4 + block_length
            if is_last:
                break
        if self.streaminfo_body is None:
            raise ValueError("FLAC stream without STREAMINFO")

        self.info = parse_streaminfo(self.streaminfo_body)
        self.first_frame_offset = pos

        first = _read_exact(fd, pos, 2)
        if first == SYNC_FIXED:
            self.variable_blocksize = False
            self.sync = SYNC_FIXED
            self.sync_pattern = re.compile(re.escape(SYNC_FIXED))
        elif first == SYNC_VARIABLE:
            self.variable_blocksize = True
            self.sync = SYNC_VARIABLE
            self.sync_pattern = re.compile(re.escape(SYNC_VARIABLE))
        else:
            raise ValueError("no frame after the FLAC metadata")

        # Parse the first frame header without knowing the fixed blocksize yet.
        self.fixed_blocksize = None
        buf = _read_exact(fd, pos, 16)
        header = self._parse_header_fields(buf, 0)
        if header is None:
            raise ValueError("first FLAC frame header does not parse")
        number, blocksize, _ = header
        if number != 0:
            raise ValueError("first FLAC frame does not start at sample 0")
        if not self.variable_blocksize:
            # Fixed-blocksize streams number frames; frame n starts at
            # n * blocksize.  Only the last frame may be shorter.
            if blocksize != self.info.max_blocksize and blocksize != self.info.min_blocksize:
                raise ValueError("first frame blocksize disagrees with STREAMINFO")
            self.fixed_blocksize = blocksize

        max_blocksize = self.info.max_blocksize or 65535
        bits = self.info.bits_per_sample
        # Worst case frame: header + one verbatim subframe per channel (a side
        # channel has one extra bit per sample) + CRC-16.
        self.max_frame_bytes = (
            18 + self.info.channels * (2 + (max_blocksize * (bits + 1) + 7) // 8) + 2
        )

    @property
    def sample_rate(self):
        return self.info.sample_rate

    def pcm_seekable(self):
        """Whether one decoded 16-bit PCM sample is one stream sample, which
        the loaders assume: mono, at most 16 bits (wider samples would need
        a format conversion)."""
        return self.info.channels == 1 and self.info.bits_per_sample <= 16

    def decoder_header(self):
        """`fLaC` + STREAMINFO as the only (last) metadata block.

        Prepended to the frames from some offset onwards, this makes a stream a
        FLAC decoder accepts.  The total sample count and MD5 are cleared as
        they describe the whole file, not the stream starting mid-way."""
        body = bytearray(self.streaminfo_body)
        packed = int.from_bytes(body[10:18], "big") & ~((1 << 36) - 1)
        body[10:18] = packed.to_bytes(8, "big")
        body[18:34] = bytes(16)
        return FLAC_MAGIC + bytes([0x80, 0, 0, 34]) + bytes(body)

    def read(self, offset, length):
        return _read_exact(self.fd, offset, length)

    def _parse_header_fields(self, buf, pos):
        """Parse and validate the frame header at buf[pos].

        Returns (coded number, blocksize, header length) or None."""
        if len(buf) - pos < 6:
            return None
        if buf[pos : pos + 2] != self.sync:
            return None
        b2 = buf[pos + 2]
        b3 = buf[pos + 3]
        blocksize_code = b2 >> 4
        rate_code = b2 & 0x0F
        channel_code = b3 >> 4
        size_code = (b3 >> 1) & 0x07
        if blocksize_code == 0 or rate_code == 15 or channel_code > 10 or size_code == 3:
            return None
        if b3 & 0x01:
            return None

        info = self.info
        channels = channel_code + 1 if channel_code < 8 else 2
        if channels != info.channels:
            return None
        if size_code != 0 and SAMPLE_SIZE_CODES[size_code] != info.bits_per_sample:
            return None
        if rate_code in SAMPLE_RATE_CODES and SAMPLE_RATE_CODES[rate_code] != info.sample_rate:
            return None

        coded = _decode_coded_number(buf, pos + 4, 7 if self.variable_blocksize else 6)
        if coded is None:
            return None
        number, coded_length = coded
        p = pos + 4 + coded_length

        if blocksize_code == 6:
            if p + 1 > len(buf):
                return None
            blocksize = buf[p] + 1
            p += 1
        elif blocksize_code == 7:
            if p + 2 > len(buf):
                return None
            blocksize = int.from_bytes(buf[p : p + 2], "big") + 1
            p += 2
        else:
            blocksize = BLOCKSIZE_CODES[blocksize_code]

        if rate_code == 12:
            if p + 1 > len(buf):
                return None
            rate = buf[p] * 1000
            p += 1
        elif rate_code in (13, 14):
            if p + 2 > len(buf):
                return None
            rate = int.from_bytes(buf[p : p + 2], "big") * (10 if rate_code == 14 else 1)
            p += 2
        else:
            rate = None
        if rate is not None and rate != info.sample_rate:
            return None

        if p + 1 > len(buf):
            return None
        if crc8(buf[pos:p]) != buf[p]:
            return None
        p += 1

        if info.max_blocksize and blocksize > info.max_blocksize:
            return None
        if self.fixed_blocksize is not None and blocksize > self.fixed_blocksize:
            return None

        return number, blocksize, p - pos

    def parse_header(self, buf, pos):
        """Parse the frame header at buf[pos] -> FrameHeader or None."""
        fields = self._parse_header_fields(buf, pos)
        if fields is None:
            return None
        number, blocksize, length = fields
        if self.variable_blocksize:
            first_sample = number
        else:
            first_sample = number * self.fixed_blocksize
        return FrameHeader(first_sample, blocksize, length)


class _Window:
    """A growable read window over the file, for scanning forward."""

    def __init__(self, stream, start, chunk):
        self.stream = stream
        self.start = start
        self.chunk = chunk
        self.buf = stream.read(start, chunk)

    @property
    def end(self):
        return self.start + len(self.buf)

    def at_eof(self):
        return self.end >= self.stream.file_size

    def ensure(self, end):
        """Make the window extend to at least `end` (or to EOF)."""
        while self.end < end and not self.at_eof():
            more = self.stream.read(self.end, max(self.chunk, end - self.end))
            if not more:
                break
            self.buf += more


class FlacFrameLocator:
    """Common interface: find the verified frame that contains a sample."""

    method = "none"

    def __init__(self, stream):
        self.stream = stream

    def locate(self, sample):
        """Return the FlacFrame containing `sample`, or None if unsure.

        For a sample past the end of the stream the last frame is returned, so
        that decoding from it reaches EOF exactly as a full decode would."""
        raise NotImplementedError

    # Shared helpers

    def _next_header(self, window, pos, limit, expect=None):
        """Find the next valid frame header at or after absolute offset `pos`
        and before `limit`.  With `expect`, only accept a header whose first
        sample is `expect`.  Returns (offset, FrameHeader) or None."""
        stream = self.stream
        sync = stream.sync_pattern
        while True:
            window.ensure(min(limit, pos + window.chunk) + 32)
            match = sync.search(window.buf, pos - window.start, max(0, limit + 1 - window.start))
            rel = match.start() if match else -1
            if rel < 0:
                if window.end >= limit or window.at_eof():
                    return None
                pos = window.end - 1
                continue
            offset = window.start + rel
            window.ensure(offset + 32)
            header = stream.parse_header(window.buf, rel)
            if header is not None and (expect is None or header.first_sample == expect):
                return offset, header
            pos = offset + 1

    def _frame_end(self, window, offset, header):
        """Find where the frame at `offset` ends: the offset of the next frame
        header in sequence (or EOF for the last frame), confirmed by the
        frame's CRC-16.  Returns the end offset or None."""
        stream = self.stream
        expect = header.first_sample + header.blocksize
        limit = min(stream.file_size, offset + stream.max_frame_bytes + 1)
        pos = offset + header.length
        while True:
            found = self._next_header(window, pos, limit, expect)
            end = found[0] if found is not None else None
            if end is None:
                # Possibly the last frame of the file.
                if stream.file_size - offset > stream.max_frame_bytes:
                    return None
                end = stream.file_size
            window.ensure(end)
            frame = window.buf[offset - window.start : end - window.start]
            if len(frame) >= header.length + 2 and crc16(frame[:-2]) == int.from_bytes(
                frame[-2:], "big"
            ):
                return end
            if found is None:
                return None
            pos = end + 1

    def verify_frame(self, offset, first_sample):
        """Check there is a genuine frame at `offset` starting at `first_sample`.

        Returns (FlacFrame, end offset) or None."""
        stream = self.stream
        window = _Window(stream, offset, stream.max_frame_bytes + 64)
        header = stream.parse_header(window.buf, 0)
        if header is None or header.first_sample != first_sample:
            return None
        end = self._frame_end(window, offset, header)
        if end is None:
            return None
        return FlacFrame(offset, header.first_sample, header.blocksize), end


class BisectLocator(FlacFrameLocator):
    """Locate frames by bisecting byte offsets and parsing headers on the fly."""

    method = "bisect"

    def __init__(self, stream):
        super().__init__(stream)
        self.probes = 0

    def _frame_at_or_after(self, pos, limit):
        """First frame starting at or after `pos` (and before `limit`) whose
        header is followed, within one maximum frame size, by the header of
        the next frame in sequence.  Returns (FlacFrame, next offset) or None.

        Data inside a frame can contain a sync code with a valid CRC-8 by
        chance; requiring the next frame header in sequence rules that out."""
        stream = self.stream
        self.probes += 1
        window = _Window(stream, pos, 2 * stream.max_frame_bytes + 64)
        while True:
            found = self._next_header(window, pos, limit)
            if found is None:
                return None
            offset, header = found
            expect = header.first_sample + header.blocksize
            next_limit = min(stream.file_size, offset + stream.max_frame_bytes + 1)
            following = self._next_header(window, offset + header.length, next_limit, expect)
            if following is not None:
                return FlacFrame(offset, header.first_sample, header.blocksize), following[0]
            if stream.file_size - offset <= stream.max_frame_bytes:
                # The last frame has no successor; accept it on its CRC-16.
                end = self._frame_end(window, offset, header)
                if end == stream.file_size:
                    return FlacFrame(offset, header.first_sample, header.blocksize), end
            pos = offset + 1

    def locate(self, sample):
        stream = self.stream
        if sample < 0:
            return None
        self.probes = 0

        found = self._frame_at_or_after(stream.first_frame_offset, stream.file_size)
        if found is None or found[0].offset != stream.first_frame_offset:
            return None
        lo, lo_next = found
        hi = stream.file_size

        # Bisect on byte offset.  Invariant: lo starts at or before `sample`;
        # every frame starting at or after `hi` starts after `sample`.
        walk_bytes = WALK_FRAMES * stream.max_frame_bytes
        while lo.first_sample + lo.blocksize <= sample and hi - lo.offset > walk_bytes:
            mid = (lo.offset + hi) // 2
            found = self._frame_at_or_after(mid, hi)
            if found is None:
                hi = mid
                continue
            frame, frame_next = found
            if frame.first_sample <= sample:
                if frame.first_sample < lo.first_sample:
                    # Positions must increase with offset.
                    return None
                lo, lo_next = frame, frame_next
            else:
                hi = mid

        # Walk forward frame by frame.
        window = _Window(stream, lo_next, 2 * stream.max_frame_bytes + 64)
        while lo.first_sample + lo.blocksize <= sample and lo_next < stream.file_size:
            expect = lo.first_sample + lo.blocksize
            window_hdr = self._next_header(
                window, lo_next, min(stream.file_size, lo_next + 32), expect
            )
            if window_hdr is None or window_hdr[0] != lo_next:
                return None
            _, header = window_hdr
            next_found = self._next_header(
                window,
                lo_next + header.length,
                min(stream.file_size, lo_next + stream.max_frame_bytes + 1),
                expect + header.blocksize,
            )
            lo = FlacFrame(lo_next, header.first_sample, header.blocksize)
            lo_next = next_found[0] if next_found is not None else stream.file_size

        verified = self.verify_frame(lo.offset, lo.first_sample)
        if verified is None:
            logger.debug("FLAC bisect: frame at %d failed verification", lo.offset)
            return None
        frame, _ = verified
        if frame.first_sample > sample:
            return None
        return frame


def index_path_for(path):
    """Where the sidecar index for `path` lives (see INDEX_PATH_ENV)."""
    override = os.environ.get(INDEX_PATH_ENV)
    if override:
        if os.path.isdir(override):
            return os.path.join(override, os.path.basename(path) + INDEX_SUFFIX)
        return override
    return path + INDEX_SUFFIX


def _first_mib_hash(fd, file_size):
    return hashlib.sha256(_read_exact(fd, 0, min(file_size, INDEX_HASH_BYTES))).digest()


def scan_frames(stream, progress=None, chunk=32 * 1024 * 1024):
    """One sequential pass over the file collecting every frame's offset and
    first sample.  Only frame headers are parsed; a header is accepted when it
    is the next one in sequence, so a sync code inside frame data can only be
    mistaken for a header if its CRC-8 passes and it carries exactly the next
    position.  (A frame is also CRC-16 checked whenever it is used to seek.)

    Returns (offsets, first_samples, total_samples) as numpy uint64 arrays and
    an int.  Raises ValueError if the frames are not one contiguous sequence."""
    fd = stream.fd
    sync = stream.sync_pattern
    file_size = stream.file_size

    offsets = array.array("Q")
    first_samples = array.array("Q")

    buf = b""
    buf_start = stream.first_frame_offset
    expect = 0
    pos = stream.first_frame_offset
    last_blocksize = 0

    while True:
        end = buf_start + len(buf)
        # (A compiled pattern searches several times faster than bytes.find.)
        match = sync.search(buf, pos - buf_start) if pos < end else None
        rel = match.start() if match else -1
        if (rel < 0 or rel + 32 > len(buf)) and end < file_size:
            # Refill, keeping a possible header (or half a sync code).
            keep_from = buf_start + rel if rel >= 0 else max(pos, end - 1, buf_start)
            buf = buf[keep_from - buf_start :] + _read_exact(fd, end, chunk)
            buf_start = keep_from
            pos = max(pos, keep_from)
            if progress is not None:
                progress(end, file_size)
            continue
        if rel < 0:
            break
        header = stream.parse_header(buf, rel)
        offset = buf_start + rel
        if header is not None and header.first_sample == expect:
            offsets.append(offset)
            first_samples.append(expect)
            expect += header.blocksize
            last_blocksize = header.blocksize
            pos = offset + header.length
        else:
            pos = offset + 1

    if not offsets or offsets[0] != stream.first_frame_offset:
        raise ValueError("FLAC frames do not start after the metadata")

    offsets = np.frombuffer(offsets, dtype=np.uint64)
    first_samples = np.frombuffer(first_samples, dtype=np.uint64)

    sizes = np.diff(np.append(offsets, np.uint64(file_size)))
    if np.any(sizes > stream.max_frame_bytes):
        raise ValueError("gap between FLAC frames larger than a frame; damaged file?")

    total = int(first_samples[-1]) + last_blocksize
    return offsets, first_samples, total


def build_index(path, index_path=None, progress=None):
    """Scan `path` and write its sidecar index.  Returns the index path.

    Layout (all little-endian, see INDEX_HEADER): a fixed 104-byte header, then
    uint64 byte offsets of every frame, then (variable-blocksize streams only)
    uint64 first-sample numbers of every frame.  The arrays are flat and
    8-byte aligned so a reader can np.memmap them without parsing."""
    if index_path is None:
        index_path = index_path_for(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        stream = FlacStream(fd)
        offsets, first_samples, total = scan_frames(stream, progress)
        file_size = stream.file_size
        mtime_ns = os.fstat(fd).st_mtime_ns
        digest = _first_mib_hash(fd, file_size)
    finally:
        os.close(fd)

    flags = 0
    if stream.variable_blocksize:
        flags |= INDEX_FLAG_VARIABLE_BLOCKSIZE

    header = INDEX_HEADER.pack(
        INDEX_MAGIC,
        INDEX_VERSION,
        INDEX_ENDIAN_MARK,
        flags,
        0,
        file_size,
        mtime_ns,
        digest,
        stream.first_frame_offset,
        len(offsets),
        total,
        stream.fixed_blocksize or 0,
    )
    tmp = index_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(offsets.astype("<u8").tobytes())
        if stream.variable_blocksize:
            f.write(first_samples.astype("<u8").tobytes())
    os.replace(tmp, index_path)
    return index_path


class StaleIndexError(ValueError):
    pass


class IndexLocator(FlacFrameLocator):
    """Locate frames with a sidecar index written by build_index.

    Opening reads the 104-byte header and hashes the capture's first MiB to
    check the index is current; a lookup reads 16 bytes (fixed blocksize) or
    O(log n) 8-byte entries (variable blocksize)."""

    method = "index"

    def __init__(self, stream, index_path):
        super().__init__(stream)
        with open(index_path, "rb") as f:
            head = f.read(INDEX_HEADER.size)
            index_size = os.fstat(f.fileno()).st_size
        if len(head) < INDEX_HEADER.size:
            raise ValueError("index file too short")
        (
            magic,
            version,
            endian_mark,
            flags,
            _reserved,
            file_size,
            mtime_ns,
            digest,
            first_frame_offset,
            nframes,
            total,
            fixed,
        ) = INDEX_HEADER.unpack(head)
        if magic != INDEX_MAGIC or version != INDEX_VERSION or endian_mark != INDEX_ENDIAN_MARK:
            raise ValueError("not a FLAC seek index (or an unsupported version)")
        variable = bool(flags & INDEX_FLAG_VARIABLE_BLOCKSIZE)
        arrays = 2 if variable else 1
        if nframes < 1 or index_size != INDEX_HEADER.size + 8 * nframes * arrays:
            raise ValueError("index file is damaged")
        if (
            file_size != stream.file_size
            or mtime_ns != stream.mtime_ns
            or first_frame_offset != stream.first_frame_offset
            or variable != stream.variable_blocksize
            or fixed != (stream.fixed_blocksize or 0)
            or digest != _first_mib_hash(stream.fd, stream.file_size)
        ):
            raise StaleIndexError("index does not match the file (stale index?)")

        self.index_path = index_path
        self.variable = variable
        self.fixed = fixed
        self.nframes = nframes
        self.total_samples = total

    # The arrays are read with pread, a few bytes per lookup, so nothing is
    # loaded or parsed up front.  (np.memmap works too, but mapping costs a
    # few ms on macOS, more than a whole lookup.)

    def _read_entries(self, fd, array, first, count):
        base = INDEX_HEADER.size + 8 * (array * self.nframes + first)
        data = _read_exact(fd, base, 8 * count)
        if len(data) != 8 * count:
            raise ValueError("index file is truncated")
        return struct.unpack("<%dQ" % count, data)

    def _find_frame(self, fd, sample):
        """Index of the frame containing `sample` (the last frame if past the end)."""
        if not self.variable:
            return min(sample // self.fixed, self.nframes - 1)
        # Binary search the first-sample array for the last entry <= sample.
        lo, hi = 0, self.nframes
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._read_entries(fd, 1, mid, 1)[0] <= sample:
                lo = mid
            else:
                hi = mid
        return lo

    def lookup(self, sample):
        """(frame index, offset, first sample, next frame offset) for `sample`,
        straight from the index (not verified against the file)."""
        fd = os.open(self.index_path, os.O_RDONLY)
        try:
            i = self._find_frame(fd, sample)
            count = 2 if i + 1 < self.nframes else 1
            offsets = self._read_entries(fd, 0, i, count)
            if self.variable:
                first_sample = self._read_entries(fd, 1, i, 1)[0]
            else:
                first_sample = i * self.fixed
        finally:
            os.close(fd)
        next_offset = offsets[1] if count == 2 else self.stream.file_size
        return i, offsets[0], first_sample, next_offset

    def locate(self, sample):
        if sample < 0:
            return None
        i, offset, first_sample, expected_end = self.lookup(sample)
        verified = self.verify_frame(offset, first_sample)
        if verified is not None and verified[1] == expected_end:
            return verified[0]
        # A damaged frame (e.g. a truncated last frame) or a bad index entry:
        # let bisection, which verifies on its own, have a go.
        logger.debug("FLAC seek index entry %d does not verify; bisecting instead", i)
        return BisectLocator(self.stream).locate(sample)


def seek_mode():
    mode = os.environ.get(SEEK_MODE_ENV, "auto").strip().lower()
    if mode not in ("auto", "index", "bisect", "off"):
        logger.warning("%s=%s not understood, using auto", SEEK_MODE_ENV, mode)
        mode = "auto"
    return mode


def open_locator(fd, path=None):
    """Return a FlacFrameLocator for the FLAC file open as `fd`, or None.

    `path` (the file's name) is used to find a sidecar index."""
    mode = seek_mode()
    if mode == "off":
        return None
    try:
        stream = FlacStream(fd)
    except (ValueError, OSError) as e:
        logger.debug("FLAC seek unavailable: %s", e)
        return None

    if path is not None and mode in ("auto", "index"):
        index_path = index_path_for(path)
        if os.path.exists(index_path):
            try:
                return IndexLocator(stream, index_path)
            except StaleIndexError as e:
                logger.warning("Ignoring FLAC seek index %s: %s", index_path, e)
            except (ValueError, OSError) as e:
                logger.warning("Cannot use FLAC seek index %s: %s", index_path, e)
    if mode == "index":
        return None
    return BisectLocator(stream)


class FramesFromOffset(io.RawIOBase):
    """A readable FLAC stream: `fLaC` + STREAMINFO, then the file's frames
    from `offset` to EOF, read with pread so the file position is untouched.

    A FLAC decoder fed this decodes exactly the samples that a full decode
    produces from frame `offset` onwards (FLAC frames are independent)."""

    def __init__(self, stream, offset):
        super().__init__()
        self.fd = stream.fd
        self.header = stream.decoder_header()
        self.offset = offset
        self.header_pos = 0

    def readable(self):
        return True

    def readinto(self, b):
        n = len(b)
        if n == 0:
            return 0
        if self.header_pos < len(self.header):
            data = self.header[self.header_pos : self.header_pos + n]
            self.header_pos += len(data)
        else:
            data = os.pread(self.fd, n, self.offset)
            self.offset += len(data)
        b[: len(data)] = data
        return len(data)


def feed_pipe(source, pipe, stop_event, chunk=1024 * 1024):
    """Copy `source` (a FramesFromOffset) into `pipe` until EOF, the reader
    goes away or `stop_event` is set.  Runs on a background thread."""
    try:
        while not stop_event.is_set():
            data = source.read(chunk)
            if not data:
                break
            pipe.write(data)
    except (BrokenPipeError, OSError, ValueError):
        # The decoder was stopped (or the input closed) before EOF.
        pass
    finally:
        try:
            pipe.close()
        except (BrokenPipeError, OSError):
            pass
