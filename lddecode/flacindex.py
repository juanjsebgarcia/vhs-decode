"""Build a seek index for a native FLAC RF capture.

    vhs-decode-index capture.flac            # writes capture.flac.idx
    vhs-decode-index capture.flac -o /elsewhere/capture.flac.idx

vhs-decode (and ld-decode) use the index automatically when it sits next to
the capture, or when VHSD_FLAC_INDEX names it (or the directory holding it).
Without an index they bisect the file instead, which needs no preparation and
is only a few milliseconds slower per seek; see lddecode/flacseek.py.
"""

import argparse
import os
import sys
import time

from lddecode import flacseek


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Write a sidecar seek index for a native FLAC capture"
    )
    parser.add_argument("infile", help="FLAC capture to index")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "index file to write (default: $%s if set, else <infile>%s)"
            % (flacseek.INDEX_PATH_ENV, flacseek.INDEX_SUFFIX)
        ),
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(args)

    index_path = args.output or flacseek.index_path_for(args.infile)

    def progress(done, total):
        if not args.quiet:
            print(f"\r{done / total * 100:5.1f}% ", end="", file=sys.stderr, flush=True)

    start = time.time()
    try:
        flacseek.build_index(args.infile, index_path, progress=progress)
    except (ValueError, OSError) as e:
        print(f"\nCannot index {args.infile}: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - start

    if not args.quiet:
        fd = os.open(args.infile, os.O_RDONLY)
        try:
            locator = flacseek.IndexLocator(flacseek.FlacStream(fd), index_path)
            print(
                f"\r{locator.nframes} frames, {locator.total_samples} samples, "
                f"index {os.path.getsize(index_path)} bytes in {elapsed:.1f} s -> {index_path}",
                file=sys.stderr,
            )
        finally:
            os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
