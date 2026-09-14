#!/usr/bin/env python3
"""Render tag36h11 images, and VERIFY each one with the detector before writing.

A tag image that looks right and decodes to nothing is the expensive failure
here: it shows up as "perception found no object", which reads like a pipeline
problem anywhere along a chain of five nodes. The bit order and polarity of the
36h11 codebook are not something to assume, so this renders all four
combinations, runs the real detector over them, and only writes the one that
decodes back to the id it encoded.

Rendered with INTER_NEAREST on purpose: a smoothed tag edge is what the quad
detector fails on first, and a texture magnified by a renderer will smooth it
again anyway.

    python3 TAMP/tamp/scripts/perception/generate_tag_images.py OUTDIR --ids 0 1
"""

import argparse
import os
import re
import sys

import apriltag
import cv2
import numpy as np

#: The 36h11 codebook, taken from the header the ROS apriltag stack ships
#: rather than re-derived, so the codes are the detector's own.
CODEBOOK_HEADER = '/opt/ros/humble/include/apriltag_mit/AprilTags/Tag36h11.h'

#: 6x6 data cells, one black ring, one white quiet ring = 10 cells across.
#: The detector's `size` parameter measures the BLACK SQUARE, which is 8 of
#: those 10 cells -- so a plate carrying this image must be size * 10 / 8 wide.
DATA_CELLS = 6
IMAGE_CELLS = DATA_CELLS + 4
BLACK_SQUARE_CELLS = DATA_CELLS + 2


def load_codes(path=CODEBOOK_HEADER):
    with open(path, encoding='utf-8') as stream:
        return [int(m, 16) for m in re.findall(r'0x([0-9a-fA-F]+)LL', stream.read())]


def render(code, order, polarity, cell=40):
    """One tag as a uint8 image, `cell` pixels per cell."""
    grid = np.zeros((DATA_CELLS, DATA_CELLS), dtype=np.uint8)
    for i in range(DATA_CELLS * DATA_CELLS):
        bit = (code >> (35 - i)) & 1 if order == 'msb' else (code >> i) & 1
        row, col = divmod(i, DATA_CELLS)
        grid[row, col] = bit
    if polarity == 'inv':
        grid = 1 - grid

    img = np.full((IMAGE_CELLS, IMAGE_CELLS), 255, dtype=np.uint8)
    img[1:-1, 1:-1] = 0                      # black ring
    img[2:-2, 2:-2] = grid * 255             # data
    return cv2.resize(img, (IMAGE_CELLS * cell,) * 2, interpolation=cv2.INTER_NEAREST)


def make_detector(family='tag36h11'):
    """A detector from whichever apriltag python binding is installed.

    ROS ships the upstream `apriltag.apriltag(family)` class; the pip package
    of the same name exposes `Detector(DetectorOptions(...))` instead, and both
    turn up on machines that have run this stack. Detections come back as dicts
    from one and objects from the other, which `detected_id` absorbs.
    """
    if hasattr(apriltag, 'Detector'):
        return apriltag.Detector(apriltag.DetectorOptions(families=family))
    return apriltag.apriltag(family)


def detected_id(detector, img):
    padded = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
    found = detector.detect(padded)
    if not found:
        return None
    first = found[0]
    return first['id'] if isinstance(first, dict) else first.tag_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('outdir')
    parser.add_argument('--ids', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--cell', type=int, default=40,
                        help='pixels per cell; 40 gives a 400 px image')
    args = parser.parse_args(argv)

    codes = load_codes()
    print('codes parsed: %d, id0=%#x' % (len(codes), codes[0]))
    detector = make_detector()

    probe = args.ids[:2] or [0]
    convention = None
    for order in ('msb', 'lsb'):
        for polarity in ('norm', 'inv'):
            got = [detected_id(detector, render(codes[i], order, polarity)) for i in probe]
            print('order=%-4s polarity=%-4s -> detected %s' % (order, polarity, got))
            if got == probe:
                convention = (order, polarity)
                break
        if convention:
            break

    if convention is None:
        print('no bit order / polarity decoded back to the ids it encoded', file=sys.stderr)
        return 1
    order, polarity = convention
    print('using order=%s polarity=%s' % (order, polarity))

    os.makedirs(args.outdir, exist_ok=True)
    for tag_id in args.ids:
        img = render(codes[tag_id], order, polarity, cell=args.cell)
        # Verify the one actually written, not just the two probed.
        got = detected_id(detector, img)
        if got != tag_id:
            print('tag %d rendered but decoded as %s; refusing to write'
                  % (tag_id, got), file=sys.stderr)
            return 1
        path = os.path.join(args.outdir, 'tag36h11_%02d.png' % tag_id)
        cv2.imwrite(path, img)
        print('  wrote %s (%dx%d, verified as id %d)' % (path, img.shape[1], img.shape[0], got))
    return 0


if __name__ == '__main__':
    # os._exit, not sys.exit: the ROS apriltag C binding segfaults in its
    # destructor at interpreter shutdown, which turns a successful run into
    # exit 139 and fails any script that checks the status. The images are
    # already written and verified by the time we get here.
    code = main()
    sys.stdout.flush()
    os._exit(code)
