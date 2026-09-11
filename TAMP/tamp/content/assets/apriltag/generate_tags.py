"""Render tag36h11 images and VERIFY them with the detector before use."""
import re, sys
import numpy as np, cv2, apriltag

HDR = "/opt/ros/humble/include/apriltag_mit/AprilTags/Tag36h11.h"
codes = [int(m, 16) for m in re.findall(r"0x([0-9a-fA-F]+)LL", open(HDR).read())]
print("codes parsed:", len(codes), "id0=%#x" % codes[0])

def render(code, order, polarity, cell=40):
    """6x6 data + 1 black border + 1 white border, `cell` px per cell."""
    n = 6
    grid = np.zeros((n, n), dtype=np.uint8)
    for i in range(36):
        bit = (code >> (35 - i)) & 1 if order == "msb" else (code >> i) & 1
        r, c = divmod(i, n)
        grid[r, c] = bit
    if polarity == "inv":
        grid = 1 - grid
    img = np.zeros((n + 4, n + 4), dtype=np.uint8)   # white ring + black ring
    img[:] = 255
    img[1:-1, 1:-1] = 0                                # black border ring
    img[2:-2, 2:-2] = grid * 255
    return cv2.resize(img, ((n + 4) * cell,) * 2, interpolation=cv2.INTER_NEAREST)

det = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
for order in ("msb", "lsb"):
    for polarity in ("norm", "inv"):
        ok = []
        for tid in (0, 1):
            img = render(codes[tid], order, polarity)
            pad = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
            res = det.detect(pad)
            ok.append(res[0].tag_id if res else None)
        print("order=%-4s polarity=%-4s -> detected ids %s" % (order, polarity, ok))
        if ok == [0, 1]:
            print("MATCH: order=%s polarity=%s" % (order, polarity))
            for tid in (0, 1):
                out = "%s/tag36h11_%02d.png" % (sys.argv[1], tid)
                cv2.imwrite(out, render(codes[tid], order, polarity))
                print("  wrote", out)
            sys.exit(0)
print("no combination detected correctly")
sys.exit(1)
