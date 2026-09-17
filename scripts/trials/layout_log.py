"""Read the object layout a trial was run against, out of its Isaac Sim log.

Shared by the recorder (which stores the layout alongside the waypoints) and the
exporter (which recovers it when the recorder was run without the log). Kept in
its own module so the exporter, which is plain post-processing, does not have to
import rclpy just to reuse a regex.
"""
import re

LAYOUT = re.compile(r"\[Task\]\s+(beaker|flask|magnet|box|stirrer)\s+"
                    r"xy=\(([-\d.]+),([-\d.]+)\)\s+yaw=([-\d.]+)deg")
HOME = re.compile(r"\[Task\]\s+home_arm\(rad\) = \[([-\d.,\s]+)\]")


def read_layout(path):
    """(objects, home_arm_rad) from a sim log; empty dict / None if absent."""
    objs, home = {}, None
    try:
        for ln in open(path, errors="ignore"):
            m = LAYOUT.search(ln)
            if m:
                objs[m.group(1)] = {"xy": [float(m.group(2)), float(m.group(3))],
                                    "yaw_deg": float(m.group(4))}
            m = HOME.search(ln)
            if m and home is None:
                home = [float(v) for v in m.group(1).split(",")]
    except OSError:
        pass
    return objs, home
