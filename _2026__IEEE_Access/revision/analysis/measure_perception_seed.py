"""Perception-in-the-loop pose error and per-camera detection rate for one seed.

Ground truth is the simulator's own spawn log: the vessel xy/yaw it printed and
the tag plate world centre computed on the stage.  Vessel centre z is fixed by
the asset (tag z + the authored tag->object z).  Writes one CSV row per object
and saves both camera frames so a missed tag can be explained.
"""
import csv, math, os, re, sys, collections
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from apriltag_msgs.msg import AprilTagDetectionArray
import tf2_ros

OBJ_Z = {"beaker": 0.0622, "flask": 0.0550}   # tag_to_object_ z, post-lift
TAG_ID = {"beaker": 0, "flask": 1}
CAMS = ["camera_1", "camera_2"]
SPIN_S = 25.0   # the detector processes ~1.2 Hz, so spin long enough that a
                # rate over processed frames carries some weight

def parse(log):
    gt, tags = {}, {}
    for ln in open(log, errors="ignore"):
        m = re.search(r"\[Task\]\s+(beaker|flask)\s+xy=\(([-\d.]+),([-\d.]+)\)\s+yaw=([-\d.]+)deg", ln)
        if m:
            gt[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        m = re.search(r"\[Task\]\s+tag /World/(beaker|flask)/visual/apriltag_\d+ world=\(([-\d.]+),([-\d.]+),([-\d.]+)\)", ln)
        if m:
            tags[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return gt, tags

def yaw_of(q):
    return math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2)))

class Meas(Node):
    def __init__(self):
        super().__init__("measure_seed")
        self.frames = collections.Counter()   # images published by the camera
        self.msgs = collections.Counter()     # frames the detector processed
        self.hits = collections.Counter()     # processed frames holding the tag
        self.first = {}
        self.tf = collections.defaultdict(list)
        self.buf = tf2_ros.Buffer(cache_time=Duration(seconds=20))
        tf2_ros.TransformListener(self.buf, self)
        for c in CAMS:
            self.create_subscription(Image, "/%s/rgb" % c,
                                     lambda m, c=c: self.on_img(c, m), 10)
            self.create_subscription(AprilTagDetectionArray,
                                     "/%s/apriltag/detections" % c,
                                     lambda m, c=c: self.on_det(c, m), 10)
    def on_img(self, cam, m):
        self.frames[cam] += 1
        if cam not in self.first:
            ch = {"rgb8": 3, "rgba8": 4, "bgr8": 3, "bgra8": 4}.get(m.encoding, 3)
            a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, ch)
            self.first[cam] = a[:, :, :3].copy()
    def on_det(self, cam, msg):
        # The detector publishes one array per processed frame, empty or not,
        # and processes far fewer frames than the camera renders, so a rate
        # must be per PROCESSED frame rather than per rendered frame.
        self.msgs[cam] += 1
        for d in msg.detections:
            self.hits[(cam, d.id)] += 1

seed, log, csv_path, img_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
gt, tags = parse(log)
if len(gt) < 2 or len(tags) < 2:
    print("seed %s: incomplete ground truth gt=%s tags=%s" % (seed, list(gt), list(tags)))
    sys.exit(1)

rclpy.init(); n = Meas()
import time
t0 = time.time()
while time.time() - t0 < SPIN_S:
    rclpy.spin_once(n, timeout_sec=0.02)
    for f in ("beaker", "flask", "beaker_tag", "flask_tag"):
        try:
            t = n.buf.lookup_transform("base_link", f, rclpy.time.Time())
        except Exception:
            continue
        tr = t.transform.translation
        n.tf[f].append((tr.x, tr.y, tr.z, yaw_of(t.transform.rotation)))

os.makedirs(img_dir, exist_ok=True)
for c, img in n.first.items():
    np.save(os.path.join(img_dir, "s%s_%s.npy" % (seed, c)), img)

new = not os.path.exists(csv_path)
fh = open(csv_path, "a", newline="")
wr = csv.writer(fh)
if new:
    wr.writerow(["seed", "object", "published", "dxy_mm", "dz_mm", "dyaw_deg",
                 "tag_dxy_mm", "tag_dz_mm", "n_tf",
                 "cam1_rate", "cam2_rate", "n_cams", "cam1_proc", "cam2_proc",
                 "gt_x", "gt_y", "gt_yaw", "tag_x", "tag_y"])

print("seed %s  rendered cam1=%d cam2=%d | processed cam1=%d cam2=%d"
      % (seed, n.frames["camera_1"], n.frames["camera_2"],
         n.msgs["camera_1"], n.msgs["camera_2"]))
for obj in ("beaker", "flask"):
    tid = TAG_ID[obj]
    rates = []
    for c in CAMS:
        m = n.msgs[c]
        rates.append((n.hits[(c, tid)] / m) if m else 0.0)
    ncam = sum(1 for r in rates if r > 0.1)
    v = n.tf.get(obj)
    tv = n.tf.get(obj + "_tag")
    g = (gt[obj][0], gt[obj][1], tags[obj][2] + OBJ_Z[obj])
    if v:
        x, y, z, w = [sum(p[i] for p in v)/len(v) for i in range(4)]
        dxy = math.hypot(x-g[0], y-g[1])*1000.0
        dz = (z-g[2])*1000.0
        dyaw = ((w-gt[obj][2]+180) % 360)-180
    else:
        dxy = dz = dyaw = float("nan")
    if tv:
        tx, ty, tz, _ = [sum(p[i] for p in tv)/len(tv) for i in range(4)]
        tdxy = math.hypot(tx-tags[obj][0], ty-tags[obj][1])*1000.0
        tdz = (tz-tags[obj][2])*1000.0
    else:
        tdxy = tdz = float("nan")
    wr.writerow([seed, obj, int(bool(v)), "%.2f" % dxy, "%.2f" % dz, "%.2f" % dyaw,
                 "%.2f" % tdxy, "%.2f" % tdz, len(v or ()),
                 "%.3f" % rates[0], "%.3f" % rates[1], ncam,
                 n.msgs["camera_1"], n.msgs["camera_2"],
                 gt[obj][0], gt[obj][1], gt[obj][2], tags[obj][0], tags[obj][1]])
    print("  %-7s pub=%d dxy=%7.2f dz=%7.2f dyaw=%6.2f | cam rates %.2f/%.2f ncam=%d"
          % (obj, int(bool(v)), dxy, dz, dyaw, rates[0], rates[1], ncam))
fh.close()
rclpy.shutdown()
