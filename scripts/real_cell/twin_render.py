"""An offscreen 3D view of the real cell: the FR5 from its URDF, the furniture, the beakers.

Used by render_run_video.py for the panel that shows the robot as its joint
states had it and the beakers where perception put them. Rendered with OpenGL
through EGL (no window, no X): the URDF's meshes and primitives through
pinocchio, a fixed camera, Lambert + Blinn shading, supersampled 2x.

    r = TwinRenderer(urdf_xml, 960, 540)
    img, P = r.render({'j1': ..., 'gripper_finger_joint': ...},
                      beakers=[(T_base_beaker, (b, g, r), alpha), ...])

`P` is the 3x4 projection (base frame -> pixels of `img`) for drawing over
the image afterwards (lines of sight, labels).
"""
import ctypes
import os
import tempfile

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np  # noqa: E402
from OpenGL import EGL, GL  # noqa: E402

# The two-beaker cell's furniture, as the planner has it (record_transfer_real.sh,
# SCENE_ENV): the bench top, the riser bar under beaker1, the balance under beaker2.
BENCH_Z = -0.013
RISER = dict(center=(0.354, -0.295), dims=(0.200, 0.070, 0.072))
SCALE = dict(body=(0.505, 0.320), dims=(0.25, 0.20, 0.092), pan=(0.520, 0.320))
BEAKER_R, BEAKER_H = 0.025, 0.070

ROBOT_RGB = (0.93, 0.93, 0.92)
GRIPPER_RGB = (0.16, 0.16, 0.18)

VERT = """
#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
uniform mat4 M, VP;
out vec3 wpos, wnrm;
void main() {
    vec4 w = M * vec4(pos, 1.0);
    wpos = w.xyz;
    wnrm = mat3(M) * nrm;
    gl_Position = VP * w;
}
"""
FRAG = """
#version 330 core
in vec3 wpos, wnrm;
uniform vec3 color, eye;
uniform float alpha;
out vec4 frag;
void main() {
    vec3 n = normalize(wnrm);
    vec3 v = normalize(eye - wpos);
    if (dot(n, v) < 0.0) n = -n;
    vec3 l1 = normalize(vec3(0.4, -0.3, 1.0)), l2 = normalize(vec3(-0.6, 0.8, 0.5));
    float d = 0.55 * max(dot(n, l1), 0.0) + 0.25 * max(dot(n, l2), 0.0);
    float s = pow(max(dot(n, normalize(l1 + v)), 0.0), 40.0) * 0.25;
    frag = vec4(color * (0.28 + d) + vec3(s), alpha);
}
"""


def _stl(path):
    """Triangles (N, 3, 3) of a binary or ASCII STL."""
    data = open(path, 'rb').read()
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], 'little')
        if 84 + 50 * n == len(data):
            rec_t = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            rec = np.frombuffer(data[84:], dtype=rec_t, count=n)
            return rec['v'].astype(np.float32)
    verts = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors='ignore').splitlines()
             if ln.strip().startswith('vertex')]
    return np.array(verts, np.float32).reshape(-1, 3, 3)


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[lo[0] if i & 1 == 0 else hi[0], lo[1] if i & 2 == 0 else hi[1],
                   lo[2] if i & 4 == 0 else hi[2]] for i in range(8)])
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    return np.array([[c[a], c[b], c[d]] for a, b, cc, d in quads for _ in [0]] +
                    [[c[b], c[cc], c[d]] for a, b, cc, d in quads], np.float32)


def _cylinder(r, h, n=48, z0=None, capped=True):
    """Along z, from z0 (default -h/2) to z0 + h."""
    z0 = -h / 2 if z0 is None else z0
    a = np.linspace(0, 2 * np.pi, n + 1)
    tris = []
    for i in range(n):
        p0 = (r * np.cos(a[i]), r * np.sin(a[i]))
        p1 = (r * np.cos(a[i + 1]), r * np.sin(a[i + 1]))
        b0, b1, t0, t1 = (*p0, z0), (*p1, z0), (*p0, z0 + h), (*p1, z0 + h)
        tris += [[b0, b1, t1], [b0, t1, t0]]
        if capped:
            tris += [[(0, 0, z0), b1, b0], [(0, 0, z0 + h), t0, t1]]
    return np.array(tris, np.float32)


def _normals(tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.repeat(n[:, None, :], 3, axis=1).astype(np.float32)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(x, float) for x in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = s, u, -f
    V[:3, 3] = -V[:3, :3] @ eye
    return V


def perspective(fovy_deg, aspect, near=0.05, far=10.0):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    P = np.zeros((4, 4))
    P[0, 0], P[1, 1] = f / aspect, f
    P[2, 2], P[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    P[3, 2] = -1.0
    return P


class _Mesh:
    def __init__(self, tris):
        tris = np.ascontiguousarray(tris, np.float32)
        data = np.concatenate([tris.reshape(-1, 3), _normals(tris).reshape(-1, 3)], axis=1)
        data = np.ascontiguousarray(data, np.float32)
        self.n = len(data)
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)


class TwinRenderer:
    def __init__(self, urdf_xml, width, height, eye=(1.62, 0.06, 1.02),
                 target=(0.26, 0.0, 0.30), fovy=42.0, ss=2):
        import pinocchio as pin
        self.w, self.h, self.ss = width, height, ss
        self._egl(width * ss, height * ss)
        prog = GL.glCreateProgram()
        for src, kind in ((VERT, GL.GL_VERTEX_SHADER), (FRAG, GL.GL_FRAGMENT_SHADER)):
            sh = GL.glCreateShader(kind)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sh))
            GL.glAttachShader(prog, sh)
        GL.glLinkProgram(prog)
        self.prog = prog
        self.u = {k: GL.glGetUniformLocation(prog, k) for k in ('M', 'VP', 'color', 'eye', 'alpha')}

        share = os.path.expanduser('~/ros2_ws/install')
        dirs = [os.path.join(share, p, 'share') for p in os.listdir(share)
                if os.path.isdir(os.path.join(share, p, 'share'))]
        with tempfile.NamedTemporaryFile('w', suffix='.urdf', delete=False) as f:
            f.write(urdf_xml)
        self.pin = pin
        self.model, _, self.vis = pin.buildModelsFromUrdf(f.name, package_dirs=dirs)
        os.unlink(f.name)
        self.data = self.model.createData()
        self.vdata = pin.GeometryData(self.vis)
        self.parts = []
        for g in self.vis.geometryObjects:
            geom = g.geometry
            kind = type(geom).__name__
            if g.meshPath and os.path.exists(g.meshPath):
                tris = _stl(g.meshPath) * np.asarray(g.meshScale, np.float32)
            elif kind == 'Box':
                hs = np.asarray(geom.halfSide)
                tris = _box(-hs, hs)
            elif kind == 'Cylinder':
                tris = _cylinder(geom.radius, 2 * geom.halfLength)
            else:
                continue
            rgba = np.asarray(g.meshColor, float)
            if rgba[3] < 0.5:
                continue        # the jaw envelope, drawn translucent for collision checks
            gripper = 'gripper' in g.name
            color = tuple(rgba[:3]) if gripper and np.ptp(rgba[:3]) + abs(rgba[0] - 0.9) > 1e-3 \
                else GRIPPER_RGB if gripper else ROBOT_RGB
            self.parts.append((g.name, _Mesh(tris), color))
        self.wrist3 = self.model.getFrameId('wrist3_link')
        # the jaws, which the URDF has no visual for: AG-95 fingers ~16 x 24 mm,
        # 110 mm along the approach from 0.205 m (pouring_controller.yaml hand_boxes)
        self.jaw = _Mesh(_box((0.0, -0.012, 0.2054), (0.016, 0.012, 0.3158)))
        self.beaker = _Mesh(_cylinder(BEAKER_R, BEAKER_H, z0=0.0, capped=False))
        self.beaker_floor = _Mesh(_cylinder(BEAKER_R, 0.002, z0=0.0))
        self.scene = []
        rx, ry = RISER['center']
        dx, dy, dz = RISER['dims']
        self.scene.append((_Mesh(_box((rx - dx / 2, ry - dy / 2, BENCH_Z), (rx + dx / 2, ry + dy / 2, BENCH_Z + dz))),
                           (0.10, 0.10, 0.11)))
        bx, by = SCALE['body']
        sx, sy, sz = SCALE['dims']
        self.scene.append((_Mesh(_box((bx - sx / 2, by - sy / 2, BENCH_Z), (bx + sx / 2, by + sy / 2, BENCH_Z + sz - 0.004))),
                           (0.86, 0.87, 0.88)))
        px, py = SCALE['pan']
        self.scene.append((_Mesh(_box((px - 0.075, py - 0.085, BENCH_Z + sz - 0.004), (px + 0.075, py + 0.085, BENCH_Z + sz))),
                           (0.70, 0.71, 0.73)))
        self.scene.append((_Mesh(_box((-0.25, -0.45, BENCH_Z - 0.02), (0.75, 0.55, BENCH_Z))), (0.62, 0.63, 0.66)))
        self.scene.append((_Mesh(_box((0.21, -0.11, BENCH_Z), (0.49, 0.10, BENCH_Z + 0.0005))), (0.96, 0.96, 0.96)))
        self.views = {}
        self.add_view('main', eye, target, fovy)
        # straight down on the bench, the robot at the top as the main view has it behind
        self.add_view('top', (0.36, 0.05, 1.75), (0.36, 0.05, 0.0), 42.0, up=(-1.0, 0.0, 0.0))
        self.view('main')

    def add_view(self, name, eye, target, fovy, up=(0, 0, 1)):
        self.views[name] = (look_at(eye, target, up), perspective(fovy, self.w / self.h),
                            np.asarray(eye, float))

    def view(self, name):
        self.V, self.P, self.eye = self.views[name]

    def _egl(self, w, h):
        dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        EGL.eglInitialize(dpy, ctypes.pointer(EGL.EGLint()), ctypes.pointer(EGL.EGLint()))
        attrs = (EGL.EGLint * 13)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_RED_SIZE, 8,
                                  EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24,
                                  EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
        cfg, n = EGL.EGLConfig(), EGL.EGLint()
        EGL.eglChooseConfig(dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n))
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        pbuf = (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16, EGL.EGL_NONE)
        surf = EGL.eglCreatePbufferSurface(dpy, cfg, pbuf)
        ctx_attrs = (EGL.EGLint * 7)(EGL.EGL_CONTEXT_MAJOR_VERSION, 3, EGL.EGL_CONTEXT_MINOR_VERSION, 3,
                                     EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                     EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL.EGL_NONE)
        ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, ctx_attrs)
        if not EGL.eglMakeCurrent(dpy, surf, surf, ctx):
            raise RuntimeError('EGL: could not make a context current')
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        rb = GL.glGenRenderbuffers(2)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb[0])
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb[0])
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb[1])
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, w, h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rb[1])
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('EGL: framebuffer incomplete')

    def _draw(self, mesh, M, color, alpha=1.0):
        GL.glUniformMatrix4fv(self.u['M'], 1, True, np.asarray(M, np.float32))
        GL.glUniform3f(self.u['color'], *color)
        GL.glUniform1f(self.u['alpha'], alpha)
        GL.glBindVertexArray(mesh.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh.n)

    def link_pose(self, q, frame):
        """4x4 base-frame pose of a URDF frame at joint values q (dict)."""
        self._fk(q)
        return self.data.oMf[self.model.getFrameId(frame)].homogeneous.copy()

    def _fk(self, q):
        pin = self.pin
        v = pin.neutral(self.model)
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            if name in q:
                v[self.model.joints[i].idx_q] = q[name]
        pin.forwardKinematics(self.model, self.data, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.vis, self.vdata)

    def projection(self):
        """3x4: base frame -> pixel coordinates of the rendered image."""
        Pix = np.array([[self.w / 2, 0, 0, self.w / 2], [0, -self.h / 2, 0, self.h / 2], [0, 0, 0, 1.0]])
        return Pix @ self.P @ self.V

    def render(self, q, beakers=(), background=(0.11, 0.11, 0.13), view='main'):
        import cv2
        self.view(view)
        self._fk(q)
        W, H = self.w * self.ss, self.h * self.ss
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, W, H)
        GL.glClearColor(*background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.u['VP'], 1, True, (self.P @ self.V).astype(np.float32))
        GL.glUniform3f(self.u['eye'], *self.eye)
        for mesh, color in self.scene:
            self._draw(mesh, np.eye(4), color)
        for (name, mesh, color), placement in zip(self.parts, self.vdata.oMg):
            self._draw(mesh, placement.homogeneous, color)
        T3 = self.data.oMf[self.wrist3].homogeneous
        finger = q.get('gripper_finger_joint', 0.0475)
        for side in (1.0, -1.0):
            J = np.eye(4)
            J[0, 0] = side
            J[0, 3] = side * finger
            self._draw(self.jaw, T3 @ J, (0.22, 0.22, 0.24))
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        for T, bgr, alpha in beakers:
            rgb = tuple(c / 255.0 for c in bgr[::-1])
            self._draw(self.beaker_floor, T, rgb, min(1.0, alpha + 0.3))
            self._draw(self.beaker, T, rgb, alpha)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        buf = GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.ss > 1:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return img, self.projection()


def project(P, pts):
    """Base-frame points (N, 3) -> pixels (N, 2) and in-front mask."""
    pts = np.atleast_2d(pts)
    h = P @ np.c_[pts, np.ones(len(pts))].T
    return (h[:2] / h[2]).T, h[2] > 0
