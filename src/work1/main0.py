import math
import taichi as ti

ti.init(arch=ti.cpu)

# =========================
# 基本参数
# =========================
WIDTH = 700
HEIGHT = 700

eye_fov = 45.0
aspect_ratio = WIDTH / HEIGHT
zNear = 0.1
zFar = 50.0

eye_pos = ti.Vector([0.0, 0.0, 5.0])

# 三角形三个顶点
vertices = [
    ti.Vector([2.0, 0.0, -2.0]),
    ti.Vector([0.0, 2.0, -2.0]),
    ti.Vector([-2.0, 0.0, -2.0]),
]

# 三条边
edges = [
    (0, 1, 0x00FF00),
    (1, 2, 0xFF0000),
    (2, 0, 0x0000FF),
]


# =========================
# 1. 模型变换矩阵：绕 Z 轴旋转
# =========================
def get_model_matrix(angle):
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)

    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


# =========================
# 2. 视图变换矩阵
# =========================
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])


# =========================
# 3. 投影矩阵
# =========================
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov_rad = math.radians(eye_fov)

    t = math.tan(fov_rad / 2.0) * zNear
    b = -t
    r = aspect_ratio * t
    l = -r

    n = -zNear
    f = -zFar

    persp_to_ortho = ti.Matrix([
        [n,   0.0,      0.0,       0.0],
        [0.0, n,        0.0,       0.0],
        [0.0, 0.0,   n + f,   -n * f],
        [0.0, 0.0,      1.0,       0.0],
    ])

    ortho_translate = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return ortho_scale @ ortho_translate @ persp_to_ortho


# =========================
# 齐次坐标
# =========================
def to_homogeneous(v):
    return ti.Vector([v[0], v[1], v[2], 1.0])


# =========================
# 投影到屏幕坐标
# =========================
def project_vertex(v, mvp):
    p = mvp @ to_homogeneous(v)

    ndc = ti.Vector([
        p[0] / p[3],
        p[1] / p[3],
        p[2] / p[3],
    ])

    return ti.Vector([
        (ndc[0] + 1.0) * 0.5,
        (ndc[1] + 1.0) * 0.5,
    ])


def main():
    gui = ti.GUI("3D Transformation (Taichi)", res=(WIDTH, HEIGHT))
    angle = 0.0

    while gui.running:
        # 按一次键，转一次
        while gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        gui.clear(0x000000)

        model = get_model_matrix(angle)
        view = get_view_matrix(eye_pos)
        projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)

        mvp = projection @ view @ model
        screen_points = [project_vertex(v, mvp) for v in vertices]

        for i, j, color in edges:
            gui.line(
                begin=(screen_points[i][0], screen_points[i][1]),
                end=(screen_points[j][0], screen_points[j][1]),
                radius=2,
                color=color
            )

        gui.text(content=f"angle = {angle:.1f}", pos=(0.02, 0.96), color=0xFFFFFF)
        gui.text(content="Press A to rotate counterclockwise", pos=(0.02, 0.92), color=0xFFFFFF)
        gui.text(content="Press D to rotate clockwise", pos=(0.02, 0.88), color=0xFFFFFF)
        gui.text(content="Press ESC to exit", pos=(0.02, 0.84), color=0xFFFFFF)

        gui.show()


if __name__ == "__main__":
    main()