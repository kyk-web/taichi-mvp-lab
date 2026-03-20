import math
import taichi as ti

ti.init(arch=ti.cpu)

WIDTH = 700
HEIGHT = 700

eye_fov = 45.0
aspect_ratio = WIDTH / HEIGHT
zNear = 0.1
zFar = 50.0

eye_pos = ti.Vector([0.0, 0.0, 6.0])

# 正方体 8 个顶点，中心在原点，边长为 2
vertices = [
    ti.Vector([-1.0, -1.0, -1.0]),  # 0
    ti.Vector([ 1.0, -1.0, -1.0]),  # 1
    ti.Vector([ 1.0,  1.0, -1.0]),  # 2
    ti.Vector([-1.0,  1.0, -1.0]),  # 3
    ti.Vector([-1.0, -1.0,  1.0]),  # 4
    ti.Vector([ 1.0, -1.0,  1.0]),  # 5
    ti.Vector([ 1.0,  1.0,  1.0]),  # 6
    ti.Vector([-1.0,  1.0,  1.0]),  # 7
]

# 正方体 12 条边
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


def rotate_x(angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s, 0.0],
        [0.0,  s,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotate_y(angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return ti.Matrix([
        [ c, 0.0,  s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def get_model_matrix(angle):
    fixed_x = rotate_x(-30.0)
    dynamic_y = rotate_y(angle)
    return dynamic_y @ fixed_x


def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])


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


def to_homogeneous(v):
    return ti.Vector([v[0], v[1], v[2], 1.0])


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
    gui = ti.GUI("3D Cube (Taichi)", res=(WIDTH, HEIGHT))
    angle = 35.0

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == 'a' or e.key == 'A':
                angle += 5.0
            elif e.key == 'd' or e.key == 'D':
                angle -= 5.0
            elif e.key == ti.GUI.ESCAPE:
                gui.running = False

        gui.clear(0x000000)

        model = get_model_matrix(angle)
        view = get_view_matrix(eye_pos)
        projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)

        mvp = projection @ view @ model
        screen_points = [project_vertex(v, mvp) for v in vertices]

        for i, j in edges:
            gui.line(
                begin=(screen_points[i][0], screen_points[i][1]),
                end=(screen_points[j][0], screen_points[j][1]),
                radius=2,
                color=0xFFFFFF
            )

        gui.show()


if __name__ == "__main__":
    main()