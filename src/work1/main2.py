import math
import taichi as ti

ti.init(arch=ti.cpu)

#基本参数
WIDTH = 700
HEIGHT = 700

eye_fov = 45.0
aspect_ratio = WIDTH / HEIGHT
zNear = 0.1
zFar = 50.0

#相机位置
eye_pos = ti.Vector([0.0, 0.0, 6.0])


#立方体顶点
#以原点为中心，边长为2
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

#立方体12条边
#前面4条、后面4条、连接前后面4条
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

#不同边使用不同颜色，便于观察空间结构
edge_colors = [
    0xFF5555, 0xFFAA00, 0xFFFF55, 0x55FF55,
    0x55FFFF, 0x5599FF, 0xAA55FF, 0xFF55AA,
    0xFFFFFF, 0xBBBBBB, 0x888888, 0x44FFAA
]

#绕X轴旋转
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

#绕Y轴旋转
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


#绕Z轴旋转
def rotate_z(angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)

    return ti.Matrix([
        [ c, -s, 0.0, 0.0],
        [ s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

#模型变换矩阵
#这里做法是：
#1. 先固定绕X轴转，让立方体有俯视感
#2. 再绕Y轴旋转，形成更明显的立体效果
#3. 再绕Z轴旋转，保留和必做题一致的Z轴旋转思想
def get_model_matrix(angle_x, angle_y, angle_z):
    model_x = rotate_x(angle_x)
    model_y = rotate_y(angle_y)
    model_z = rotate_z(angle_z)

    return model_z @ model_y @ model_x

#视图变换矩阵
#将相机平移到原点
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

#投影变换矩阵
#先透视到正交，再正交归一化
#与实验要求保持一致
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov_rad = math.radians(eye_fov)

    t = math.tan(fov_rad / 2.0) * zNear
    b = -t
    r = aspect_ratio * t
    l = -r

    #右手系下，相机朝-Z方向看
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

#转齐次坐标
def to_homogeneous(v):
    return ti.Vector([v[0], v[1], v[2], 1.0])

#顶点投影到屏幕
#先变换到裁剪空间，再除w做透视除法，最后映射到[0, 1]
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
    gui = ti.GUI("3D Cube Transformation (Taichi)", res=(WIDTH, HEIGHT))

    #初始角度
    angle_x = -25.0
    angle_y = 35.0
    angle_z = 15.0

    auto_rotate = False

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False

            elif e.key == 'a' or e.key == 'A':
                angle_z += 5.0
            elif e.key == 'd' or e.key == 'D':
                angle_z -= 5.0

            elif e.key == 'j' or e.key == 'J':
                angle_y += 5.0
            elif e.key == 'l' or e.key == 'L':
                angle_y -= 5.0

            elif e.key == 'i' or e.key == 'I':
                angle_x += 5.0
            elif e.key == 'k' or e.key == 'K':
                angle_x -= 5.0

            elif e.key == ti.GUI.SPACE:
                auto_rotate = not auto_rotate

            elif e.key == 'r' or e.key == 'R':
                angle_x = -25.0
                angle_y = 35.0
                angle_z = 15.0
                auto_rotate = False

        if auto_rotate:
            angle_y += 1.0
            angle_z += 0.6

        gui.clear(0x000000)

        model = get_model_matrix(angle_x, angle_y, angle_z)
        view = get_view_matrix(eye_pos)
        projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)

        mvp = projection @ view @ model
        screen_points = [project_vertex(v, mvp) for v in vertices]

        for idx, (i, j) in enumerate(edges):
            gui.line(
                begin=(screen_points[i][0], screen_points[i][1]),
                end=(screen_points[j][0], screen_points[j][1]),
                radius=2,
                color=edge_colors[idx]
            )

        gui.text(content=f"angle_x = {angle_x:.1f}", pos=(0.02, 0.96), color=0xFFFFFF)
        gui.text(content=f"angle_y = {angle_y:.1f}", pos=(0.02, 0.92), color=0xFFFFFF)
        gui.text(content=f"angle_z = {angle_z:.1f}", pos=(0.02, 0.88), color=0xFFFFFF)

        gui.text(content="I / K : rotate around X", pos=(0.02, 0.82), color=0xFFFFFF)
        gui.text(content="J / L : rotate around Y", pos=(0.02, 0.78), color=0xFFFFFF)
        gui.text(content="A / D : rotate around Z", pos=(0.02, 0.74), color=0xFFFFFF)
        gui.text(content="SPACE : auto rotate on/off", pos=(0.02, 0.70), color=0xFFFFFF)
        gui.text(content="R : reset", pos=(0.02, 0.66), color=0xFFFFFF)
        gui.text(content="ESC : exit", pos=(0.02, 0.62), color=0xFFFFFF)

        gui.show()


if __name__ == "__main__":
    main()