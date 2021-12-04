# Tutorials (Chinese):
# - https://www.bilibili.com/video/BV1UK4y177iH
# - https://www.bilibili.com/video/BV1DK411A771

import taichi as ti

ti.init(arch=ti.cpu, debug=True, excepthook=True)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-2
substeps = 5

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
H = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(
    max_num_particles, max_num_particles))
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32,
                       shape=(max_num_particles, max_num_particles))
# solve A@v=b   v=dx/dt*substepts
v_new = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
b = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
Av = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
Ad = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
r = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
d = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)


@ti.kernel
def substep1():
    n = num_particles[None]
    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if (rest_length[i, j] != 0):
                x_ij = x[i] - x[j]
                l = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] -
                                           1) * l

                # # Dashpot damping
                # v_rel = (v[i] - v[j]).dot(d)
                # f[i] += -dashpot_damping[None] * v_rel * d


@ti.kernel
def substep2():
    n = num_particles[None]
    # Compute Hessian
    I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    for i in range(n-1):
        for j in range(i,n):
            if (rest_length[i, j] != 0):
                dx = x[i] - x[j]
                l = dx.norm()
                dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],[dx[1] * dx[0], dx[1] * dx[1]]])
                if l != 0.0:
                    l = 1.0 / l
                H[i, j] = -(I - rest_length[i,j] * l *(I - dxtdx * l**2))*spring_Y[None]
                H[j, i] = H[i, j]
    for i in range(n):
        tmp = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for j in range(n):
            tmp -= H[i, j]
        H[i, i] = tmp

    # # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    # for i in range(n):
    #     if not fixed[i]:
    #         v[i] += dt * f[i] / particle_mass
    #         v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

    #         x[i] += v[i] * dt
    #     else:
    #         v[i] = ti.Vector([0, 0])

    #     # Collide with four walls
    #     for d in ti.static(range(2)):
    #         # d = 0: treating X (horizontal) component
    #         # d = 1: treating Y (vertical) component

    #         if x[i][d] < 0:  # Bottom and left
    #             x[i][d] = 0  # move particle inside
    #             v[i][d] = 0  # stop it from moving further

    #         if x[i][d] > 1:  # Top and right
    #             x[i][d] = 1  # move particle inside
    #             v[i][d] = 0  # stop it from moving further


@ti.func
def dot(a, b):
    result = 0.0
    for i in range(num_particles[None]):
        result += a[i].dot(b[i])
    return result


@ti.kernel
def getAd():
    # Ad=A@d
    coeff = (dt/substeps)**2
    for i in range(num_particles[None]):
        Ad[i] = particle_mass*d[i]
    for i in range(num_particles[None]):
        for j in range(num_particles[None]):
            Ad[i] -= coeff*H[i, j]@d[i]


@ti.kernel
def before_iter() -> ti.f32:
    # Av=A@v
    coeff = (dt/substeps)**2
    for i in range(num_particles[None]):
        v_new[i] = ti.Vector([0.0, 0.0])
    for i in range(num_particles[None]):
        Av[i] = particle_mass*v_new[i]
    for i in range(num_particles[None]):
        for j in range(num_particles[None]):
            Av[i] -= coeff*H[i, j]@v_new[i]

    h = dt/substeps
    for i in range(num_particles[None]):
        b[i] = particle_mass*v[i]+h*f[i]

    # r = b - A @ x
    for i in range(num_particles[None]):
        r[i] = b[i]-Av[i]

    # d=r
    for i in range(num_particles[None]):
        d[i] = r[i]

    return dot(r, r)


@ti.kernel
def run_iteration(delta_new: ti.f32) -> ti.f32:

    alpha = delta_new / dot(d, Ad)  # alpha = (r^T * r) / dot(d, Ad)
    for i in range(num_particles[None]):
        v_new[i] += alpha * d[i]  # x^{i+1} = x^{i} + alpha * d
        r[i] -= alpha * Ad[i]  # r^{i+1} = r^{i} + alpha * Ad
    delta_old = delta_new
    delta_new = dot(r, r)
    beta = delta_new / delta_old
    for i in range(num_particles[None]):
        d[i] = r[i] + beta * d[i]  # p^{i+1} = r^{i+1} + beta * p^{i}
    return delta_new

# Use conjugate gradient


def cg():
    i = 0
    i_max = 2*num_particles[None]
    delta_new = before_iter()
    delta_0 = delta_new

    while i < i_max and delta_new/delta_0 > 1.0e-6:
        getAd()  # Ad = A @ d
        delta_new = run_iteration(delta_new)
        i += 1


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = dist
            rest_length[new_particle_id, i] = dist


@ti.kernel
def upDateVX():
    for i in range(num_particles[None]):
        if not fixed[i]:
            v[i] = v_new[i]
            x[i] += v_new[i]*dt/substeps
        else:
            v[i] = ti.Vector([0, 0])

    # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further


@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 100


def main():
    gui = ti.GUI('Explicit Mass Spring System',
                 res=(512, 512),
                 background_color=0xDDDDDD)

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100
    num_particles[None] = 0
    paused[None] = 0
    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]
            elif e.key == ti.GUI.LMB:
                new_particle(e.pos[0], e.pos[1],
                             int(gui.is_pressed(ti.GUI.SHIFT)))
            elif e.key == 'c':
                num_particles[None] = 0
                rest_length.fill(0)
            elif e.key == 'y':
                if gui.is_pressed('Shift'):
                    spring_Y[None] /= 1.1
                else:
                    spring_Y[None] *= 1.1
            elif e.key == 'd':
                if gui.is_pressed('Shift'):
                    drag_damping[None] /= 1.1
                else:
                    drag_damping[None] *= 1.1
            elif e.key == 'x':
                if gui.is_pressed('Shift'):
                    dashpot_damping[None] /= 1.1
                else:
                    dashpot_damping[None] *= 1.1

        if gui.is_pressed(ti.GUI.RMB):
            cursor_pos = gui.get_cursor_pos()
            attract(cursor_pos[0], cursor_pos[1])

        if not paused[None]:
            for step in range(substeps):
                substep1()
                substep2()
                cg()
                upDateVX()
        X = x.to_numpy()
        n = num_particles[None]

        # Draw the springs
        for i in range(n):
            for j in range(i + 1, n):
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i], end=X[j], radius=2, color=0x444444)

        # Draw the particles
        for i in range(n):
            c = 0xFF0000 if fixed[i] else 0x111111
            gui.circle(pos=X[i], color=c, radius=5)

        gui.text(
            content=f'Left click: add mass point (with shift to fix); Right click: attract',
            pos=(0, 0.99),
            color=0x0)
        gui.text(content=f'C: clear all; Space: pause',
                 pos=(0, 0.95),
                 color=0x0)
        gui.text(content=f'Y: Spring Young\'s modulus {spring_Y[None]:.1f}',
                 pos=(0, 0.9),
                 color=0x0)
        gui.text(content=f'D: Drag damping {drag_damping[None]:.2f}',
                 pos=(0, 0.85),
                 color=0x0)
        gui.text(content=f'X: Dashpot damping {dashpot_damping[None]:.2f}',
                 pos=(0, 0.8),
                 color=0x0)
        gui.show()


if __name__ == '__main__':
    main()
