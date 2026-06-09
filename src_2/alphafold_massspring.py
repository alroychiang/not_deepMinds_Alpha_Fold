from alphafold_robot_config import robots
import argparse
import random
import sys
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
import time

random.seed(0)
np.random.seed(0)
real = ti.f32

# ti.init(default_fp=real, arch=ti.cpu, debug = False, flatten_if=True)
ti.init(default_fp=real, arch=ti.metal, debug = False, flatten_if=True)

max_steps = 4096
vis_interval = 2
output_vis_interval = 8
steps = 256
# ensures the condition is met, else will show assertion error.
assert steps * 2 <= max_steps

# a declartion of a scalar field and a vector field, which will be used in the simulation.
scalar = lambda: ti.field(dtype=real)
# comprises of 3D vectors (Z is the up axis)
vec = lambda: ti.Vector.field(3, dtype=real)
loss = scalar()

# have yet to declare
x = vec()
v = vec()
v_inc = vec()

head_id = 0
goal = vec()
n_objects = 0
# target_ball = 0
elasticity = 0.0
ground_height = 0.1
gravity = -4.8      # normal; compression + friction now hold wrinkles on their own
friction = 1.5      # lowered so wrinkles persist but still yield to the iron

gradient_clip = 1
spring_omega = 10
damping = 20        # normal-ish; compression + friction keep wrinkles stable

n_springs = 0
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()

center = vec()

# --- iron (Phase 1: passive, hardcoded sweep) ---
iron_pos = vec()        # iron center position per timestep, shape (max_steps,)
iron_half_x = 0.08      # plate half-width in X (physics contact box)
iron_half_y = 0.10      # plate half-width in Y — spans full cloth width (Y 0.1-0.25)
iron_push = 2000.0      # soft push-down stiffness; high enough to overpower pinned wrinkles
iron_smooth = 600.0     # horizontal drag in the sweep direction (-X) to spread material
iron_active = True      # iron sweeps and flattens the settled wrinkles
iron_z = ground_height + 0.005  # plate bottom height (the press is really a ceiling)
iron_speed = 2.0        # scales nn2's velocity command; large enough to sweep the
                        # ~0.6-wide cloth within the iron phase

# Phase 2: nn2 now drives the iron's horizontal motion (Δx, Δy) instead of
# per-spring actuation. The cloth is passive; only the iron acts.
n_iron_act = 2          # iron velocity command components: X, Y

# --- Option B: settle-then-relax compression ---
# The rollout has two phases. During the settle phase the cloth is compressed
# (excess material) and buckles into wrinkles while the iron is parked away.
# Then compression relaxes so flattened material STAYS flat, and the iron irons.
settle_steps = 100              # steps of settling before the iron engages
comp_ramp = 20                  # steps to ramp compression high -> low
comp_high = 1.5                 # compression during settle (creates wrinkles)
comp_low = 1.25                 # compression during ironing (partly relaxed: keeps
                                # residual wrinkles but weaker re-forming push)
iron_parked_z = ground_height + 1.0   # iron height while parked (no contact)
comp_schedule = scalar()        # per-timestep compression factor, shape (max_steps,)

act = scalar()
mean_y = scalar()

dt = 0.004
learning_rate = 25
use_toi = True

def n_input_states():
    # per object: 3 displacement + 3 velocity features; +2 iron-position features
    # (iron X, Y relative to cloth center). Goal features dropped (no target point).
    return n_sin_waves + 6 * n_objects + 2


def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b,
                                         spring_length, spring_stiffness,
                                         spring_actuation)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_iron_act, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_iron_act).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_iron_act)).place(act)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.dense(ti.i, max_steps).place(iron_pos)
    ti.root.dense(ti.i, max_steps).place(comp_schedule)
    ti.root.place(loss, goal, mean_y)
    ti.root.lazy_grad()


@ti.kernel
def advance_iron(t: ti.i32):
    # Option B two-phase: during the settle phase the iron is parked high (no
    # contact) so the cloth can buckle. Afterwards it drops to plate height and
    # integrates its horizontal motion from nn2's velocity command (learned).
    for _ in range(1):
        if t <= settle_steps:
            # parked above the cloth CENTRE so that when it drops at the start of
            # the iron phase it lands in contact (nonzero gradient from step one).
            iron_pos[t] = ti.Vector([0.4, 0.175, iron_parked_z])
        else:
            new_x = iron_pos[t - 1][0] + act[t - 1, 0] * iron_speed * dt
            new_y = iron_pos[t - 1][1] + act[t - 1, 1] * iron_speed * dt
            iron_pos[t] = ti.Vector([new_x, new_y, iron_z])


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = c / n_objects


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        # 10 features
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(spring_omega * t * dt + 2 * math.pi / n_sin_waves * j)
        # per object: 3 displacement features + 3 velocity features
        # (ti.static so the kernel is straight-line — autodiff forbids mixing a
        # real for-loop with the top-level iron-feature/bias statements below)
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            actuation += weights1[i, j * 6 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 2] * offset[2] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 3] * v[t, j][0] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 4] * v[t, j][1] * 0.05
            actuation += weights1[i, j * 6 + n_sin_waves + 5] * v[t, j][2] * 0.05
        # iron position features (2): iron X, Y relative to cloth center
        actuation += weights1[i, n_objects * 6 + n_sin_waves] * (iron_pos[t][0] - center[t][0])
        actuation += weights1[i, n_objects * 6 + n_sin_waves + 1] * (iron_pos[t][1] - center[t][1])
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    # Output a horizontal velocity command for the iron (X, Y).
    for i in range(n_iron_act):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        act[t, i] = actuation


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        # Phase 2: cloth is passive (no muscles). Option B: compression comes
        # from a per-timestep schedule (high while settling -> low while ironing)
        # so wrinkles form, then relaxed cloth holds its flattened shape.
        target_length = spring_length[i] * comp_schedule[t]
        impulse = dt * (length -
                        target_length) * spring_stiffness[i] / length * dist

        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        if new_x[2] < ground_height and old_v[2] < -1e-4:
            toi = -(old_x[2] - ground_height) / old_v[2]
            new_v = ti.Vector([0.0, 0.0, 0.0])
        new_x = old_x + toi * old_v + (dt - toi) * new_v

        # --- iron contact (smooth horizontal falloff for differentiability) ---
        # The controller moves the iron horizontally, so the loss MUST depend
        # continuously on iron X/Y. A hard box gives zero gradient w.r.t. iron
        # X/Y and the controller can't learn. Use a Gaussian weight: ~1 under the
        # plate centre, smoothly -> 0 away from it, nonzero everywhere so the
        # gradient always points the iron toward cloth it should press.
        di = new_x - iron_pos[t]
        iron_bottom = iron_pos[t][2]
        w = ti.exp(-((di[0] / iron_half_x) ** 2 + (di[1] / iron_half_y) ** 2))
        if new_x[2] > iron_bottom:                       # cloth poking above plate
            penetration = new_x[2] - iron_bottom
            new_v[2] -= w * penetration * iron_push * dt   # push down (weighted)
            new_v[0] -= w * penetration * iron_smooth * dt # drag material along -X
            new_x = old_x + dt * new_v                     # reflect push in position
        # -------------------------------------------------------------

        # --- tangential ground friction (pin settled wrinkles) ---
        # Points resting on the ground resist sliding, so a buckle that touches
        # down stays put instead of slowly creeping flat.
        if new_x[2] <= ground_height + 1e-3:
            fr = ti.exp(-dt * friction * 10)
            new_v[0] *= fr
            new_v[1] *= fr
        # ---------------------------------------------------------

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 0.0, 1.0
                                                            ]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_v = old_v
        depth = old_x[2] - ground_height
        if depth < 0 and new_v[2] < 0:
            # friction projection: kill the two horizontal axes and the downward velocity
            new_v[0] = 0
            new_v[1] = 0
            new_v[2] = 0
        new_x = old_x + dt * new_v

        # --- iron contact (smooth horizontal falloff for differentiability) ---
        # The controller moves the iron horizontally, so the loss MUST depend
        # continuously on iron X/Y. A hard box gives zero gradient w.r.t. iron
        # X/Y and the controller can't learn. Use a Gaussian weight: ~1 under the
        # plate centre, smoothly -> 0 away from it, nonzero everywhere so the
        # gradient always points the iron toward cloth it should press.
        di = new_x - iron_pos[t]
        iron_bottom = iron_pos[t][2]
        w = ti.exp(-((di[0] / iron_half_x) ** 2 + (di[1] / iron_half_y) ** 2))
        if new_x[2] > iron_bottom:                       # cloth poking above plate
            penetration = new_x[2] - iron_bottom
            new_v[2] -= w * penetration * iron_push * dt   # push down (weighted)
            new_v[0] -= w * penetration * iron_smooth * dt # drag material along -X
            new_x = old_x + dt * new_v                     # reflect push in position
        # -------------------------------------------------------------

        # --- tangential ground friction (pin settled wrinkles) ---
        if new_x[2] <= ground_height + 1e-3:
            fr = ti.exp(-dt * friction * 10)
            new_v[0] *= fr
            new_v[1] *= fr
        # ---------------------------------------------------------

        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_mean_y(t: ti.i32):
    s = 0.0
    # ti.static unrolls into straight-line code; required here because Taichi
    # autodiff forbids mixing a real for-loop with the trailing assignment.
    for i in ti.static(range(n_objects)):
        s += x[t, i][2]
    mean_y[None] = s / n_objects


@ti.kernel
def compute_loss(t: ti.i32):
    err = 0.0
    # ti.static for autodiff compatibility (see compute_mean_y)
    for i in ti.static(range(n_objects)):
        diff = x[t, i][2] - mean_y[None]
        err += diff * diff
    loss[None] = err / n_objects


window = ti.ui.Window("Mass Spring Robot 3D", (768, 768), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1.0, 1.0, 1.0))
scene = window.get_scene()
camera = ti.ui.Camera()

# Persistent render buffers (sized once the robot is set up).
render_nodes = None        # ti.Vector.field(3) of node positions for the current frame
render_node_colors = None  # ti.Vector.field(3) per-node colors
spring_line_verts = None   # 2 vertices per spring (a, b) for line rendering
ground_verts = None        # 4 corners of the ground quad
ground_indices = None      # 2 triangles
iron_render = ti.Vector.field(3, dtype=real, shape=1)  # iron marker (1 point)


def allocate_render_fields():
    """Allocate ti.ui draw buffers after n_objects / n_springs are known."""
    global render_nodes, render_node_colors, spring_line_verts
    global ground_verts, ground_indices
    render_nodes = ti.Vector.field(3, dtype=real, shape=n_objects)
    render_node_colors = ti.Vector.field(3, dtype=real, shape=n_objects)
    spring_line_verts = ti.Vector.field(3, dtype=real, shape=n_springs * 2)

    ground_verts = ti.Vector.field(3, dtype=real, shape=4)
    ground_indices = ti.field(ti.i32, shape=6)
    # A large quad on the Z = ground_height plane (X/Y horizontal, Z up).
    g = ground_height
    ground_verts[0] = [-1.0, -1.0, g]
    ground_verts[1] = [2.0, -1.0, g]
    ground_verts[2] = [2.0, 2.0, g]
    ground_verts[3] = [-1.0, 2.0, g]
    for k, idx in enumerate([0, 1, 2, 0, 2, 3]):
        ground_indices[k] = idx


@ti.kernel
def fill_render_buffers(t: ti.i32):
    # mean height (Z) for colouring
    mean_z = 0.0
    for i in range(n_objects):
        mean_z += x[t, i][2]
    mean_z /= n_objects

    for i in range(n_objects):
        render_nodes[i] = x[t, i]
        dev = x[t, i][2] - mean_z
        intensity = ti.min(ti.abs(dev) / 0.1, 1.0)
        # red = above mean (not yet flat), blue = below mean, white = at mean
        if dev > 0:
            render_node_colors[i] = ti.Vector([1.0, 1.0 - intensity, 1.0 - intensity])
        else:
            render_node_colors[i] = ti.Vector([1.0 - intensity, 1.0 - intensity, 1.0])

    for i in range(n_springs):
        spring_line_verts[i * 2] = x[t, spring_anchor_a[i]]
        spring_line_verts[i * 2 + 1] = x[t, spring_anchor_b[i]]


def render_frame(t):
    """Draw one frame of the 3D scene (Z-up world)."""
    # Look down at the sheet from above and slightly to one side (Z-up world),
    # so the flat X-Y sheet reads as a table top and folds lift toward the camera.
    camera.position(0.4, -0.5, 1.2)
    camera.lookat(0.4, 0.2, 0.1)
    camera.up(0.0, 0.0, 1.0)
    scene.set_camera(camera)
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light(pos=(0.5, -1.0, 1.5), color=(0.6, 0.6, 0.6))

    fill_render_buffers(t)
    scene.mesh(ground_verts, ground_indices, color=(0.85, 0.85, 0.85), two_sided=True)
    scene.lines(spring_line_verts, width=1.0, color=(0.67, 0.67, 0.67))
    scene.particles(render_nodes, radius=0.012, per_vertex_color=render_node_colors)
    # iron marker — radius roughly matches the physics contact half-width (0.08)
    iron_render[0] = iron_pos[t]
    scene.particles(iron_render, radius=iron_half_x, color=(0.1, 0.1, 0.1))
    canvas.scene(scene)


def forward(output=None, visualize=True):
    # Iron starts parked high above the cloth centre during the settle phase; it
    # drops to plate height once the iron phase begins (see advance_iron).
    iron_pos[0] = [0.4, 0.175, iron_parked_z]

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring/{}/'.format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2

    for t in range(1, total_steps):
        compute_center(t - 1)
        nn1(t - 1)                # reads iron_pos[t-1]
        nn2(t - 1)                # writes act[t-1] = iron velocity command
        apply_spring_force(t - 1)
        advance_iron(t)           # integrate iron_pos[t] from act[t-1]
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

        # 3D rendering below (Z-up world)

        if (t + 1) % interval == 0 and visualize:
            # NaN/Inf guard for debugging blow-ups
            px = float(x[t, 0][0])
            if not math.isfinite(px):
                print(f"NaN/Inf at t={t}, node 0")

            render_frame(t)

            if output:
                window.save_image('mass_spring/{}/{:04d}.png'.format(output, t))
            else:
                window.show()

    loss[None] = 0
    compute_mean_y(steps - 1)
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0, 0.0])


def clear():
    clear_states()


def set_compression_schedule():
    # Option B: build the per-timestep compression factor. High during the
    # settle phase (cloth has excess material -> buckles into wrinkles), then a
    # short linear ramp down to nearly-relaxed during the iron phase so that
    # material the iron flattens has no strong push driving it back up.
    # spring_length stays the TRUE rest length; this schedule scales it per step.
    for t in range(max_steps):
        if t < settle_steps:
            f = comp_high
        elif t < settle_steps + comp_ramp:
            f = comp_high + (comp_low - comp_high) * (t - settle_steps) / comp_ramp
        else:
            f = comp_low
        comp_schedule[t] = f


fixed_perturbation = None  # precomputed Z jitter, identical every rollout


def seed_perturbation(amplitude=0.01):
    # A perfectly flat sheet stays flat even when compressed (no force tips a
    # node up). Seed a tiny upward jitter to break the symmetry so the
    # compressed cloth buckles into wrinkles instead of staying flat.
    #
    # IMPORTANT: the jitter is precomputed ONCE and reused, so every training
    # rollout starts from the SAME wrinkle pattern. Re-randomizing each rollout
    # makes the optimizer chase a moving target (noisy, wandering gradient).
    global fixed_perturbation
    if fixed_perturbation is None:
        rng = random.Random(0)
        fixed_perturbation = [rng.uniform(0, amplitude) for _ in range(n_objects)]
    for i in range(n_objects):
        p = x[0, i]
        x[0, i] = [p[0], p[1], p[2] + fixed_perturbation[i]]


def apply_fold(fold_height=0.1):
    fold_x = 0.375  # center of 12-column grid, between col 5 (x=0.35) and col 6 (x=0.40)
    for i in range(n_objects):
        pos = x[0, i]
        if pos[0] > fold_x:
            x[0, i] = [pos[0], pos[1], pos[2] + fold_height]


flat_positions = None  # saved initial flat layout, for resetting each rollout


def setup_robot(objects, springs):
    global n_objects, n_springs, flat_positions
    n_objects = len(objects)
    n_springs = len(springs)
    allocate_fields()
    allocate_render_fields()

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    # from config file -> x() field vector
    for i in range(n_objects):
        x[0, i] = objects[i]
    flat_positions = [[objects[i][0], objects[i][1], objects[i][2]]
                      for i in range(n_objects)]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]


def reset_cloth():
    # Restore the flat layout and re-seed the buckling perturbation. Called once
    # per rollout so every training iteration starts from the same wrinkled task.
    for i in range(n_objects):
        x[0, i] = flat_positions[i]
    seed_perturbation()


POLICY_PATH = 'policy.npz'


def save_policy(path=POLICY_PATH):
    # Persist the learned controller weights so training isn't lost on exit.
    np.savez(path,
             weights1=weights1.to_numpy(), bias1=bias1.to_numpy(),
             weights2=weights2.to_numpy(), bias2=bias2.to_numpy())
    print(f'Saved policy to {path}')


def load_policy(path=POLICY_PATH):
    # Load a previously trained controller (call after setup_robot, which
    # allocates the weight fields). Returns True if a file was found.
    if not os.path.exists(path):
        print(f'No saved policy at {path}')
        return False
    d = np.load(path)
    weights1.from_numpy(d['weights1']); bias1.from_numpy(d['bias1'])
    weights2.from_numpy(d['weights2']); bias2.from_numpy(d['bias2'])
    print(f'Loaded policy from {path}')
    return True


def optimize(toi):
    global use_toi
    use_toi = toi
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_input_states())) * 2

    for i in range(n_iron_act):
        for j in range(n_hidden):
            weights2[i, j] = np.random.randn() * math.sqrt(
                2 / (n_hidden + n_iron_act)) * 3

    # Build the compression schedule once (settle high -> iron-phase low).
    set_compression_schedule()

    losses = []
    for iter in range(options.iters):
        reset_cloth()   # flat layout + fresh buckling perturbation each rollout
        clear()
        # Render kernels are not autodiff-safe and must not run inside the Tape,
        # so the training rollout never visualizes. After training, the caller
        # can replay forward(visualize=True) outside the Tape to watch the result.
        with ti.ad.Tape(loss):
            forward(visualize=False)

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                total_norm_sqr += weights1.grad[i, j]**2
            total_norm_sqr += bias1.grad[i]**2

        for i in range(n_iron_act):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j]**2
            total_norm_sqr += bias2.grad[i]**2

        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        gradient_clip = 0.2
        scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_iron_act):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]

        print(f"Iteration {iter} loss value: {loss[None]}")
        losses.append(loss[None])

    return losses

class Options:
    robot_id = 4 # 1 == A, 2 == B, 3 == C
    task = "plot"   # "train" | "run_saved_weights" | "view" | "plot"
    iters = 100

options = Options()

def main():
    setup_robot(*robots[options.robot_id]())
    if options.task == 'view':
        set_compression_schedule()
        while window.running:
            reset_cloth()
            clear()
            forward(visualize=True)
    elif options.task == 'run_saved_weights':
        # Load a previously trained policy and replay it — no retraining.
        if not load_policy():
            print('Train first (task="train") to create a saved policy.')
            return
        set_compression_schedule()
        while window.running:
            reset_cloth()
            clear()
            forward(visualize=True)
    elif options.task == 'plot':
        # Train once and plot the loss curve (flatness vs iteration).
        losses = optimize(toi=True)
        save_policy()
        np.save('losses.npy', np.array(losses))
        plt.plot(losses, 'b-')
        plt.xlabel('iteration')
        plt.ylabel('loss (cloth Z-variance / flatness)')
        plt.title('Ironing loss over training')
        plt.grid(True, alpha=0.3)
        plt.savefig('loss_curve.png', dpi=120)
        print('Saved loss_curve.png and losses.npy')
        plt.show()
    else:  # 'train': learn the iron path (no viz during training), then replay
        optimize(toi=True)
        save_policy()   # persist weights so the training run isn't lost on exit
        # Replay the trained controller with rendering, outside the Tape, so you
        # can watch the learned iron motion repeatedly.
        set_compression_schedule()
        while window.running:
            reset_cloth()
            clear()
            forward(visualize=True)


if __name__ == '__main__':
    main()