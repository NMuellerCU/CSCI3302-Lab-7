import json, math, os
import numpy as np
from controller import Robot, Supervisor
import heapq
import matplotlib.pyplot as plt
# ══════════════════════════════════════════════════════════════════════════════
# PR2 ARM KINEMATICS  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

# DH parameters for the PR2 right arm [a, d, alpha, theta_offset]
# Joints: shoulder_pan, shoulder_lift, upper_arm_roll, elbow_flex, wrist_roll
PR2_RIGHT_ARM_DH = [
    [0.0,   0.0, -math.pi/2,  0.0],   # r_shoulder_pan_joint
    [0.0,   0.0,  math.pi/2,  0.0],   # r_shoulder_lift_joint
    [0.4,   0.0, -math.pi/2,  0.0],   # r_upper_arm_roll_joint  (0.4 m upper arm)
    [0.0,   0.0,  math.pi/2,  0.0],   # r_elbow_flex_joint
    [0.321, 0.0,  0.0,        0.0],   # r_wrist_roll_joint      (0.321 m forearm)
]

# Physical joint limits [min_rad, max_rad]
# NOTE: r_elbow_flex is NEGATIVE when bent (0.0 = fully extended)
PR2_JOINT_LIMITS = [
    (-2.1353,  0.5646),   # shoulder_pan
    (-0.5236,  1.3963),   # shoulder_lift
    (-3.9000,  0.8000),   # upper_arm_roll
    (-2.1213,  0.0000),   # elbow_flex  ← negative means bent!
    (-6.2832,  6.2832),   # wrist_roll  (continuous)
]

GRIPPER_OFFSET = 0.18    # metres from wrist to gripper fingertip along Z
ARM_BASE_XY    = np.array([-0.05, -0.188])   # right shoulder in robot body frame (x,y)
ARM_BASE_Z     = 1.07    # shoulder height (metres) when torso = 0.33 m
MAP_SIZE = [14, 14]
MAP_CENTER = [7, 7]
# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION CONSTANTS  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

MAX_WHEEL_SPEED  = 6.0   # rad/s  — maximum wheel angular velocity
ROBOT_RADIUS     = 0.335 # m      — PR2 base half-width (for collision check)

# Potential field tuning (you may adjust these in your TODO section)
K_ATT    = 1.5   # attractive gain
K_REP    = 0.8   # repulsive gain
D0       = 1.2   # influence radius (m) — obstacles beyond this are ignored
STUCK_THRESHOLD = 0.03   # m — if robot moves less than this, it may be stuck


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDED HELPER — DH transform  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def _dh(a, d, alpha, theta):
    """
    Returns the 4×4 Denavit-Hartenberg homogeneous transform for one joint.

    Parameters
    ----------
    a     : link length  (metres)
    d     : link offset  (metres)
    alpha : twist angle  (radians)
    theta : joint angle  (radians) — this is what you solve for in IK
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,  sa,     ca,    d   ],
        [0.,  0.,     0.,    1.  ],
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDED HELPER — coordinate frames  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def world_to_base(world_pos, robot_x, robot_y, robot_yaw):
    """
    Convert a world-frame [x, y, z] position into the PR2 right-arm base frame.

    The arm base frame has:
      +X = forward (in the direction the robot faces)
      +Y = left
      +Z = up

    Parameters
    ----------
    world_pos  : array-like [x, y, z] in world coordinates
    robot_x    : robot base X in world frame
    robot_y    : robot base Y in world frame
    robot_yaw  : robot heading in radians (0 = facing +X, π/2 = facing +Y)

    Returns
    -------
    np.ndarray shape (3,) — target in arm base frame
    """
    cy, sy = math.cos(robot_yaw), math.sin(robot_yaw)
    # Shoulder position in world frame
    sx  = robot_x + cy * ARM_BASE_XY[0] - sy * ARM_BASE_XY[1]
    sy_ = robot_y + sy * ARM_BASE_XY[0] + cy * ARM_BASE_XY[1]
    dx, dy = world_pos[0] - sx, world_pos[1] - sy_
    dz     = world_pos[2] - ARM_BASE_Z
    # Rotate into body frame
    return np.array([cy*dx + sy*dy,  -sy*dx + cy*dy,  dz])


def base_to_world(base_pos, robot_x, robot_y, robot_yaw):
    """Inverse of world_to_base — converts arm base frame → world frame."""
    cy, sy = math.cos(robot_yaw), math.sin(robot_yaw)
    sx  = robot_x + cy * ARM_BASE_XY[0] - sy * ARM_BASE_XY[1]
    sy_ = robot_y + sy * ARM_BASE_XY[0] + cy * ARM_BASE_XY[1]
    bx, by, bz = base_pos
    wx = sx + cy*bx - sy*by
    wy = sy_ + sy*bx + cy*by
    wz = ARM_BASE_Z + bz
    return np.array([wx, wy, wz])

#TODO Implement
def forward_kinematics(q):
    raise NotImplementedError("TODO 1: Implement forward_kinematics(q)")


#TODO Implement
def compute_jacobian(q):
    raise NotImplementedError("TODO 2: Implement compute_jacobian(q)")


#TODO Implement
def gradient_descent_ik(target, q0=None, alpha=0.5, tol=0.008, max_iter=4000):
    raise NotImplementedError("TODO 3: Implement gradient_descent_ik(...)")


#TODO Implement

def compute_potential_field(robot_x, robot_y, goal_x, goal_y, lidar_ranges, lidar_fov):
    raise NotImplementedError("TODO 4: Implement compute_potential_field(...)")

# ALL CODE HERE IS TAKEN FROM hw4_template
def h_euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(grid, start, goal, heuristic):
    start = start
    goal = goal
    g = {start: 0.0}
    came_from = {}
    heap = [(heuristic(start, goal), 0, start)]
    expanded = []
    eid = 0
    while heap:
        _, _, current = heapq.heappop(heap)
        if current in [e for e in expanded]:
            continue
        expanded.append(current)
        if current == goal:
            return reconstruct_path(came_from, goal), expanded, g[goal]
        for nb in neighbors_8(*current, grid):
            mc = np.sqrt((nb[0] - current[0])**2 +
                         (nb[1] - current[1])**2)
            tentative_g = g[current] + mc
            if tentative_g < g.get(nb, float('inf')):
                g[nb] = tentative_g
                came_from[nb] = current
                eid += 1
                heapq.heappush(heap, (tentative_g + heuristic(nb, goal),
                                      eid, nb))
    return None, expanded, float('inf')
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def neighbors_8(r, c, grid):
    """Return free 8-connected neighbors of cell (r, c)."""
    rows, cols = grid.shape
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                yield (nr, nc)
# END OF CODE USED FROM hw4_template
def grid_to_world(cell, MAP_CENTER = MAP_CENTER, resolution = 0.05):
    row, column = cell
    x_c, y_c = MAP_CENTER
    x = (column + 0.5)*resolution - x_c
    y = (row + 0.5)*resolution - y_c
    return [x, y]
def object_to_grid(grid, position, obj_to_world, size, resolution, robot_radius):
    radius_inflated = robot_radius*1.05
    
    wx = size[0]/2 + radius_inflated
    wy = size[1]/2 + radius_inflated
    
    pos_world_x = position[0] + obj_to_world[0]
    pos_world_y = position[1] + obj_to_world[1]
    
    min_x = pos_world_x - wx
    max_x = pos_world_x + wx
    
    min_y = pos_world_y - wy
    max_y = pos_world_y + wy
    
    min_x_grid = int(min_x/resolution)
    max_x_grid = int(max_x/resolution)
    
    min_y_grid = int(min_y/resolution)
    max_y_grid = int(max_y/resolution)
    
    min_x_grid = max(0, min_x_grid)
    max_x_grid = min(grid.shape[0] - 1,max_x_grid)
    
    min_y_grid = max(0, min_y_grid)
    max_y_grid = min(grid.shape[1] - 1, max_y_grid)
    
    grid[min_y_grid:max_y_grid+1, min_x_grid:max_x_grid+1] = 1

# def plot_grid(grid, start, goal, path=None, expanded=None, title=""):
#     rows, cols = grid.shape
#     aspect = rows / cols
#     fig_w = 7
#     fig_h = max(3, fig_w * aspect + 1.5)

#     fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#     ax.imshow(grid, cmap="Greys", origin="upper")

#     # expanded nodes
#     if expanded is not None and len(expanded) > 0:
#         er, ec = zip(*expanded)
#         ax.scatter(ec, er, c="dodgerblue", s=10, alpha=0.5, label="Expanded")

#     # A* path
#     if path is not None and len(path) > 0:
#         pr, pc = zip(*path)
#         ax.plot(pc, pr, c="red", linewidth=2, label="A* Path")

#     # start and goal
#     ax.plot(start[1], start[0], "go", markersize=10, label="Start")
#     ax.plot(goal[1], goal[0], "r*", markersize=14, label="Goal")

#     ax.set_title(title if title else "Grid")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig("grid.png")
#     plt.close()

def plot_grid(grid, start, goal, path=None, expanded=None, title=""):
    import numpy as np

    # transform grid for display:
    # 1) flip horizontally
    # 2) rotate 90 deg left
    disp_grid = np.rot90(np.fliplr(grid), 1)

    rows, cols = disp_grid.shape
    aspect = rows / cols
    fig_w = 7
    fig_h = max(3, fig_w * aspect + 1.5)

    # original grid shape
    orig_rows, orig_cols = grid.shape

    def transform_point(r, c):
        # step 1: horizontal flip
        c1 = orig_cols - 1 - c
        r1 = r

        # step 2: rotate 90 deg left
        r2 = orig_cols - 1 - c1
        c2 = r1

        return r2, c2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(disp_grid, cmap="Greys", origin="upper")

    # expanded nodes
    if expanded is not None and len(expanded) > 0:
        transformed_expanded = [transform_point(r, c) for r, c in expanded]
        er, ec = zip(*transformed_expanded)
        ax.scatter(ec, er, c="dodgerblue", s=10, alpha=0.5, label="Expanded")

    # A* path
    if path is not None and len(path) > 0:
        transformed_path = [transform_point(r, c) for r, c in path]
        pr, pc = zip(*transformed_path)
        ax.plot(pc, pr, c="red", linewidth=2, label="A* Path")

    # start and goal
    sr, sc = transform_point(start[0], start[1])
    gr, gc = transform_point(goal[0], goal[1])

    ax.plot(sc, sr, "go", markersize=10, label="Start")
    ax.plot(gc, gr, "r*", markersize=14, label="Goal")

    ax.set_title(title if title else "Grid")
    ax.legend()
    plt.tight_layout()
    plt.savefig("grid.png")
    plt.close()
    

def build_grid(env_map,start, goal, MAP_SIZE = MAP_SIZE, MAP_CENTER = MAP_CENTER, resolution = 0.05, robot_radius= ROBOT_RADIUS):
    #resolution is m/index
    MAP_X, MAP_Y = MAP_SIZE # [m]
    MAP_CENTER_X, MAP_CENTER_Y = MAP_CENTER
    
    start_col = int((start[0] + MAP_CENTER_X)/resolution)
    start_row = int((start[1] + MAP_CENTER_Y)/resolution)
    start_pos = (start_row, start_col)
    
    goal_col = int((goal[0] + MAP_CENTER_X)/resolution)
    goal_row = int((goal[1] + MAP_CENTER_Y)/resolution)
    goal_pos = (goal_row, goal_col)
    
    rows = int(MAP_X/resolution)
    cols = int(MAP_Y/resolution)
    # grid_center = int(MAP_CENTER/resolution)
    grid = np.zeros((rows,cols), dtype=int)
    for obs in env_map["obstacles"]:
        # name = obs["def_name"]
        pos = obs["position"]
        size = obs["size"]
        object_to_grid(grid,pos,MAP_CENTER,size,resolution,robot_radius)

    return grid, start_pos, goal_pos      
            
            
        #     size = 
        #     position_world = 
        # elif "wall_inner" in name:
        
        # elif "pick_table" in name:
            
        # elif "wall" in name:
        #     if "wall_n" in name:
                
        #     elif "wall_e" in name:
        #     elif "wall_s" in name:
        #     elif "wall_w" in name:
                
    
 
    # obstacles = env_map.get("obstacles", [])


    
    
#TODO Implement
def pick_object(pr2, obj_data):
    raise NotImplementedError("TODO 5: Implement pick_object(pr2, obj_data)")


#TODO Implement

def place_objects(pr2, place_zone):
    raise NotImplementedError("TODO 6: Implement place_objects(pr2, place_zone)")


# ══════════════════════════════════════════════════════════════════════════════
# PR2 CONTROLLER CLASS  — provided, do not modify
# ══════════════════════════════════════════════════════════════════════════════

class PR2Controller:
    """
    Hardware abstraction layer for the PR2 in Webots.
    Exposes the devices you need — read the docstrings before using.
    """

    TIME_STEP = 32   # ms

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self):
        self.robot = Robot()
        self.ts    = int(self.robot.getBasicTimeStep())

        # Dead-reckoning state (updated by supervisor pose file)
        self._x   = 0.0
        self._y   = 1.0
        self._yaw = -math.pi / 2   # PR2 starts facing south (-Y)
        self._collision_count = 0

        self._setup_devices()
        self._enable_devices()

        # Initial safe pose
        self.set_right_arm([0., 1.35, 0., -2.2, 0.])
        self.set_left_arm( [0., 1.35, 0., -2.2, 0.])
        self.open_gripper(right=True)
        self.open_gripper(right=False)
        self.set_torso(0.33)

    # ── Device setup (internal) ───────────────────────────────────────────────

    def _setup_devices(self):
        G = self.robot.getDevice
        self._wm = [G(n) for n in [
            "fl_caster_l_wheel_joint","fl_caster_r_wheel_joint",
            "fr_caster_l_wheel_joint","fr_caster_r_wheel_joint",
            "bl_caster_l_wheel_joint","bl_caster_r_wheel_joint",
            "br_caster_l_wheel_joint","br_caster_r_wheel_joint"]]
        self._ws  = [m.getPositionSensor() if m else None for m in self._wm]
        self._cm  = [G(n) for n in [
            "fl_caster_rotation_joint","fr_caster_rotation_joint",
            "bl_caster_rotation_joint","br_caster_rotation_joint"]]
        self._cs  = [m.getPositionSensor() if m else None for m in self._cm]
        self._ra  = [G(n) for n in [
            "r_shoulder_pan_joint","r_shoulder_lift_joint",
            "r_upper_arm_roll_joint","r_elbow_flex_joint","r_wrist_roll_joint"]]
        self._ras = [m.getPositionSensor() if m else None for m in self._ra]
        self._la  = [G(n) for n in [
            "l_shoulder_pan_joint","l_shoulder_lift_joint",
            "l_upper_arm_roll_joint","l_elbow_flex_joint","l_wrist_roll_joint"]]
        self._las = [m.getPositionSensor() if m else None for m in self._la]
        self._rf  = G("r_finger_gripper_motor::l_finger")
        self._lf  = G("l_finger_gripper_motor::l_finger")
        self._rfs = self._rf.getPositionSensor() if self._rf else None
        self._lfs = self._lf.getPositionSensor() if self._lf else None
        self._rcl = G("r_gripper_l_finger_tip_contact_sensor")
        self._rcr = G("r_gripper_r_finger_tip_contact_sensor")
        self._tor = G("torso_lift_joint")
        self._tors= G("torso_lift_joint_sensor")
        self._imu = G("imu_sensor")
        self._lid = G("base_laser")

    def _enable_devices(self):
        ts = self.ts
        for s in self._ws:
            if s: s.enable(ts)
        for m in self._wm:
            if m: m.setPosition(float('inf')); m.setVelocity(0.)
        for s in self._cs + self._ras + self._las:
            if s: s.enable(ts)
        for s in [self._rfs, self._lfs, self._tors]:
            if s: s.enable(ts)
        if self._imu: self._imu.enable(ts)
        if self._lid: self._lid.enable(ts)
        if self._rcl: self._rcl.enable(ts)
        if self._rcr: self._rcr.enable(ts)

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self):
        """Advance simulation by one time step. Returns False when sim ends."""
        alive = self.robot.step(self.ts) != -1
        if alive:
            self._check_collision()
        return alive

    # ── Pose (supervisor-corrected) ───────────────────────────────────────────

    def get_pose(self):
        """
        Return (x, y, yaw) in world frame.
        Reads robot_pose.json written by the supervisor for accuracy.
        Falls back to IMU + dead-reckoning if file not available.
        """
        for path in ["robot_pose.json",
                     "../lab7_supervisor/robot_pose.json",
                     "../../robot_pose.json"]:
            try:
                with open(path) as f:
                    d = json.load(f)
                    self._x, self._y, self._yaw = d["x"], d["y"], d["yaw"]
                    return self._x, self._y, self._yaw
            except Exception:
                pass
        # IMU fallback for yaw
        if self._imu:
            self._yaw = self._imu.getRollPitchYaw()[2]
        return self._x, self._y, self._yaw

    # ── Lidar ─────────────────────────────────────────────────────────────────

    def get_lidar(self):
        """
        Return (ranges, fov) from the base laser.
        ranges : list of floats (metres); may contain float('inf') or nan
        fov    : total field of view in radians
        """
        if not self._lid:
            return [], 0.0
        return self._lid.getRangeImage(), self._lid.getFov()

    # ── Collision counting ────────────────────────────────────────────────────

    def _check_collision(self):
        if not self._lid: return
        ranges = self._lid.getRangeImage()
        if not ranges: return
        n = len(ranges); fov = self._lid.getFov()
        c = n // 2; hw = int(0.20 / fov * n)
        for i in range(max(0, c-hw), min(n-1, c+hw)+1):
            r = ranges[i]
            if not (math.isnan(r) or math.isinf(r)) and 0.05 < r < 0.25:
                self._collision_count += 1
                break   # count once per step

    def get_collision_count(self):
        """Returns total collision count (used for grading penalty)."""
        return self._collision_count

    # ── Wheel control ─────────────────────────────────────────────────────────

    def set_wheel_speeds(self, left_speed, right_speed):
        """
        Set differential drive wheel velocities (rad/s).
        All four left wheels get left_speed; all four right wheels get right_speed.
        Casters must be in forward-facing configuration (use rotate_in_place for turns).

        Parameters
        ----------
        left_speed  : float in [-MAX_WHEEL_SPEED, MAX_WHEEL_SPEED]
        right_speed : float in [-MAX_WHEEL_SPEED, MAX_WHEEL_SPEED]
        """
        ls = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left_speed))
        rs = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right_speed))
        speeds = [ls, ls, rs, rs, ls, ls, rs, rs]
        for m, v in zip(self._wm, speeds):
            if m: m.setVelocity(v)

    def stop(self):
        """Stop all wheels."""
        for m in self._wm:
            if m: m.setVelocity(0.)

    def rotate_in_place(self, angle_rad):
        """
        Rotate the robot in place by angle_rad (positive = CCW).
        Blocks until rotation is complete (encoder-based).
        """
        self.stop()
        self._set_casters(3*math.pi/4, math.pi/4,
                         -3*math.pi/4,-math.pi/4, wait=True)
        spd = 2.0 if angle_rad > 0 else -2.0
        for m in self._wm:
            if m: m.setVelocity(spd)
        expected = abs(angle_rad) * 0.5 * (0.4492 + 0.098)
        s0 = self._ws[0]
        if s0:
            init = s0.getValue()
            while True:
                t = abs(0.08 * (s0.getValue() - init))
                if t >= expected: break
                if expected - t < 0.02:
                    for m in self._wm:
                        if m: m.setVelocity(0.15 * spd)
                self.step()
        self._set_casters(0., 0., 0., 0., wait=True)
        self.stop()
        if self._imu:
            self._yaw = self._imu.getRollPitchYaw()[2]

    # ── Torso ─────────────────────────────────────────────────────────────────

    def set_torso(self, height, wait=True):
        print("set_torso called")
        print("torso motor:", self._tor)
        print("torso sensor:", self._tors)

        if self._tor:
            self._tor.setPosition(max(0., min(0.33, height)))

        if wait and self._tors:
            for i in range(600):
                if not self.step():
                    print("step returned false during torso")
                    return
                val = self._tors.getValue()
                if i % 20 == 0:
                    print("torso sensor value:", val, "target:", height)
                if abs(val - height) < 0.01:
                    print("torso reached target")
                    break

    # ── Arm control ───────────────────────────────────────────────────────────

    def set_right_arm(self, q, wait=True):
        """
        Command the right arm to joint angles q (length-5 list/array, radians).
        Clamps to PR2_JOINT_LIMITS automatically.
        If wait=True, blocks until all joints reach their targets.
        """
        self._set_arm(self._ra, self._ras, q, wait)

    def set_left_arm(self, q, wait=True):
        """Same as set_right_arm but for the left arm."""
        self._set_arm(self._la, self._las, q, wait)

    def get_right_arm_q(self):
        """Return current right arm joint angles as np.ndarray shape (5,)."""
        return np.array([s.getValue() if s else 0. for s in self._ras])

    # ── Gripper ───────────────────────────────────────────────────────────────

    def open_gripper(self, right=True, wait=True):
        """
        Open the specified gripper fully.
        right=True  → right gripper (default)
        right=False → left gripper
        """
        m = self._rf if right else self._lf
        s = self._rfs if right else self._lfs
        if m: m.setPosition(0.5)
        if wait and s:
            for _ in range(300):
                if not self.step(): return
                if abs(s.getValue() - 0.5) < 0.02: break

    def close_gripper(self, right=True, torque=20., wait=True):
        """
        Close the specified gripper until contact or fully closed.
        torque : gripping force in N·m (default 20 N·m)
        """
        m  = self._rf  if right else self._lf
        s  = self._rfs if right else self._lfs
        cl = self._rcl if right else None
        cr = self._rcr if right else None
        if not m: return
        m.setPosition(0.)
        if wait:
            for _ in range(500):
                if not self.step(): return
                contact = cl and cr and cl.getValue() > 0 and cr.getValue() > 0
                closed  = s and abs(s.getValue()) < 0.01
                if contact or closed:
                    if s:
                        m.setAvailableTorque(torque)
                        m.setPosition(max(0., s.getValue() * 0.95))
                    break

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_arm(self, motors, sensors, q, wait=True):
        for m, a in zip(motors, q):
            lo, hi = -6.28, 6.28
            if m: m.setPosition(max(lo, min(hi, float(a))))
        if wait:
            for _ in range(900):
                if not self.step(): return
                if all(s is None or abs(s.getValue() - float(a)) < 0.05
                       for s, a in zip(sensors, q)): break

    def _set_casters(self, fl, fr, bl, br, wait=True):
        T = [fl, fr, bl, br]
        if wait:
            self.stop()
            for m in self._wm:
                if m: m.setAvailableTorque(0.)
        for m, t in zip(self._cm, T):
            if m: m.setPosition(t)
        if wait:
            for _ in range(300):
                if all(s is None or abs(s.getValue()-t) < 0.05
                       for s, t in zip(self._cs, T)): break
                self.step()
            for m in self._wm:
                if m: m.setAvailableTorque(0.5)




def load_environment_map():
    print("running func")
    with open('environment_map.json', 'r') as file:
            # json.load() parses the file and returns a dict or list
            print("opened file")
            return json.load(file)
def wrap_to_pi(angle):
    return((angle + np.pi) % (2*np.pi) - np.pi)

def navigate_to_goal(pr2, goal_x, goal_y, goal_yaw, env_map):
    robot_x, robot_y, robot_yaw = pr2.get_pose()
    start_xy = [robot_x, robot_y]
    goal_xy = [goal_x, goal_y]
    
    grid, start_grid, goal_grid = build_grid(env_map, start_xy, goal_xy)
    print("build grid working")
    plot_grid(grid, start_grid, goal_grid, path=None, expanded=None,title="Grid with A* Path")
    path, expanded, cost = astar(grid, start_grid, goal_grid, h_euclidean)
    plot_grid(grid, start_grid, goal_grid, path=path, expanded=expanded,title="Grid with A* Path")
    if path is None:
        print("No path was found")
        return
    waypoints = [grid_to_world(cell) for cell in path]


    # print(f"waypoints:\n {waypoints[:,1]}, {waypoints[:,2]}")
    for wp_x, wp_y in waypoints:
        while pr2.step():
            robot_x, robot_y, robot_yaw = pr2.get_pose()
            # print(f"robot_x: {robot_x} robot_y: {robot_y} robot_yaw: {robot_yaw}")
            delta_x = wp_x - robot_x
            delta_y = wp_y - robot_y
            dist = np.sqrt(delta_x**2 +  delta_y**2)
            dir = np.atan2(delta_y, delta_x)
            # print(f"robot yaw: {round(robot_yaw,3)}, desired dir: {round(dir,3)}")
            heading = wrap_to_pi(dir - robot_yaw)
            if dist < 0.1:
                pr2.stop()
                break
            elif abs(heading) > 0.1:
                pr2.stop()
                pr2.rotate_in_place(heading)
                # sign_heading = np.sign(heading)
                # vL = -sign_heading*0.7*MAX_WHEEL_SPEED
                # vR = sign_heading*0.7*MAX_WHEEL_SPEED
                # pr2.set_wheel_speeds(vL, vR)
                
            else:
                vL = 1*MAX_WHEEL_SPEED
                vR = 1*MAX_WHEEL_SPEED
                pr2.set_wheel_speeds(vL, vR)
                
    while pr2.step():
        robot_x, robot_y, robot_yaw = pr2.get_pose()
        heading = wrap_to_pi(goal_yaw - robot_yaw)
        
        if abs(heading) > 0.02:
            pr2.stop()
            pr2.rotate_in_place(heading)
        else:
            pr2.stop()
            break
        
            
            
        
        
        
        

#Do not modify the main
def main():
    pr2 = PR2Controller()
    env = load_environment_map()

    objects    = env.get("pick_objects",    {})
    nav_goals  = env.get("navigation_goals", {})
    place_zone = env.get("place_zone",      {})

    # ── Pick OBJECT_1 ─────────────────────────────────────────────────────────
    if "OBJECT_1" in objects:
        g1 = nav_goals["OBJECT_1"]
        navigate_to_goal(pr2, g1["position"][0], g1["position"][1],
                         g1["yaw_radians"], env)
        pick_object(pr2, objects["OBJECT_1"])

    # ── Pick OBJECT_2 ─────────────────────────────────────────────────────────
    if "OBJECT_2" in objects:
        g2 = nav_goals["OBJECT_2"]
        navigate_to_goal(pr2, g2["position"][0], g2["position"][1],
                         g2["yaw_radians"], env)
        pick_object(pr2, objects["OBJECT_2"])

    # ── Place both objects ────────────────────────────────────────────────────
    if place_zone:
        pg = place_zone["nav_goal"]
        navigate_to_goal(pr2, pg["position"][0], pg["position"][1],
                         pg["yaw_radians"], env)
        place_objects(pr2, place_zone)

    print(f"\n[Lab7] Done!  Total collisions: {pr2.get_collision_count()}")
    print(f"[Lab7] Collision penalty: -{pr2.get_collision_count() * 5} pts")
    while pr2.step():
        pass


if __name__ == "__main__":
    main()