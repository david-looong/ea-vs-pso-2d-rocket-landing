"""2D rigid-body rocket physics with quadratic atmospheric drag and O-U wind."""

import numpy as np
from dataclasses import dataclass
from config import ROCKET_RADIUS, ROCKET_BODY_HEIGHT


@dataclass
class RocketState:
    """Full state of the rocket at a single time-step."""
    x: float = 0.0
    y: float = 200.0
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0   # angle from vertical (rad); 0 = upright, + = tilted right
    omega: float = 0.0   # angular velocity (rad/s)
    fuel: float = 1.0    # normalized remaining fuel [0, 1]

    def copy(self):
        return RocketState(self.x, self.y, self.vx, self.vy,
                           self.theta, self.omega, self.fuel)


class WindModel:
    """Ornstein–Uhlenbeck process for slowly varying horizontal wind."""

    def __init__(self, ou_theta: float = 0.3, max_wind: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.ou_theta = ou_theta
        self.max_wind = max_wind
        self.rng = rng or np.random.default_rng()
        # σ chosen so 3σ of the stationary distribution ≈ max_wind
        self.ou_sigma = (max_wind * np.sqrt(2.0 * ou_theta) / 3.0
                         if max_wind > 0 else 0.0)
        self.speed = 0.0

    def reset(self):
        self.speed = 0.0

    def step(self, dt: float) -> float:
        if self.ou_sigma > 0:
            self.speed += (self.ou_theta * (-self.speed) * dt
                           + self.ou_sigma * np.sqrt(dt) * self.rng.standard_normal())
        return self.speed


# ── ground-contact parameters ──────────────────────────────────────
GROUND_FRICTION = 8.0       # horizontal deceleration on ground (m/s²)
GROUND_ANGULAR_DAMP = 3.0   # angular velocity damping on ground (1/s)
SETTLE_TIME = 2.0           # max seconds the rocket can sit on the ground
TOPPLE_ANGLE = np.radians(45)  # if |θ| exceeds this on the ground → crash
MIN_GROUND_TIME = 0.3       # seconds on ground before at-rest check can fire
SETTLED_SPEED = 0.5         # below this speed + ω the rocket counts as at rest
LANDED_SPEED = 8.0          # max total speed at final rest to count as landed
LANDED_TILT = np.radians(25)   # max tilt at final rest to count as landed


class RocketSim:
    """Full 2D rocket-landing simulation (one episode)."""

    def __init__(self, gravity: float, mass: float, moi: float,
                 max_thrust: float, max_gimbal: float, thruster_arm: float,
                 fuel_rate: float, air_density: float, drag_cd_a: float,
                 dt: float, max_time: float, pad_x: float,
                 pad_half_width: float, wind_model: WindModel):
        self.gravity = gravity
        self.mass = mass
        self.moi = moi
        self.max_thrust = max_thrust
        self.max_gimbal = max_gimbal
        self.thruster_arm = thruster_arm
        self.fuel_rate = fuel_rate
        self.air_density = air_density
        self.drag_cd_a = drag_cd_a
        self.dt = dt
        self.max_time = max_time
        self.max_steps = int(max_time / dt)
        self.pad_x = pad_x
        self.pad_half_width = pad_half_width
        self.wind = wind_model

        self.state = RocketState()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.landed = False
        self.on_ground = False
        self.ground_time = 0.0
        self.impact_speed = 0.0
        self.current_wind = 0.0

    def reset(self, initial_state: RocketState) -> RocketState:
        self.state = initial_state.copy()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.landed = False
        self.on_ground = False
        self.ground_time = 0.0
        self.impact_speed = 0.0
        self.current_wind = 0.0
        self.wind.reset()
        return self.state

    def step(self, throttle: float, gimbal: float) -> tuple[bool, dict]:
        """Advance physics by one time-step.

        Args
        ----
        throttle : float in [0, 1]  — normalised thrust level
        gimbal   : float in [-1, 1] — normalised gimbal angle

        Returns
        -------
        (done, info_dict)
        """
        s = self.state
        throttle = float(np.clip(throttle, 0.0, 1.0))
        gimbal_rad = float(np.clip(gimbal, -1.0, 1.0)) * self.max_gimbal

        # --- thrust (cut entirely once on the ground) ---
        if s.fuel <= 0 or self.on_ground:
            thrust_force = 0.0
            actual_throttle = 0.0
        else:
            thrust_force = throttle * self.max_thrust
            actual_throttle = throttle

        fire_angle = s.theta + gimbal_rad
        sin_fa, cos_fa = np.sin(fire_angle), np.cos(fire_angle)
        thrust_ax = (thrust_force / self.mass) * sin_fa
        thrust_ay = (thrust_force / self.mass) * cos_fa

        # --- quadratic drag: F = -0.5 ρ |v| v Cd_A ---
        speed_sq = s.vx * s.vx + s.vy * s.vy
        if speed_sq > 1e-12:
            speed = np.sqrt(speed_sq)
            drag_k = 0.5 * self.air_density * self.drag_cd_a * speed / self.mass
            drag_ax = -drag_k * s.vx
            drag_ay = -drag_k * s.vy
        else:
            drag_ax = drag_ay = 0.0

        # --- wind (horizontal force) ---
        w = self.wind.step(self.dt)
        self.current_wind = w
        wind_force = 0.5 * self.air_density * self.drag_cd_a * w * abs(w)
        wind_ax = wind_force / self.mass

        # --- totals ---
        ax = thrust_ax + drag_ax + wind_ax
        ay = thrust_ay + drag_ay - self.gravity

        # gimbal torque → angular acceleration
        alpha = -(thrust_force * np.sin(gimbal_rad) * self.thruster_arm) / self.moi

        # --- ground-contact physics ---
        if self.on_ground:
            # Normal force: prevent falling through the ground
            if ay < 0:
                ay = 0.0
            # Ground friction bleeds horizontal speed
            if abs(s.vx) > 0.01:
                friction_a = min(GROUND_FRICTION, abs(s.vx) / self.dt)
                ax -= np.sign(s.vx) * friction_a

            # Rocking torque (OBB):
            # The tilted rocket pivots on its lower base corner.  Gravity
            # pulls the CG down, creating a torque around the pivot:
            #   α = (m·g / I) · (h_cg·sin θ − R·sign θ)
            # Within the stable range (|h_cg·sin θ| < R) this is a
            # restoring torque → wobble.  Beyond it, it topples.
            h_cg = ROCKET_BODY_HEIGHT / 2
            if abs(s.theta) > 1e-6:
                alpha += ((self.mass * self.gravity / self.moi)
                          * (h_cg * np.sin(s.theta)
                             - ROCKET_RADIUS * np.sign(s.theta)))

            # Angular damping from ground-contact friction
            alpha -= GROUND_ANGULAR_DAMP * s.omega

        # --- semi-implicit Euler integration ---
        new_vx = s.vx + ax * self.dt
        new_vy = s.vy + ay * self.dt
        new_x = s.x + new_vx * self.dt
        new_y = s.y + new_vy * self.dt
        new_omega = s.omega + alpha * self.dt
        new_theta = s.theta + new_omega * self.dt
        new_fuel = max(0.0, s.fuel - actual_throttle * self.fuel_rate * self.dt)

        # --- OBB ground collision ---
        # The lowest point of the rotated body is a base corner:
        #   lowest_y = y - ROCKET_RADIUS * |sin(θ)|
        # Ground contact when that reaches y = 0.
        corner_drop = ROCKET_RADIUS * abs(np.sin(new_theta))
        lowest_y = new_y - corner_drop

        if lowest_y <= 0 and not self.on_ground:
            self.on_ground = True
            self.ground_time = 0.0
            self.impact_speed = np.sqrt(new_vx ** 2 + new_vy ** 2)

        if self.on_ground:
            # Keep the lowest corner at or above y = 0
            new_y = max(new_y, corner_drop)
            if new_vy < 0:
                new_vy = 0.0

        self.state = RocketState(new_x, new_y, new_vx, new_vy,
                                 new_theta, new_omega, new_fuel)
        self.time += self.dt
        self.steps += 1

        info = {
            "wind": w,
            "throttle": actual_throttle,
            "gimbal_rad": gimbal_rad,
            "thrust": thrust_force,
        }

        # --- termination ---
        if self.on_ground:
            self.ground_time += self.dt
            ground_speed = np.sqrt(new_vx ** 2 + new_vy ** 2)
            h_cg = ROCKET_BODY_HEIGHT / 2
            cg_stable = abs(h_cg * np.sin(new_theta)) < ROCKET_RADIUS
            at_rest = (self.ground_time >= MIN_GROUND_TIME
                       and ground_speed < SETTLED_SPEED
                       and abs(new_omega) < 0.1
                       and cg_stable)

            if abs(new_theta) > TOPPLE_ANGLE:
                # Tipped over — crash
                self.done = True
                self.landed = False
                info["result"] = "crashed"
            elif at_rest or self.ground_time >= SETTLE_TIME:
                # Settled (or ran out of settling time) — evaluate landing
                self.done = True
                on_pad = abs(new_x - self.pad_x) <= self.pad_half_width
                upright = abs(new_theta) < LANDED_TILT
                slow = ground_speed < LANDED_SPEED
                self.landed = on_pad and upright and slow
                info["result"] = "landed" if self.landed else "crashed"
        elif self.steps >= self.max_steps:
            self.done = True
            info["result"] = "timeout"
        elif abs(new_x) > 500 or new_y > 600:
            self.done = True
            info["result"] = "out_of_bounds"

        return self.done, info
