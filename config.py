"""
All hyperparameters for the 2D Rocket Landing GA.
Edit this file to tune the simulation, controller, GA, novelty, and visualization.
"""
import numpy as np

# ========================= RANDOM SEED =========================
RANDOM_SEED = 42  # Set to None for non-reproducible runs

# ========================= PHYSICS ==============================
GRAVITY = 10.0              # m/s² (base value, subject to curriculum randomization)
ROCKET_MASS = 100.0         # kg
MOMENT_OF_INERTIA = 100.0   # kg·m²
MAX_THRUST = 2500.0         # N (base; gives thrust-to-weight ≈ 2.55)
MAX_GIMBAL_ANGLE = np.radians(15)  # ±15° gimbal range
THRUSTER_ARM = 3.0          # m, distance from center of gravity to nozzle
INITIAL_FUEL = 1.0          # normalized [0, 1]
FUEL_CONSUMPTION = 0.20     # fuel/s at full throttle (~25 s endurance)

# Rocket body geometry (used for OBB collision and visual rendering)
ROCKET_RADIUS = 2.0             # m, half-width of the body
ROCKET_BODY_HEIGHT = 13.0       # m, body height (base to shoulder)
ROCKET_NOSE_HEIGHT = 2.0        # m, nose cone above the body

# Atmospheric drag: F_drag = -0.5 · ρ · |v| · v · Cd_A
AIR_DENSITY = 1.225         # kg/m³
DRAG_CD_A = 0.7             # Cd × reference area (m²), tunable

# Simulation timing
SIM_DT = 1.0 / 60.0        # 60 Hz physics timestep
MAX_SIM_TIME = 30.0         # seconds per episode

# Landing pad
PAD_X = 0.0                 # pad center (m)
PAD_WIDTH = 40.0            # total pad width (m)

# ========================= WIND (Ornstein–Uhlenbeck) ============
WIND_OU_THETA = 0.3         # mean-reversion rate

# ========================= INITIAL CONDITIONS ===================
ALTITUDE_RANGE = (150.0, 250.0)
HORIZONTAL_RANGE = (-200.0, 200.0)
INIT_VY_RANGE = (-50.0, -5.0)      # downward
INIT_VX_RANGE = (-50.0, 10.0)
INIT_ANGLE_RANGE = np.radians(60)   # ±20°

# ========================= CURRICULUM SCHEDULE ==================
# (start_gen, gravity_variation, thrust_variation, max_wind_m/s)
#
# The schedule lets the population learn basic control in a calm, fixed world
# before introducing perturbations that demand robust, adaptive strategies.
CURRICULUM = [
    (0,   0.00, 0.00,  0.0),   # Gen 0–29:   fixed physics, no wind
    (30,  0.10, 0.10,  5.0),   # Gen 30–99:  ±10 % variation, light gusts
    (100, 0.20, 0.20, 15.0),   # Gen 100+:   full randomization
]

NUM_EVAL_TRIALS = 4  # scenarios per genome (averaged for noise reduction)

# ========================= NEURAL NETWORK =======================
NN_LAYERS = [6, 16, 16, 2]  # input → hidden → hidden → output

# Input normalization divisors (keep inputs roughly in [-1, 1])
NORM_X = 200.0
NORM_Y = 300.0
NORM_VX = 50.0
NORM_VY = 50.0
NORM_THETA = np.pi
NORM_OMEGA = 5.0

# ========================= GENETIC ALGORITHM ====================
POPULATION_SIZE = 100
NUM_GENERATIONS = 300
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1         # per-gene probability
MUTATION_SIGMA = 0.1        # Gaussian noise σ
ELITISM_COUNT = 5

# ========================= NOVELTY SEARCH =======================
# Novelty bonus encourages behavioural diversity so the population explores
# many landing strategies instead of collapsing to a single policy.
# Tune NOVELTY_WEIGHT:  too high → chases weirdness, too low → one strategy.
NOVELTY_K = 15              # k nearest neighbors
NOVELTY_WEIGHT = 15.0       # bonus weight (scaled so max bonus ≈ this value)
NOVELTY_ARCHIVE_PROB = 0.05 # probability of archiving each behaviour

# ========================= VISUALIZATION ========================
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
VIS_SCALE = 2.4             # pixels per meter
VIS_FPS = 60
