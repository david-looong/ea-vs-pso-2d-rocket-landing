"""Pygame visualisation for rocket-landing replay after each generation.

Shows all rockets simultaneously: the best genome in full colour, others at
~20% opacity.  The preview continues for ~1 s after the best rocket settles.
"""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from config import ROCKET_RADIUS, ROCKET_BODY_HEIGHT, ROCKET_NOSE_HEIGHT
from controller import NeuralNetwork


# ═══════════════════════════════════════════════════════════════════
#  Exhaust particle system
# ═══════════════════════════════════════════════════════════════════

class ExhaustParticle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x, y, vx, vy, life, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size


class ExhaustSystem:
    """Per-rocket particle emitter for thruster exhaust."""

    HOT_CORE   = np.array([255, 255, 220], dtype=float)
    MID_FLAME  = np.array([255, 180, 50],  dtype=float)
    COOL_TIP   = np.array([255, 60,  15],  dtype=float)
    SMOKE      = np.array([80,  80,  90],  dtype=float)

    def __init__(self):
        self.particles: list[ExhaustParticle] = []
        self._rng = np.random.default_rng()

    def emit(self, sx, sy, throttle, gimbal_rad, theta, scale):
        """Spawn new particles at the nozzle (screen coords)."""
        if throttle < 0.02:
            return

        rng = self._rng
        n = max(1, int(throttle * 8 + rng.uniform(0, 2)))

        fa = theta + gimbal_rad
        sin_f = np.sin(fa)
        cos_f = np.cos(fa)

        base_speed = (40 + throttle * 120) * scale / 60.0

        for _ in range(n):
            spread = rng.uniform(-0.25, 0.25)
            a = fa + spread
            sa, ca = np.sin(a), np.cos(a)

            spd = base_speed * rng.uniform(0.6, 1.3)
            # exhaust goes opposite to thrust direction (downward from nozzle)
            vx = -sa * spd
            vy =  ca * spd

            ox = rng.uniform(-2.5, 2.5) * cos_f
            oy = rng.uniform(-2.5, 2.5) * sin_f

            life = rng.uniform(0.15, 0.4 + throttle * 0.2)
            size = rng.uniform(1.5, 3.0 + throttle * 2.0)
            self.particles.append(
                ExhaustParticle(sx + ox, sy + oy, vx, vy, life, size))

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt * 60
            p.y += p.vy * dt * 60
            p.vx *= 0.96
            p.vy *= 0.96
            p.size *= 0.985
            alive.append(p)
        self.particles = alive

    def draw(self, surface, alpha=255):
        for p in self.particles:
            t = 1.0 - (p.life / p.max_life)
            if t < 0.3:
                clr = self.HOT_CORE + (self.MID_FLAME - self.HOT_CORE) * (t / 0.3)
            elif t < 0.7:
                clr = self.MID_FLAME + (self.COOL_TIP - self.MID_FLAME) * ((t - 0.3) / 0.4)
            else:
                clr = self.COOL_TIP + (self.SMOKE - self.COOL_TIP) * ((t - 0.7) / 0.3)

            a = int(alpha * (1.0 - t * 0.8))
            c = (int(clr[0]), int(clr[1]), int(clr[2]), max(0, min(255, a)))
            sz = max(1, int(p.size * (1.0 - t * 0.5)))
            rect = pygame.Rect(int(p.x) - sz, int(p.y) - sz, sz * 2, sz * 2)
            pygame.draw.rect(surface, c, rect)


# ═══════════════════════════════════════════════════════════════════
#  Renderer
# ═══════════════════════════════════════════════════════════════════

class Renderer:
    """Opens a persistent window and replays episodes between generations."""

    # ── colour palette ──────────────────────────────────────────────
    BG        = (8, 8, 32)
    GROUND    = (50, 42, 35)
    PAD       = (200, 200, 200)
    PAD_MARK  = (255, 80, 80)
    ROCKET    = (215, 218, 225)
    ROCKET_EDGE = (130, 132, 140)
    NOSE      = (180, 182, 190)
    TRAIL     = (80, 140, 255)
    TRAIL_DIM = (40, 70, 130)
    HUD       = (190, 255, 190)
    HUD_DIM   = (120, 120, 120)
    STAR      = (190, 190, 170)
    WIND_CLR  = (100, 200, 255)

    GHOST_ALPHA = 50

    def __init__(self, width: int = 1200, height: int = 800,
                 scale: float = 2.4, fps: int = 60):
        if not HAS_PYGAME:
            raise RuntimeError("pygame is required for visualisation")
        pygame.init()
        pygame.display.set_caption("2D Rocket Landing — Genetic Algorithm")
        self.W = width
        self.H = height
        self.scale = scale
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_lg = pygame.font.Font(None, 30)
        self.font_xl = pygame.font.Font(None, 48)
        self._init_particles()

    # ── public API ──────────────────────────────────────────────────

    def replay_generation(self, best_idx, all_genomes, nn_layers, make_sim,
                          gen_info, normalize_fn, diverse_genomes=None):
        """Replay all genomes simultaneously from one generation.

        Parameters
        ----------
        best_idx      : int — population index of the best genome
        all_genomes   : list[(pop_idx, genome)] — genomes to show
        nn_layers     : list[int]
        make_sim      : callable() → (RocketSim, RocketState)
        gen_info      : dict with keys gen, total_gen, fitness, success_rate
        normalize_fn  : callable(RocketState) → np.ndarray(6,)
        diverse_genomes : list[(label, genome)] or None

        Returns
        -------
        "quit" | "skip" | "done"
        """
        n = len(all_genomes)

        nns = [NeuralNetwork(nn_layers) for _ in range(n)]
        sims = []
        trails = [[] for _ in range(n)]
        exhausts = [ExhaustSystem() for _ in range(n)]
        last_infos = [{"wind": 0, "throttle": 0, "gimbal_rad": 0, "thrust": 0}
                      for _ in range(n)]
        is_best = [False] * n

        for j, (pop_idx, genome) in enumerate(all_genomes):
            nns[j].set_genome(genome)
            sim, initial = make_sim()
            sim.reset(initial)
            sims.append(sim)
            if pop_idx == best_idx:
                is_best[j] = True

        best_j = next(j for j in range(n) if is_best[j])

        # ── simulation phase: step all rockets each frame ──────────
        best_done_ms = None
        POST_REST_MS = 1000

        while True:
            action = self._poll()
            if action == "quit":
                return "quit"
            if action == "skip":
                return "skip"

            dt = 1.0 / max(self.fps, 1)

            any_active = False
            for j in range(n):
                if sims[j].done:
                    exhausts[j].update(dt)
                    continue
                any_active = True
                inp = normalize_fn(sims[j].state)
                throttle, gimbal = nns[j].forward(inp)
                done, info = sims[j].step(throttle, gimbal)
                last_infos[j] = info
                trails[j].append((sims[j].state.x, sims[j].state.y))

                sx, sy = self._w2s(sims[j].state.x, sims[j].state.y)
                exhausts[j].emit(sx, sy,
                                 info.get("throttle", 0.0),
                                 info.get("gimbal_rad", 0.0),
                                 sims[j].state.theta, self.scale)
                exhausts[j].update(dt)

            if sims[best_j].done and best_done_ms is None:
                best_done_ms = 0

            if best_done_ms is not None:
                best_done_ms += self.clock.get_time()

            all_done = not any_active
            best_rested = best_done_ms is not None and best_done_ms >= POST_REST_MS

            if all_done or best_rested:
                break

            self._draw_all(sims, trails, exhausts, last_infos,
                           is_best, best_j, gen_info)
            self.clock.tick(self.fps)

        # ── end-of-episode: show result overlay ────────────────────
        bs = sims[best_j]
        if bs.landed:
            result_txt, result_clr = "LANDED!", (80, 255, 100)
        elif last_infos[best_j].get("result") == "timeout":
            result_txt, result_clr = "TIMEOUT", (255, 200, 60)
        else:
            result_txt, result_clr = "CRASHED", (255, 80, 80)

        wait = 0
        while wait < 1500:
            action = self._poll()
            if action == "quit":
                return "quit"
            if action in ("skip", "next"):
                break
            self._draw_all(sims, trails, exhausts, last_infos,
                           is_best, best_j, gen_info,
                           overlay=result_txt, overlay_clr=result_clr)
            wait += self.clock.tick(self.fps)

        return "done"

    def close(self):
        pygame.quit()

    # ── event handling ──────────────────────────────────────────────

    def _poll(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    return "skip"
                if ev.key == pygame.K_n:
                    return "next"
        return None

    # ── coordinate helpers ──────────────────────────────────────────

    def _w2s(self, wx, wy):
        """World → screen."""
        sx = self.W / 2 + wx * self.scale
        sy = self.H - 30 - wy * self.scale
        return int(sx), int(sy)

    # ── particle pool for wind visualisation ────────────────────────

    def _init_particles(self):
        rng = np.random.default_rng(0)
        self.stars = [(rng.integers(0, self.W), rng.integers(0, int(self.H * 0.7)),
                       rng.integers(1, 3)) for _ in range(100)]
        self.wind_parts = [[rng.uniform(0, self.W),
                            rng.uniform(0, self.H * 0.85),
                            rng.uniform(0.3, 1.0)]
                           for _ in range(80)]

    # ── main composite draw routine ────────────────────────────────

    def _draw_all(self, sims, trails, exhausts, infos, is_best, best_j, gi,
                  overlay=None, overlay_clr=None):
        scr = self.screen
        scr.fill(self.BG)

        # stars
        for sx, sy, r in self.stars:
            pygame.draw.circle(scr, self.STAR, (sx, sy), r)

        # wind particles (use best sim's wind)
        wind = infos[best_j].get("wind", 0.0)
        dt = 1.0 / max(self.fps, 1)
        for p in self.wind_parts:
            p[0] += wind * self.scale * dt * 3
            p[1] += np.random.uniform(-0.3, 0.3)
            if p[0] > self.W:
                p[0] -= self.W
            elif p[0] < 0:
                p[0] += self.W
            if p[1] > self.H * 0.85:
                p[1] = 0.0
            elif p[1] < 0:
                p[1] = self.H * 0.85
            bright = min(200, int(40 + abs(wind) * 12))
            pygame.draw.circle(scr, (bright, bright + 5, bright + 15),
                               (int(p[0]), int(p[1])),
                               1 if abs(wind) < 5 else 2)

        # ground
        gy = self._w2s(0, 0)[1]
        pygame.draw.rect(scr, self.GROUND, (0, gy, self.W, self.H - gy))

        # landing pad
        bs = sims[best_j]
        pl = self._w2s(bs.pad_x - bs.pad_half_width, 0)
        pr = self._w2s(bs.pad_x + bs.pad_half_width, 0)
        pygame.draw.rect(scr, self.PAD, (pl[0], pl[1] - 4, pr[0] - pl[0], 6))
        mx = (pl[0] + pr[0]) // 2
        pygame.draw.line(scr, self.PAD_MARK, (mx - 10, pl[1] - 1), (mx + 10, pl[1] - 1), 2)

        # ── draw ghost rockets first (behind best) ─────────────────
        ghost_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for j in range(len(sims)):
            if is_best[j]:
                continue
            trail = trails[j]
            state = sims[j].state

            for i in range(1, len(trail)):
                frac = i / max(len(trail), 1)
                c = tuple(int(v * frac) for v in self.TRAIL_DIM) + (self.GHOST_ALPHA,)
                p1 = self._w2s(*trail[i - 1])
                p2 = self._w2s(*trail[i])
                pygame.draw.line(ghost_surf, c, p1, p2, 1)

            exhausts[j].draw(ghost_surf, alpha=self.GHOST_ALPHA)
            self._draw_rocket_body(ghost_surf, state, alpha=self.GHOST_ALPHA)

        scr.blit(ghost_surf, (0, 0))

        # ── draw best rocket exhaust behind body ───────────────────
        best_exhaust_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        exhausts[best_j].draw(best_exhaust_surf, alpha=255)
        scr.blit(best_exhaust_surf, (0, 0))

        # ── draw best rocket trail + body on top ───────────────────
        trail = trails[best_j]
        for i in range(1, len(trail)):
            frac = i / max(len(trail), 1)
            c = tuple(int(v * frac) for v in self.TRAIL)
            pygame.draw.line(scr, c, self._w2s(*trail[i - 1]), self._w2s(*trail[i]), 1)

        self._draw_rocket_body(scr, sims[best_j].state, alpha=255)

        # HUD
        self._draw_hud(sims[best_j].state, infos[best_j], gi, sims[best_j])

        # overlay text
        if overlay:
            surf = self.font_xl.render(overlay, True, overlay_clr)
            rect = surf.get_rect(center=(self.W // 2, 70))
            pygame.draw.rect(scr, (0, 0, 0), rect.inflate(20, 10))
            scr.blit(surf, rect)

        pygame.display.flip()

    # ── rocket body (no flame — exhaust is handled by particle system) ──

    def _draw_rocket_body(self, surface, state, alpha=255):
        """Draw just the rocket body polygon onto the given surface."""
        sx, sy = self._w2s(state.x, state.y)
        th = state.theta
        cos_t, sin_t = np.cos(th), np.sin(th)

        h = ROCKET_BODY_HEIGHT / 2 * self.scale
        w = ROCKET_RADIUS * self.scale
        nh = ROCKET_NOSE_HEIGHT * self.scale

        local = [
            (0, 2 * h + nh),
            (-w, h * 1.35),
            (-w, 0),
            ( w, 0),
            ( w, h * 1.35),
        ]
        verts = [(lx * cos_t + ly * sin_t + sx,
                  lx * sin_t - ly * cos_t + sy) for lx, ly in local]

        if alpha >= 255:
            pygame.draw.polygon(surface, self.ROCKET, verts)
            pygame.draw.polygon(surface, self.ROCKET_EDGE, verts, 1)
            pygame.draw.polygon(surface, self.NOSE, [verts[0], verts[1], verts[4]])
        else:
            pygame.draw.polygon(surface, self.ROCKET + (alpha,), verts)
            pygame.draw.polygon(surface, self.ROCKET_EDGE + (alpha,), verts, 1)
            pygame.draw.polygon(surface, self.NOSE + (alpha,), [verts[0], verts[1], verts[4]])

    # ── heads-up display ────────────────────────────────────────────

    def _draw_hud(self, state, info, gi, sim):
        vspd = -state.vy
        hspd = state.vx
        thr = info.get("throttle", 0.0)
        wind = info.get("wind", 0.0)
        lines = [
            (self.font_lg, self.HUD,
             f"Gen {gi.get('gen','?')}/{gi.get('total_gen','?')}  |  Best"),
            (self.font, self.HUD,
             f"Fitness: {gi.get('fitness',0):.1f}   Success: {gi.get('success_rate',0):.0%}"),
            (self.font, self.HUD,
             f"Alt: {state.y:.1f} m   "
             f"Vspd: {vspd:+.1f} m/s   Hspd: {hspd:+.1f} m/s"),
            (self.font, self.HUD,
             f"Tilt: {np.degrees(state.theta):.1f}\u00b0   "
             f"Throttle: {thr:.0%}   Fuel: {state.fuel:.0%}"),
            (self.font, self.HUD,
             f"Wind: {wind:+.1f} m/s"),
            (self.font, self.HUD_DIM,
             "[SPACE] skip"),
        ]
        y = 10
        for fnt, clr, txt in lines:
            s = fnt.render(txt, True, clr)
            self.screen.blit(s, (14, y))
            y += s.get_height() + 3

        # wind arrow
        ax, ay = self.W - 130, 30
        alen = min(abs(wind) * 5, 70)
        if abs(wind) > 0.3:
            ex = ax + alen * np.sign(wind)
            pygame.draw.line(self.screen, self.WIND_CLR, (ax, ay), (int(ex), ay), 3)
            d = int(np.sign(wind))
            pygame.draw.polygon(self.screen, self.WIND_CLR, [
                (int(ex), ay), (int(ex - 6 * d), ay - 5), (int(ex - 6 * d), ay + 5)])
        self.screen.blit(self.font.render("Wind", True, self.WIND_CLR), (ax - 12, ay + 12))
