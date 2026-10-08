"""
Independent Pedestrian Evacuation Simulation Testbed.
Implements a microscopic social-force / continuous agent evacuation environment
to evaluate whether CRDA model-based risk-driver attributions correspond to
measurable improvements in simulator-native physical egress outcomes.
Zero Circularity: Simulator measures physical clearance time, exit throughput,
and congestion duration independently from computer-vision feature formulas.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Agent:
    """Microscopic pedestrian agent state."""
    agent_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    desired_speed: float = 1.3       # m/s
    radius: float = 0.25              # meters
    mass: float = 70.0                # kg
    evacuated: bool = False
    exit_time: Optional[float] = None
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Environment geometry and physical intervention parameters."""
    room_width: float = 15.0          # meters
    room_height: float = 12.0         # meters
    door_y: float = 6.0               # center y of door on right wall (x = room_width)
    door_width: float = 1.0           # meters (Bottleneck intervention variable)
    num_agents: int = 80              # agents (Density intervention variable)
    desired_speed_mean: float = 1.3   # m/s (Kinematics intervention variable)
    desired_speed_std: float = 0.2    # m/s
    noise_strength: float = 0.1       # radians (Disorder intervention variable)
    relaxation_time: float = 0.5      # tau in seconds
    repulsion_strength: float = 2.0   # social force A
    repulsion_range: float = 0.4      # social force B (meters)
    dt: float = 0.05                  # timestep in seconds
    max_steps: int = 1500             # max simulation timesteps (75 seconds)


@dataclass
class SimulationOutcome:
    """Simulator-native egress performance outcomes (independent of model features)."""
    scenario_name: str
    seed: int
    total_agents: int
    evacuated_count: int
    remaining_count: int
    clearance_time: float             # seconds until last evacuation (or cutoff)
    exit_throughput: float            # agents evacuated per second
    congestion_duration: float        # seconds where choke point had >= 4 agents queueing
    mean_evacuation_time: float       # average seconds to egress per evacuated agent
    collision_count: int              # proximity overlaps (< 2*radius)

    def to_dict(self) -> Dict[str, float]:
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "total_agents": self.total_agents,
            "evacuated_count": self.evacuated_count,
            "remaining_count": self.remaining_count,
            "clearance_time": round(self.clearance_time, 2),
            "exit_throughput": round(self.exit_throughput, 3),
            "congestion_duration": round(self.congestion_duration, 2),
            "mean_evacuation_time": round(self.mean_evacuation_time, 2),
            "collision_count": self.collision_count
        }


class PedestrianEvacuationSimulator:
    """
    Microscopic agent-based evacuation simulator with continuous coordinates.
    Simulates room egress through an exit bottleneck on the boundary wall.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.agents: List[Agent] = []
        self.current_step = 0
        self.time = 0.0
        self.congestion_steps = 0
        self.total_collisions = 0
        self.choke_feature_history: List[np.ndarray] = []

    def reset(self, seed: Optional[int] = None):
        """Initializes agent positions and velocities using deterministic seed."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.time = 0.0
        self.congestion_steps = 0
        self.total_collisions = 0
        self.choke_feature_history.clear()
        self.agents.clear()

        w, h = self.cfg.room_width, self.cfg.room_height
        door_y = self.cfg.door_y
        dw = self.cfg.door_width

        # Spawn agents in the left and central zones of the room
        spawn_x_max = w * 0.75
        spawn_margin = 0.8

        placed = 0
        attempts = 0
        max_attempts = self.cfg.num_agents * 50

        while placed < self.cfg.num_agents and attempts < max_attempts:
            attempts += 1
            x = float(np.random.uniform(spawn_margin, spawn_x_max))
            y = float(np.random.uniform(spawn_margin, h - spawn_margin))

            # Ensure minimum separation at spawn
            overlap = False
            for ag in self.agents:
                if math.hypot(x - ag.x, y - ag.y) < 2 * ag.radius:
                    overlap = True
                    break

            if not overlap:
                v0 = float(np.clip(
                    np.random.normal(self.cfg.desired_speed_mean, self.cfg.desired_speed_std),
                    0.4,
                    4.5
                ))
                agent = Agent(
                    agent_id=placed + 1,
                    x=x,
                    y=y,
                    desired_speed=v0,
                    radius=0.25,
                    trajectory=[(x, y)]
                )
                self.agents.append(agent)
                placed += 1

        # If room was too packed for rejection sampling, spawn remaining with minimal jitter
        while len(self.agents) < self.cfg.num_agents:
            idx = len(self.agents)
            grid_col = idx % 10
            grid_row = idx // 10
            x = float(1.0 + grid_col * 0.9)
            y = float(1.0 + grid_row * 0.9)
            self.agents.append(Agent(
                agent_id=idx + 1,
                x=x,
                y=y,
                desired_speed=self.cfg.desired_speed_mean,
                trajectory=[(x, y)]
            ))

    def step(self):
        """Advances simulation by one timestep dt using Social Force dynamics."""
        dt = self.cfg.dt
        w = self.cfg.room_width
        h = self.cfg.room_height
        door_y = self.cfg.door_y
        half_door = self.cfg.door_width / 2.0
        door_top = door_y - half_door
        door_bot = door_y + half_door

        active_agents = [ag for ag in self.agents if not ag.evacuated]
        if not active_agents:
            self.time += dt
            self.current_step += 1
            return

        # Check choke point queue congestion (within 3m of door on inside)
        near_door = sum(1 for ag in active_agents if (w - ag.x < 3.0 and abs(ag.y - door_y) < max(2.0, half_door * 2.0)))
        if near_door >= 4:
            self.congestion_steps += 1
            if self.current_step % 5 == 0:
                self.choke_feature_history.append(self.extract_driver_vector_at_choke())

        forces_x = [0.0] * len(active_agents)
        forces_y = [0.0] * len(active_agents)

        # 1. Driving Force toward Exit
        for i, ag in enumerate(active_agents):
            # Target is door center (w, door_y)
            dx = w - ag.x
            dy = door_y - ag.y
            dist_to_door = math.hypot(dx, dy)

            if dist_to_door > 0.05:
                e_x = dx / dist_to_door
                e_y = dy / dist_to_door
            else:
                e_x, e_y = 1.0, 0.0

            # Directional noise / disorder
            if self.cfg.noise_strength > 0:
                angle_jitter = np.random.normal(0, self.cfg.noise_strength)
                cos_j, sin_j = math.cos(angle_jitter), math.sin(angle_jitter)
                e_x_noise = e_x * cos_j - e_y * sin_j
                e_y_noise = e_x * sin_j + e_y * cos_j
                e_x, e_y = e_x_noise, e_y_noise

            desired_vx = ag.desired_speed * e_x
            desired_vy = ag.desired_speed * e_y

            f_drive_x = (desired_vx - ag.vx) / self.cfg.relaxation_time
            f_drive_y = (desired_vy - ag.vy) / self.cfg.relaxation_time

            forces_x[i] += f_drive_x
            forces_y[i] += f_drive_y

        # 2. Inter-Agent Repulsion & Physical Collision
        n_act = len(active_agents)
        for i in range(n_act):
            ag_i = active_agents[i]
            for j in range(i + 1, n_act):
                ag_j = active_agents[j]
                dij_x = ag_i.x - ag_j.x
                dij_y = ag_i.y - ag_j.y
                dist = math.hypot(dij_x, dij_y)
                radii = ag_i.radius + ag_j.radius

                if dist < 0.01:
                    dist = 0.01
                    dij_x, dij_y = 0.01, 0.0

                n_x = dij_x / dist
                n_y = dij_y / dist

                # Social repulsion
                f_rep = self.cfg.repulsion_strength * math.exp(-(dist - radii) / self.cfg.repulsion_range)

                # Physical contact force if overlapping
                if dist < radii:
                    self.total_collisions += 1
                    f_contact = 50.0 * (radii - dist)
                    f_rep += f_contact

                forces_x[i] += f_rep * n_x
                forces_y[i] += f_rep * n_y
                forces_x[j] -= f_rep * n_x
                forces_y[j] -= f_rep * n_y

        # 3. Wall Repulsion
        for i, ag in enumerate(active_agents):
            # Top wall (y = 0)
            if ag.y < 1.0:
                f_w = self.cfg.repulsion_strength * math.exp(-ag.y / self.cfg.repulsion_range)
                forces_y[i] += f_w
            # Bottom wall (y = h)
            if h - ag.y < 1.0:
                f_w = self.cfg.repulsion_strength * math.exp(-(h - ag.y) / self.cfg.repulsion_range)
                forces_y[i] -= f_w
            # Left wall (x = 0)
            if ag.x < 1.0:
                f_w = self.cfg.repulsion_strength * math.exp(-ag.x / self.cfg.repulsion_range)
                forces_x[i] += f_w
            # Right wall with door opening (x = w)
            if w - ag.x < 1.0:
                # If within door gap, no repulsion; if hitting wall boundary, push back
                if not (door_top <= ag.y <= door_bot):
                    f_w = self.cfg.repulsion_strength * math.exp(-(w - ag.x) / self.cfg.repulsion_range)
                    forces_x[i] -= f_w

        # 4. Integrate Velocities and Positions
        for i, ag in enumerate(active_agents):
            ag.vx += forces_x[i] * dt
            ag.vy += forces_y[i] * dt

            # Velocity clamping to physical limits
            spd = math.hypot(ag.vx, ag.vy)
            max_v = ag.desired_speed * 1.5
            if spd > max_v:
                ag.vx = (ag.vx / spd) * max_v
                ag.vy = (ag.vy / spd) * max_v

            ag.x += ag.vx * dt
            ag.y += ag.vy * dt

            # Keep inside room boundaries
            ag.x = max(ag.radius, ag.x)
            ag.y = max(ag.radius, min(h - ag.radius, ag.y))

            # Check Evacuation (Passing through door at x >= w)
            if ag.x >= w - 0.1 and (door_top - 0.2 <= ag.y <= door_bot + 0.2):
                ag.evacuated = True
                ag.exit_time = self.time

            ag.trajectory.append((ag.x, ag.y))

        self.time += dt
        self.current_step += 1

    def run(self, seed: Optional[int] = None) -> SimulationOutcome:
        """Runs complete simulation until all agents evacuate or max steps reached."""
        self.reset(seed=seed)

        for _ in range(self.cfg.max_steps):
            active = sum(1 for ag in self.agents if not ag.evacuated)
            if active == 0:
                break
            self.step()

        evacuated = [ag for ag in self.agents if ag.evacuated]
        evac_count = len(evacuated)
        rem_count = len(self.agents) - evac_count

        clearance = float(max([ag.exit_time for ag in evacuated])) if evacuated else float(self.time)
        throughput = float(evac_count / max(1.0, clearance))
        congestion_duration = float(self.congestion_steps * self.cfg.dt)
        mean_evac = float(np.mean([ag.exit_time for ag in evacuated])) if evacuated else float(self.time)

        return SimulationOutcome(
            scenario_name="",
            seed=seed if seed is not None else -1,
            total_agents=len(self.agents),
            evacuated_count=evac_count,
            remaining_count=rem_count,
            clearance_time=clearance,
            exit_throughput=throughput,
            congestion_duration=congestion_duration,
            mean_evacuation_time=mean_evac,
            collision_count=self.total_collisions
        )

    def extract_driver_vector_at_choke(self) -> np.ndarray:
        """
        Computes the observed [D, O, B, K] feature vector at the exit bottleneck zone
        (for feeding to CRDA).
        """
        w, h = self.cfg.room_width, self.cfg.room_height
        door_y = self.cfg.door_y
        choke_x1, choke_x2 = w - 4.0, w
        choke_y1, choke_y2 = door_y - 2.5, door_y + 2.5

        active = [ag for ag in self.agents if not ag.evacuated and choke_x1 <= ag.x <= choke_x2 and choke_y1 <= ag.y <= choke_y2]
        count = len(active)

        # D: Density relative to nominal choke capacity (15 agents)
        d_val = float(np.clip(count / 15.0, 0.0, 1.0))

        # O: Directional Disorder
        if count >= 2:
            angles = [math.atan2(ag.vy, ag.vx) for ag in active if math.hypot(ag.vx, ag.vy) > 0.05]
            if len(angles) >= 2:
                cos_s = sum(math.cos(a) for a in angles)
                sin_s = sum(math.sin(a) for a in angles)
                r_bar = math.hypot(cos_s, sin_s) / len(angles)
                o_val = float(np.clip((1.0 - r_bar) * 2.5, 0.0, 1.0))
            else:
                o_val = 0.0
        else:
            o_val = 0.0

        # B: Bottleneck Jamming (speed reduction and queue freezing)
        if count >= 1:
            speeds = [math.hypot(ag.vx, ag.vy) for ag in active]
            mean_spd = float(np.mean(speeds))
            spd_drop = float(np.clip(max(0.0, 1.0 - mean_spd / self.cfg.desired_speed_mean), 0.0, 1.0))
            stopped_ratio = sum(1 for s in speeds if s < 0.2) / count
            b_val = float(np.clip(0.6 * spd_drop + 0.4 * stopped_ratio, 0.0, 1.0))
        else:
            b_val = 0.0

        # K: Kinematic Instability (speed variability / surge)
        if count >= 2:
            speeds = [math.hypot(ag.vx, ag.vy) for ag in active]
            spd_var = float(np.var(speeds))
            k_val = float(np.clip(math.sqrt(spd_var) / (self.cfg.desired_speed_mean * 0.5 + 1e-4), 0.0, 1.0))
        else:
            k_val = 0.0

        return np.array([d_val, o_val, b_val, k_val], dtype=np.float32)

    def get_congested_choke_driver_vector(self) -> np.ndarray:
        """
        Returns the mean [D, O, B, K] vector observed during congested queue states.
        If no congestion occurred, falls back to the instantaneous choke vector.
        """
        if self.choke_feature_history:
            return np.mean(self.choke_feature_history, axis=0).astype(np.float32)
        return self.extract_driver_vector_at_choke()
