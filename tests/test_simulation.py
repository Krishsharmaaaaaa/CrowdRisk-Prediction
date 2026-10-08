"""
Unit tests for Independent Pedestrian Evacuation Simulation Testbed.
Verifies:
- Deterministic seed behaviour
- Agent initialization within room geometry
- Physical intervention parameter changes (D, B, K, O)
- Exit door egress and evacuation logic
- Metric calculation consistency (clearance time, throughput, congestion)
- Reproducibility across runs
- Choke driver feature vector extraction
"""

import math
import numpy as np
import pytest

from src.simulation.pedestrian_sim import (
    Agent,
    PedestrianEvacuationSimulator,
    SimulationConfig,
    SimulationOutcome
)


def test_simulation_agent_initialization():
    """Verify agent spawn geometry, count, and initial parameters."""
    cfg = SimulationConfig(room_width=15.0, room_height=12.0, num_agents=40)
    sim = PedestrianEvacuationSimulator(cfg)
    sim.reset(seed=42)

    assert len(sim.agents) == 40
    for ag in sim.agents:
        assert 0.0 < ag.x < cfg.room_width
        assert 0.0 < ag.y < cfg.room_height
        assert ag.evacuated is False
        assert ag.exit_time is None
        assert ag.desired_speed > 0.0
        assert ag.radius == 0.25


def test_simulation_deterministic_seed_reproducibility():
    """Verify that identical random seeds produce byte-for-byte identical outcomes."""
    cfg = SimulationConfig(num_agents=30, max_steps=400)

    sim1 = PedestrianEvacuationSimulator(cfg)
    out1 = sim1.run(seed=123)

    sim2 = PedestrianEvacuationSimulator(cfg)
    out2 = sim2.run(seed=123)

    assert out1.evacuated_count == out2.evacuated_count
    assert out1.remaining_count == out2.remaining_count
    assert abs(out1.clearance_time - out2.clearance_time) < 1e-5
    assert abs(out1.exit_throughput - out2.exit_throughput) < 1e-5
    assert abs(out1.congestion_duration - out2.congestion_duration) < 1e-5
    assert out1.collision_count == out2.collision_count


def test_simulation_different_seeds_vary():
    """Verify that different random seeds produce distinct stochastic trajectories."""
    cfg = SimulationConfig(num_agents=35, max_steps=400)

    sim1 = PedestrianEvacuationSimulator(cfg)
    out1 = sim1.run(seed=42)

    sim2 = PedestrianEvacuationSimulator(cfg)
    out2 = sim2.run(seed=999)

    # Positions and clearance times must differ due to differing spawn and noise jitter
    assert abs(out1.clearance_time - out2.clearance_time) > 1e-4 or out1.collision_count != out2.collision_count


def test_intervention_parameter_changes():
    """Verify that physical intervention parameters alter simulation configuration as expected."""
    # Bottleneck intervention (door widening)
    cfg_narrow = SimulationConfig(door_width=0.6)
    cfg_wide = SimulationConfig(door_width=2.0)
    assert cfg_wide.door_width > cfg_narrow.door_width

    # Density intervention (population reduction)
    cfg_dense = SimulationConfig(num_agents=100)
    cfg_sparse = SimulationConfig(num_agents=40)
    assert cfg_sparse.num_agents < cfg_dense.num_agents

    # Disorder intervention (heading jitter suppression)
    cfg_chaotic = SimulationConfig(noise_strength=0.4)
    cfg_orderly = SimulationConfig(noise_strength=0.02)
    assert cfg_orderly.noise_strength < cfg_chaotic.noise_strength

    # Kinematics intervention (speed pacification)
    cfg_surge = SimulationConfig(desired_speed_mean=2.8, desired_speed_std=0.8)
    cfg_calm = SimulationConfig(desired_speed_mean=1.2, desired_speed_std=0.1)
    assert cfg_calm.desired_speed_mean < cfg_surge.desired_speed_mean


def test_exit_evacuation_logic():
    """Verify that agents exiting through the door boundary are correctly recorded as evacuated."""
    cfg = SimulationConfig(num_agents=20, max_steps=600)
    sim = PedestrianEvacuationSimulator(cfg)
    outcome = sim.run(seed=42)

    assert outcome.evacuated_count > 0
    assert outcome.total_agents == 20
    assert outcome.evacuated_count + outcome.remaining_count == 20

    # Evacuated agents must have exit_time recorded
    evacuated_agents = [ag for ag in sim.agents if ag.evacuated]
    assert len(evacuated_agents) == outcome.evacuated_count
    for ag in evacuated_agents:
        assert ag.exit_time is not None
        assert 0.0 < ag.exit_time <= outcome.clearance_time


def test_metric_calculation_consistency():
    """Verify that simulator-native outcomes satisfy mathematical identities."""
    cfg = SimulationConfig(num_agents=25, max_steps=500)
    sim = PedestrianEvacuationSimulator(cfg)
    outcome = sim.run(seed=42)

    # Throughput identity: throughput = evacuated / clearance_time
    if outcome.evacuated_count > 0:
        expected_throughput = outcome.evacuated_count / outcome.clearance_time
        assert abs(outcome.exit_throughput - expected_throughput) < 1e-3
        assert outcome.mean_evacuation_time <= outcome.clearance_time

    assert outcome.congestion_duration >= 0.0
    assert outcome.collision_count >= 0

    d = outcome.to_dict()
    assert "clearance_time" in d
    assert "exit_throughput" in d
    assert "congestion_duration" in d
    assert "remaining_count" in d


def test_choke_driver_vector_extraction():
    """Verify extraction of [D, O, B, K] vector at exit choke zone."""
    cfg = SimulationConfig(num_agents=40, door_width=0.8, max_steps=150)
    sim = PedestrianEvacuationSimulator(cfg)
    sim.reset(seed=42)

    # Step simulation into active egress queue
    for _ in range(80):
        sim.step()

    vec = sim.extract_driver_vector_at_choke()
    assert len(vec) == 4
    for val in vec:
        assert 0.0 <= val <= 1.0

    congested_vec = sim.get_congested_choke_driver_vector()
    assert len(congested_vec) == 4
    for val in congested_vec:
        assert 0.0 <= val <= 1.0
