"""Unit tests for TopologyGenerator in seam.sharing.topology."""

from __future__ import annotations

import numpy as np
import pytest

from seam.sharing.topology import TopologyGenerator


def test_topology_full_broadcast():
    topo = TopologyGenerator(n_agents=4, topology_type="full_broadcast")
    assert topo.get_neighbors("agent_0") == ["agent_1", "agent_2", "agent_3"]
    assert topo.get_neighbors("agent_1") == ["agent_0", "agent_2", "agent_3"]
    assert np.trace(topo.adjacency_matrix) == 0


def test_topology_ring():
    topo = TopologyGenerator(n_agents=4, topology_type="ring")
    # agent_0 receives from agent_3 (3%4) and agent_1 (1%4)
    neighbors = topo.get_neighbors("agent_0")
    assert sorted(neighbors) == ["agent_1", "agent_3"]


def test_topology_star():
    topo = TopologyGenerator(n_agents=4, topology_type="star")
    # Hub agent_0 connects to all leaves
    assert sorted(topo.get_neighbors("agent_0")) == ["agent_1", "agent_2", "agent_3"]
    # Leaf agent_1 only receives from hub agent_0
    assert topo.get_neighbors("agent_1") == ["agent_0"]


def test_topology_cluster():
    topo = TopologyGenerator(n_agents=4, topology_type="cluster")
    # Cluster 1: agent_0 and agent_1
    assert topo.get_neighbors("agent_0") == ["agent_1"]
    assert topo.get_neighbors("agent_1") == ["agent_0"]
    # Cluster 2: agent_2 and agent_3
    assert topo.get_neighbors("agent_2") == ["agent_3"]
    assert topo.get_neighbors("agent_3") == ["agent_2"]


def test_topology_off():
    topo = TopologyGenerator(n_agents=4, topology_type="off")
    assert topo.get_neighbors("agent_0") == []
    assert np.sum(topo.adjacency_matrix) == 0


def test_invalid_topology():
    with pytest.raises(ValueError, match="Unsupported topology"):
        TopologyGenerator(n_agents=4, topology_type="invalid_shape")


def test_invalid_agent_id():
    topo = TopologyGenerator(n_agents=4, topology_type="full_broadcast")
    with pytest.raises(KeyError, match="not found in topology"):
        topo.get_neighbors("agent_99")
