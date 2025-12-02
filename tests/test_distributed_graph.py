import pytest
from DGraph.data.graph import get_round_robin_node_rank_map

def test_round_robin_node_rank_map_zero_nodes():
  node_count = 0
  rank_count = 3
  expected = []
  result = get_round_robin_node_rank_map(node_count, rank_count)
  assert result.tolist() == expected

def test_round_robin_node_rank_map_single_rank():
  node_count = 4
  rank_count = 1
  expected = [0, 0, 0, 0]
  result = get_round_robin_node_rank_map(node_count, rank_count)
  assert result.tolist() == expected

def test_round_robin_node_rank_map_more_nodes_than_ranks():
  node_count = 6
  rank_count = 3
  expected = [0, 1, 2, 0, 1, 2]
  result = get_round_robin_node_rank_map(node_count, rank_count)
  assert result.tolist() == expected

def test_round_robin_node_rank_map_more_ranks_than_nodes():
  node_count = 3
  rank_count = 5
  expected = [0, 1, 2]
  result = get_round_robin_node_rank_map(node_count, rank_count)
  assert result.tolist() == expected
