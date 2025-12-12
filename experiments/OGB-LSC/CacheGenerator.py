# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)

import torch

import os
import os.path as osp
from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCacheGenerator,
)

def gen_cache_filename(base_dir, prefix, rel, rank, world_size):
    return osp.join(base_dir, f"{prefix}_cache_rel{rel}_rank{rank}_ws{world_size}.pt")

def get_cache(
    src_gather_cache,
    dest_gather_cache,
    dest_scatter_cache,
    src_gather_cache_file,
    dest_gather_cache_file,
    dest_scatter_cache_file,
    rank,
    world_size,
    src_indices,
    dest_indices,
    edge_location,
    src_data_mappings,
    dest_data_mappings,
    num_src_rows,
    num_dest_rows,
):
    """ """
    if src_gather_cache is None:

        _src_gather_cache = NCCLGatherCacheGenerator(
            indices=src_indices,
            edge_placement=edge_location,
            edge_dest_ranks=src_data_mappings,
            num_input_rows=num_src_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_src_gather_cache, src_gather_cache_file)
    else:
        _src_gather_cache = src_gather_cache

    if dest_scatter_cache is None:
        _dest_scatter_cache = NCCLScatterCacheGenerator(
            indices=src_indices,
            edge_placement=edge_location,
            edge_dest_ranks=src_data_mappings,
            num_output_rows=num_src_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_dest_scatter_cache, dest_scatter_cache_file)
    else:
        _dest_scatter_cache = dest_scatter_cache

    if dest_gather_cache is None:
        _dest_gather_cache = NCCLGatherCacheGenerator(
            indices=dest_indices,
            edge_placement=edge_location,
            edge_dest_ranks=dest_data_mappings,
            num_input_rows=num_dest_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_dest_gather_cache, dest_gather_cache_file)
    else:
        _dest_gather_cache = dest_gather_cache

    # Unit tests

    return _src_gather_cache, _dest_scatter_cache, _dest_gather_cache


if __name__ == "__main__":
    from fire import Fire
    from functools import partial
    from config import SyntheticDatasetConfig

    # Use this script to generate the caches prior to running the main training script
    # This is useful because cache generation can take a long time and could cause issues
    # with timeouts on some systems.

    def main(dataset: str = "synthetic", world_size: int =4, cache_out_dir: str ="cache", mag240_data_dir: str ="data/MAG240M",):
        print(f"Generating caches for dataset: {dataset} with world size: {world_size}")
        print(f"Caches will be saved to: {cache_out_dir}")
        print(f"MAG240M data directory: {mag240_data_dir}")
        assert dataset in ["synthetic", "mag240m"]

        if dataset == "synthetic":
            from synthetic.synthetic_dataset import HeterogeneousDataset as Dataset

            synthetic_config = SyntheticDatasetConfig()
            graph_dataset = partial(
                Dataset,
                num_papers=synthetic_config.num_papers,
                num_authors=synthetic_config.num_authors,
                num_institutions=synthetic_config.num_institutions,
                num_features=synthetic_config.num_features,
                num_classes=synthetic_config.num_classes,
            )
        elif dataset == "mag240m":
            from mag240m.DGraph_MAG240M import DGraph_MAG240M as Dataset

            graph_dataset = partial(Dataset, data_dir=mag240_data_dir)

        rank = 0
        COMM = type(
            "dummy_comm",
            (object,),
            {"get_rank": lambda self: rank, "get_world_size": lambda self: world_size, "barrier": lambda self: None},
        )
        comm = COMM()

        dataset = graph_dataset(
            comm=comm,
        )

        dataset = dataset.add_batch_dimension()
        dataset = dataset.to("cpu")

        xs, edge_indices, edge_types, rank_mappings = dataset[0]

        os.makedirs(cache_out_dir, exist_ok=True)
        for simulated_rank in range(world_size):
            rel = 0

            for edge_index, edge_type, rank_mapping in zip(
                edge_indices, edge_types, rank_mappings
            ):
                # Debug only code?
                # if rel != 3:
                #     rel += 1
                #     continue
                print(f"Processing relation {rel} for simulated rank {simulated_rank}")
                print(f"Edge index shape: {edge_index.shape}")
                print(f"Edge type shape: {edge_type}")
                print(f"Rank mapping shape: {rank_mapping[0].shape}")
                print(f"Rank mapping shape: {rank_mapping[1].shape}")

                get_cache(
                    src_gather_cache=None,
                    dest_gather_cache=None,
                    dest_scatter_cache=None,
                    src_gather_cache_file=gen_cache_filename(cache_out_dir, "src_gather", rel, simulated_rank, world_size),
                    dest_gather_cache_file=gen_cache_filename(cache_out_dir, "dest_gather", rel, simulated_rank, world_size),
                    dest_scatter_cache_file=gen_cache_filename(cache_out_dir, "dest_scatter", rel, simulated_rank, world_size),
                    rank=simulated_rank,
                    world_size=world_size,
                    src_indices=edge_index[:, 0],
                    dest_indices=edge_index[:, 1],
                    edge_location=rank_mapping[0],
                    src_data_mappings=rank_mapping[0],
                    dest_data_mappings=rank_mapping[1],
                    num_src_rows=xs[edge_type[0]].shape[1],
                    num_dest_rows=xs[edge_type[1]].shape[1],
                )

                rel += 1

        print("Cache generation complete.")

        # rel = 3
        # synthetic_scatter_cache_1 = torch.load(
        #     f"{cache_out_dir}/synthetic_dest_scatter_cache_{rel}_1_{world_size}.pt",
        #     weights_only=False,
        # )
        # synthetic_scatter_cache_0 = torch.load(
        #     f"{cache_out_dir}/synthetic_dest_scatter_cache_{rel}_0_{world_size}.pt",
        #     weights_only=False,
        # )

        # print(synthetic_scatter_cache_1.scatter_recv_local_placement)
        # print(synthetic_scatter_cache_0.scatter_recv_local_placement)

    Fire(main)
