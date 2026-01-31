"""
Nanotron training script for numpy tokenized shards.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 run_train_numpy.py --config-file config_numpy_llama.yaml
```
"""
import argparse
import glob
import os
from typing import Dict, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nanotron import logging
from nanotron.config import DataArgs, DatasetStageArgs
from nanotron.data.clm_collator import DataCollatorForCLM
from nanotron.data.dataloader import get_dataloader_worker_init
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer

logger = logging.get_logger(__name__)


class NumpyShardDataset(Dataset):
    """
    Dataset that loads pre-tokenized numpy shards.
    
    Expects files named like: train_shard_00000.npy, train_shard_00001.npy, etc.
    Each shard is a 1D numpy array of token IDs.
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int,
        num_samples: int,
        seed: int = 42,
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        
        # Find all shards
        shard_pattern = os.path.join(data_dir, "hindi_shard_*.npy")
        self.shard_files = sorted(glob.glob(shard_pattern))
        
        if not self.shard_files:
            raise ValueError(f"No shard files found matching pattern: {shard_pattern}")
        
        # Use all shards for full training
        # max_shards = 10  # Removed - now using all shards
        
        log_rank(
            f"Using {len(self.shard_files)} shard files from {data_dir}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        
        # Memory-map all shards and calculate offsets
        self.shards = []
        self.shard_offsets = [0]  # Starting offset for each shard
        total_tokens = 0
        
        for shard_file in self.shard_files:
            shard = np.load(shard_file, mmap_mode='r')
            self.shards.append(shard)
            total_tokens += len(shard)
            self.shard_offsets.append(total_tokens)
        
        self.total_tokens = total_tokens
        # +1 for labels (we need seq_len + 1 tokens per sample)
        self.max_samples = total_tokens // (sequence_length + 1)
        
        log_rank(
            f"Total tokens: {total_tokens:,}, Max samples: {self.max_samples:,}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        
        # Pre-compute sample indices with shuffling
        rng = np.random.RandomState(seed)
        self.sample_indices = rng.permutation(self.max_samples)
        
    def __len__(self) -> int:
        return min(self.num_samples, self.max_samples)
    
    def _get_token_at(self, global_idx: int) -> int:
        """Get token at global index across all shards."""
        # Binary search to find the right shard
        shard_idx = np.searchsorted(self.shard_offsets[1:], global_idx, side='right')
        local_idx = global_idx - self.shard_offsets[shard_idx]
        return int(self.shards[shard_idx][local_idx])
    
    def _get_tokens(self, start_idx: int, length: int) -> np.ndarray:
        """Get a contiguous sequence of tokens starting at global start_idx."""
        tokens = np.zeros(length, dtype=np.int64)
        
        for i in range(length):
            global_idx = (start_idx + i) % self.total_tokens
            tokens[i] = self._get_token_at(global_idx)
        
        return tokens
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Use shuffled index
        shuffled_idx = self.sample_indices[idx % len(self.sample_indices)]
        
        # Calculate starting position for this sample
        start_pos = shuffled_idx * (self.sequence_length + 1)
        
        # Get sequence_length + 1 tokens (input + target)
        tokens = self._get_tokens(start_pos, self.sequence_length + 1)
        
        return {"input_ids": tokens}
    
    @property
    def folder_path(self) -> str:
        """Return the data directory path."""
        return self.data_dir
    
    def get_consumption_stats(self) -> Dict:
        """Return consumption statistics for the trainer."""
        return {self.data_dir: {"tokens": 0}}


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Custom data path - modify this to your data location
    DATA_DIR = "/workspace/2b_hindi_indicbart_shards"
    
    log_rank(
        f"Loading numpy shards from {DATA_DIR}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    
    # Calculate total samples needed
    total_samples_needed = trainer.global_batch_size * num_remaining_train_steps
    
    # Create dataset
    train_dataset = NumpyShardDataset(
        data_dir=DATA_DIR,
        sequence_length=trainer.sequence_length,
        num_samples=total_samples_needed + consumed_train_samples,
        seed=data.seed,
        dp_rank=trainer.parallel_context.dp_pg.rank(),
        dp_size=trainer.parallel_context.dp_pg.size(),
    )
    
    log_rank(
        f"Dataset created with {len(train_dataset):,} samples",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    data_collator = DataCollatorForCLM(
        sequence_length=trainer.sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=trainer.parallel_context,
    )
    
    # Create sampler for distributed training
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=trainer.parallel_context.dp_pg.size(),
        rank=trainer.parallel_context.dp_pg.rank(),
        shuffle=False,  # Already shuffled in dataset
        drop_last=True,
    )

    return DataLoader(
        train_dataset,
        batch_size=trainer.micro_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=True,
        num_workers=data.num_loading_workers,
        pin_memory=True,
        worker_init_fn=get_dataloader_worker_init(dp_rank=trainer.parallel_context.dp_pg.rank()),
    )


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples, _ = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
