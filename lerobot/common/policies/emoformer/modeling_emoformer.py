#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import os
import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.emoformer.configuration_emoformer import EmoFormerConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class EmoFormerPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy adapted for FFT-based emotion trajectory prediction
    """

    config_class = EmoFormerConfig
    name = "emoformer"

    def __init__(
        self,
        config: EmoFormerConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Adapt normalization to the new data format
        # For FFT dataset, we need to normalize:
        # - inputs: joint states (num_joints)
        # - targets: FFT features (num_joints, num_freq_bins)
        
        self.normalize_inputs = FFTNormalize("inputs", dataset_stats)
        self.normalize_targets = FFTNormalize("targets", dataset_stats)
        self.unnormalize_outputs = FFTUnnormalize("targets", dataset_stats)

        # Initialize the model with proper dimensions from config
        self.model = EmoFormer(
            config=config,
            num_joints=config.input_dim,
            emotion_vocab_size=config.emotion_vocab_size,
            d_model=config.dim_model,
            nhead=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            ff_dim=config.intermediate_size,
            fft_feature_dim=config.output_dim
        )

    def get_optim_params(self) -> list[dict]:
        """Return parameters for optimization."""
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict FFT features given joint state and emotion."""
        self.eval()
        
        # Normalize inputs
        batch = self.normalize_inputs(batch)
        
        # Forward pass
        fft_features = self.model(batch)
        
        # Unnormalize outputs
        outputs = {"targets": fft_features}
        outputs = self.unnormalize_outputs(outputs)
        
        return outputs["targets"]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        # Store original targets for loss computation
        targets = batch["targets"]
        
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        # Forward pass through model
        fft_features_pred = self.model(batch)
        
        # Compute loss (MSE loss for FFT features)
        mse_loss = F.mse_loss(batch["targets"], fft_features_pred)
        
        loss_dict = {"mse_loss": mse_loss.item()}
        loss = mse_loss

        return loss, loss_dict


class FFTNormalize:
    """Normalization class for FFT dataset features."""
    
    def __init__(self, key, dataset_stats=None):
        """
        Initialize the normalization class.
        
        Args:
            key: The key to normalize ("inputs" or "targets")
            dataset_stats: Dictionary with statistics for normalization
        """
        self.key = key
        self.dataset_stats = dataset_stats
        
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Normalize the specified key in the batch.
        
        Args:
            batch: Dictionary with tensors
            
        Returns:
            Updated batch with normalized values
        """
        if self.dataset_stats is None:
            # Return batch unchanged if no stats available
            return batch
            
        if self.key not in batch:
            return batch
            
        # Get normalization stats
        stats = self.dataset_stats[self.key]
        mean = stats["mean"]
        std = stats["std"]
        
        # Normalize (assumes mean and std have proper broadcasting dimensions)
        batch[self.key] = (batch[self.key] - mean) / (std + 1e-8)
        
        return batch


class FFTUnnormalize:
    """Unnormalization class for FFT dataset features."""
    
    def __init__(self, key, dataset_stats=None):
        """
        Initialize the unnormalization class.
        
        Args:
            key: The key to unnormalize ("inputs" or "targets")
            dataset_stats: Dictionary with statistics for normalization
        """
        self.key = key
        self.dataset_stats = dataset_stats
        
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Unnormalize the specified key in the batch.
        
        Args:
            batch: Dictionary with tensors
            
        Returns:
            Updated batch with unnormalized values
        """
        if self.dataset_stats is None:
            # Return batch unchanged if no stats available
            return batch
            
        if self.key not in batch:
            return batch
            
        # Get normalization stats
        stats = self.dataset_stats[self.key]
        mean = stats["mean"]
        std = stats["std"]
        
        # Unnormalize
        batch[self.key] = batch[self.key] * (std + 1e-8) + mean
        
        return batch


# Function to compute dataset statistics for normalization
def compute_dataset_stats(dataloader):
    """
    Compute mean and std for dataset normalization.
    
    Args:
        dataloader: DataLoader object
        
    Returns:
        Dictionary with statistics for inputs and targets
    """
    # Initialize accumulators
    input_sum = 0
    input_squared_sum = 0
    target_sum = 0
    target_squared_sum = 0
    count = 0
    
    # First pass: compute sums for mean calculation
    for batch in dataloader:
        inputs = batch["inputs"]
        targets = batch["targets"]
        
        batch_size = inputs.shape[0]
        count += batch_size
        
        input_sum += inputs.sum(dim=0)
        input_squared_sum += (inputs ** 2).sum(dim=0)
        
        # For targets, we need to reshape to handle the 3D tensor
        targets_flat = targets.view(batch_size, -1)
        target_sum += targets_flat.sum(dim=0)
        target_squared_sum += (targets_flat ** 2).sum(dim=0)
    
    # Compute mean
    input_mean = input_sum / count
    target_mean = target_sum / count
    
    # Reshape target mean to match original dimensions
    # Assuming all target tensors have the same shape
    target_shape = dataloader.dataset[0]["targets"].shape
    target_mean = target_mean.view(target_shape)
    
    # Compute std
    input_std = torch.sqrt(input_squared_sum / count - input_mean ** 2)
    target_std = torch.sqrt(target_squared_sum / count - (target_sum / count) ** 2)
    target_std = target_std.view(target_shape)
    
    # Create stats dictionary
    stats = {
        "inputs": {
            "mean": input_mean,
            "std": input_std
        },
        "targets": {
            "mean": target_mean,
            "std": target_std
        }
    }
    
    return stats

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class EmoFormer(nn.Module):
    """Emotions Transformer: The underlying neural network for EmoFormerPolicy."""
    def __init__(self, config: EmoFormerConfig = None, 
                num_joints=6, 
                emotion_vocab_size=7, 
                d_model=64, 
                nhead=4, 
                num_layers=3, 
                ff_dim=128, 
                max_len=100,
                fft_feature_dim=None):
        super().__init__()

        self.config = config
        
        # Input projection for joint state
        self.joint_proj = nn.Linear(num_joints, d_model // 2)
        
        # Emotion embedding
        self.emotion_embedding = nn.Embedding(emotion_vocab_size, d_model // 2)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # FFT prediction head
        # If fft_feature_dim is not provided, we'll assume a default size
        if fft_feature_dim is None:
            # Default FFT output shape based on max_freq=5.0, fps=30, window_sec=1.5
            # Assuming around 8 frequency bins after filtering (this would need adjustment based on actual output)
            self.fft_feature_dim = num_joints * 8
        else:
            self.fft_feature_dim = fft_feature_dim
            
        self.action_head = nn.Linear(d_model, self.fft_feature_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the EmoFormer model.
        
        Args:
            batch: Dictionary containing:
                - inputs: (batch_size, num_joints) - Current joint state
                - emotion_idx: (batch_size,) - Emotion indices
                
        Returns:
            fft_features: (batch_size, num_joints, num_freq_bins) - Predicted FFT features
        """
        # Get inputs from batch
        joint_state = batch["inputs"]  # (batch_size, num_joints)
        emotion_idx = batch["emotion_idx"]  # (batch_size,)
        
        batch_size = joint_state.shape[0]
        
        # Project joint state to embedding space
        joint_emb = self.joint_proj(joint_state)  # (batch_size, d_model//2)
        
        # Get emotion embeddings
        emo_emb = self.emotion_embedding(emotion_idx)  # (batch_size, d_model//2)
        
        # Concatenate joint and emotion embeddings
        x = torch.cat([joint_emb, emo_emb], dim=-1)  # (batch_size, d_model)
        
        # Add sequence dimension for transformer (batch_size, 1, d_model)
        x = x.unsqueeze(1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.encoder(x)  # (batch_size, 1, d_model)
        
        # Extract features from sequence
        x = x.squeeze(1)  # (batch_size, d_model)
        
        # Project to FFT feature space
        fft_features = self.action_head(x)  # (batch_size, fft_feature_dim)
        
        # Reshape to match the expected output format [batch_size, num_joints, num_freq_bins]
        num_joints = batch["inputs"].shape[1]
        num_freq_bins = self.fft_feature_dim // num_joints
        
        fft_features = fft_features.reshape(batch_size, num_joints, num_freq_bins)
        
        return fft_features

def create_emoformer_model(dataset):
    """Create an EmoFormer model based on the dataset dimensions."""
    # Get dataset dimensions
    num_joints = dataset.get_input_dim()
    output_shape = dataset.get_output_shape()  # [num_joints, num_freq_bins]
    num_emotions = len(dataset.emotions)
    
    # Calculate FFT feature dimension
    fft_feature_dim = output_shape[0] * output_shape[1]  # num_joints * num_freq_bins
    
    # Create model
    model = EmoFormer(
        num_joints=num_joints,
        emotion_vocab_size=num_emotions,
        d_model=64,  # Can be adjusted
        nhead=4,     # Can be adjusted
        num_layers=3,  # Can be adjusted
        ff_dim=128,    # Can be adjusted
        fft_feature_dim=fft_feature_dim
    )
    
    return model

# # Example usage:
# if __name__ == "__main__":
#     from lerobot.common.datasets.trajectory_fft_dataset import TrajectoryFFTLabelDataset

#     # Example: create synthetic data for testing
#     num_episodes = 100
#     timesteps_per_episode = 600  # 500-800 samples per trajectory
#     num_joints = 6  # 6-DOF
    
#     # Create random trajectories
#     np.random.seed(42)
#     trajectories = []
#     task_descriptions = []
    
#     for i in range(num_episodes):
#         # Create sinusoidal trajectories with different frequencies for each joint
#         t = np.linspace(0, 20, timesteps_per_episode)
#         traj = np.zeros((timesteps_per_episode, num_joints))
        
#         for j in range(num_joints):
#             freq = 0.5 + j * 0.2  # Different frequency for each joint
#             traj[:, j] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
#         trajectories.append(traj)
        
#         # Assign random emotions
#         emotions = ['happy', 'angry', 'sad', 'surprised', 'fearful', 'curious', 'playful']
#         emotion = np.random.choice(emotions)
#         task_descriptions.append(f"Move with {emotion} emotion")
    
#     # Create dataset
#     dataset = TrajectoryFFTLabelDataset(
#         trajectories=trajectories,
#         task_descriptions=task_descriptions,
#         fps=30, 
#         window_sec=1.5,
#         overlap=0.5,
#         max_freq=5.0
#     )
    
    
#     # Create model
#     model = create_emoformer_model(dataset)
    
#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
#     # Example forward pass
#     for batch in dataloader:
#         outputs = model(batch)
#         print(f"Model output shape: {outputs.shape}")
#         break

if __name__ == "__main__":
    # Add missing imports
    import torch
    from torch.utils.data import DataLoader
    from lerobot.common.datasets.trajectory_fft_dataset import TrajectoryFFTLabelDataset
    # Example: create synthetic data for testing
    num_episodes = 100
    timesteps_per_episode = 600  # 500-800 samples per trajectory
    num_joints = 6  # 6-DOF
    
    # Create random trajectories
    np.random.seed(42)
    trajectories = []
    task_descriptions = []
    
    for i in range(num_episodes):
        # Create sinusoidal trajectories with different frequencies for each joint
        t = np.linspace(0, 20, timesteps_per_episode)
        traj = np.zeros((timesteps_per_episode, num_joints))
        
        for j in range(num_joints):
            freq = 0.5 + j * 0.2  # Different frequency for each joint
            traj[:, j] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        trajectories.append(traj)
        
        # Assign random emotions
        emotions = ['happy', 'angry', 'sad', 'surprised', 'fearful', 'curious', 'playful']
        emotion = np.random.choice(emotions)
        task_descriptions.append(f"Move with {emotion} emotion")
    
    # Create dataset
    dataset = TrajectoryFFTLabelDataset(
        trajectories=trajectories,
        task_descriptions=task_descriptions,
        fps=30, 
        window_sec=1.5,
        overlap=0.5,
        max_freq=5.0
    )
    
    # Print dataset statistics
    print(f"Dataset size: {len(dataset)}")
    print(f"Emotion distribution: {dataset.get_emotion_distribution()}")
    print(f"Input dimension: {dataset.get_input_dim()}")
    print(f"Output shape: {dataset.get_output_shape()}")
    
    # Create model
    model = create_emoformer_model(dataset)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example forward pass
    for batch in dataloader:
        outputs = model(batch)
        print(f"Input shape: {batch['inputs'].shape}")
        print(f"Emotion indices: {batch['emotion_idx'][:5]}")
        print(f"Model output shape: {outputs.shape}")
        print(f"Target shape: {batch['targets'].shape}")
        
        # Compare output and target shapes
        if outputs.shape == batch['targets'].shape:
            print("Output shape matches target shape ✓")
        else:
            print(f"Shape mismatch! Output: {outputs.shape}, Target: {batch['targets'].shape}")
        
        # Calculate sample loss
        mse_loss = F.mse_loss(outputs, batch['targets']).item()
        print(f"Sample MSE loss: {mse_loss:.4f}")
        break
    
    print("EmoFormer test completed successfully!")
