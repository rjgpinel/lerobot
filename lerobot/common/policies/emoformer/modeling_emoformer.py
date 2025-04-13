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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class EmoFormerPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
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

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = EmoFormer(config)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        targets = batch["targets"]
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        return loss, loss_dict


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

# Example usage:
if __name__ == "__main__":
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
    
    
    # Create model
    model = create_emoformer_model(dataset)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example forward pass
    for batch in dataloader:
        outputs = model(batch)
        print(f"Model output shape: {outputs.shape}")
        break
