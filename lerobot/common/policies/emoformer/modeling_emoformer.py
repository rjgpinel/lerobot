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



def extract_emotion(text):
    text = text.lower()

    if "sad" in text:
        return "sad"
    elif "surprised" in text:
        return "surprised"
    elif "happy" in text or "happiness" in text or "cheerful" in text:
        return "happy"
    elif "angry" in text or "anger" in text or "rage" in text:
        return "angry"
    elif "fear" in text or "terrified" in text or "terror" in text:
        return "fearful"
    elif "curious" in text or "curiosity" in text:
        return "curious"
    elif "playful" in text or "playfulness" in text:
        return "playful"
    else:
        return "unknown"


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

class Emoformer(nn.Module):
    #Emotions Transformer: The underlying neural network for EmoFormerPolicy.
    def __init__(self, config: EmoFormerConfig, num_joints=12, emotion_vocab_size=7, d_model=64, nhead=4, num_layers=3, ff_dim=128, max_len=100):
        super().__init__()

        self.config = config

        self.joint_proj = nn.Linear(num_joints, d_model // 2)
        self.emotion_embedding = nn.Embedding(emotion_vocab_size, d_model // 2)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        joint_states: (batch_size, seq_len, num_joints)
        emotion_ids:  (batch_size, seq_len) or (batch_size,) if emotion is constant
        """
        emotion_id = batch["observation.emotion_ids"]
        batch_size = batch["observation.state"].shape[0]
        joint_state = batch["observation.state"]

        joint_emb = self.joint_proj(joint_state)  # (B, S, d_model//2)
        
        if emotion_ids.ndim == 2:
            emo_emb = self.emotion_embedding(emotion_id)  # (B, S, d_model//2)
        else:
            emo_emb = self.emotion_embedding(emotion_ids).unsqueeze(1)  # (B, 1, d_model//2)
            emo_emb = emo_emb.expand(-1, joint_states.size(1), -1)  # (B, S, d_model//2)

        x = torch.cat([joint_emb, emo_emb], dim=-1)  # (B, S, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (B, S, d_model)

        actions = self.action_head(decoder_out)

        return actions


if __name__ == "__main__":
    data = load_robot_data(base_dir="/home/rgarciap/Data2/lerobot_emotions")
    
    print(data)
