import torch
import torch.nn as nn
import math


import torch
import torch.nn as nn
import math


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

class RobotTransformer(nn.Module):
    def __init__(self, num_joints=6, emotion_vocab_size=7, d_model=64, nhead=4, num_layers=3, ff_dim=128, act_dim=6, max_len=100):
        super().__init__()
        self.joint_proj = nn.Linear(num_joints, d_model // 2)
        self.emotion_embedding = nn.Embedding(emotion_vocab_size, d_model // 2)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, joint_states, emotion_ids):
        """
        joint_states: (batch_size, seq_len, num_joints)
        emotion_ids:  (batch_size, seq_len) or (batch_size,) if emotion is constant
        """
        joint_emb = self.joint_proj(joint_states)  # (B, S, d_model//2)
        
        if emotion_ids.ndim == 2:
            emo_emb = self.emotion_embedding(emotion_ids)  # (B, S, d_model//2)
        else:
            emo_emb = self.emotion_embedding(emotion_ids).unsqueeze(1)  # (B, 1, d_model//2)
            emo_emb = emo_emb.expand(-1, joint_states.size(1), -1)  # (B, S, d_model//2)

        x = torch.cat([joint_emb, emo_emb], dim=-1)  # (B, S, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (B, S, d_model)
        return self.output_head(x)  # e.g., feed into another module for action prediction or feature extraction

# Example
if __name__ == "__main__":
    model = RobotTransformer(num_joints=12)
    joint_states = torch.randn(4, 20, 12)           # (batch, seq_len, num_joints)
    emotion_ids = torch.randint(0, 7, (4,))         # (batch,)
    output = model(joint_states, emotion_ids)       # (4, 20, d_model)
    print(output.shape)