import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class TrajectoryFFTLabelDataset(LeRobotDataset):
    def __init__(self, df, fps=30, window_sec=1.5):
        
        self.fps = fps
        self.window_size = int(fps * window_sec)
        self.joint_cols = [col for col in df.columns if col.startswith("joint_")]
        self.df = df.reset_index(drop=True)
        
        self.emotions = ['sad', 'surprised', 'happy', 'angry', 'fearful', 'curious', 'playful']
        self.episodes = self.df.groupby("episode_id")
        self.samples = []

        for episode_id, ep_df in self.episodes:
            emotion_str = extract_emotion(ep_df["single_task"].iloc[0])
            if emotion_str not in self.emotions:
                continue
            emotion_idx = self.emotions.index(emotion_str)
            emotion_vec = np.eye(len(self.emotions))[emotion_idx]

            for i in range(len(ep_df) - self.window_size):
                state = ep_df.iloc[i][self.joint_cols].values.astype(np.float32)
                window = ep_df.iloc[i:i + self.window_size]
                label_fft = self.compute_fft(window)
                self.samples.append((emotion_vec, state, label_fft))

    def compute_fft(self, window):
        fft_features = []
        for col in self.joint_cols:
            signal = window[col].values
            fft_vals = np.fft.rfft(signal)
            amplitudes = np.abs(fft_vals)
            fft_features.append(amplitudes)
        return np.concatenate(fft_features).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emotion_vec, state, label_fft = self.samples[idx]
        input_vec = np.concatenate([emotion_vec, state]).astype(np.float32)
        return {"": torch.tensor(input_vec), "": torch.tensor(label_fft)}

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_robot_data(base_dir="/home/rgarciap/Data2/lerobot_emotions/", chunk_dir="data/chunk-000", episodes_file="meta/episodes.jsonl"):
    episodes_path = os.path.join(base_dir, episodes_file)
    chunk_path = os.path.join(base_dir, chunk_dir)

    df_episodes = pd.read_json(episodes_path, lines=True)
    df_episodes["episode_id"] = df_episodes["episode_index"]
    df_episodes["single_task"] = df_episodes["tasks"].apply(lambda x: x[0])

    all_dfs = []
    for _, row in df_episodes.iterrows():
        episode_id = row["episode_id"]
        single_task = row["single_task"]

        parquet_file = f"episode_{episode_id:06d}.parquet"
        parquet_path = os.path.join(chunk_path, parquet_file)

        if not os.path.exists(parquet_path):
            print(f"Warning: File not found: {parquet_path}")
            continue

        df_temp = pd.read_parquet(parquet_path)
        df_temp["episode_id"] = episode_id
        df_temp["single_task"] = single_task
        all_dfs.append(df_temp)

    if not all_dfs:
        raise RuntimeError("No episodes loaded. Check file paths and metadata.")

    df_final = pd.concat(all_dfs, ignore_index=True)

    # Expand observation.state into joint columns
    joint_vectors = df_final["observation.state"].tolist()
    num_joints = len(joint_vectors[0])
    joint_cols = [f"joint_{i}" for i in range(num_joints)]
    df_joints = pd.DataFrame(joint_vectors, columns=joint_cols)
    df_final = pd.concat([df_final.reset_index(drop=True), df_joints], axis=1)

    return df_final

if __name__ == "__main__":
    df_raw = load_robot_data()
    dataset = TrajectoryFFTLabelDataset(df_raw, fps=30, window_sec=1.5)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Example iteration
    for x, y in dataloader:
        print(x[0])
        print("Input shape:", x.shape)  # [B, num_emotions + num_joints]
        print("Label shape:", y.shape)  # [B, fft_features]
        break