import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def extract_emotion(text):
    """Extract emotion from text description."""
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

class TrajectoryFFTLabelDataset(Dataset):
    """Dataset for trajectory FFT features with emotion labels for LeRobot."""
    
    def __init__(self, 
                 trajectories=None, 
                 task_descriptions=None,
                 episode_ids=None,
                 data_path=None,
                 fps=30, 
                 window_sec=1.5, 
                 overlap=0.5,
                 max_freq=5.0):
        """
        Initialize the dataset.
        
        Args:
            trajectories: List of trajectory arrays with shape [num_timesteps, num_joints] 
                         or a single array with shape [num_episodes, num_timesteps, num_joints]
            task_descriptions: List of task descriptions for emotion extraction
            episode_ids: List of episode IDs (optional)
            data_path: Path to data directory if trajectories not provided directly
            fps: Frames per second of the trajectory data
            window_sec: Window size in seconds for FFT computation
            overlap: Overlap between consecutive windows (0.0-1.0)
            max_freq: Maximum frequency to keep in FFT features (Hz)
        """
        super().__init__()
        
        # Set parameters
        self.fps = fps
        self.window_size = int(fps * window_sec)
        self.step_size = int(self.window_size * (1 - overlap))
        self.max_freq = max_freq
        
        # Define emotion categories
        self.emotions = ['sad', 'surprised', 'happy', 'angry', 'fearful', 'curious', 'playful']
        
        # Load data if not provided
        if trajectories is None:
            if data_path is None:
                raise ValueError("Either trajectories or data_path must be provided")
            trajectories, task_descriptions, episode_ids = self._load_data(data_path)
        
        # Process data
        self.samples = []
        self._process_trajectories(trajectories, task_descriptions, episode_ids)
        
    def _load_data(self, data_path):
        """
        Load trajectory data from files.
        Can be customized based on your specific file format.
        
        Returns:
            trajectories: List of trajectory arrays
            task_descriptions: List of task descriptions
            episode_ids: List of episode IDs
        """
        # This is a placeholder - implement based on your file format
        # Example implementation for numpy files:
        episode_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npy')])
        meta_file = os.path.join(data_path, 'metadata.npy')
        
        trajectories = []
        task_descriptions = []
        episode_ids = []
        
        # Load metadata if exists
        if os.path.exists(meta_file):
            metadata = np.load(meta_file, allow_pickle=True).item()
            task_descriptions = metadata.get('tasks', [])
            episode_ids = metadata.get('episode_ids', [])
        
        # Load trajectories
        for i, file in enumerate(episode_files):
            traj = np.load(os.path.join(data_path, file))
            trajectories.append(traj)
            
            # If metadata doesn't contain enough entries, create placeholders
            if i >= len(task_descriptions):
                task_descriptions.append(f"Unknown task {i}")
            if i >= len(episode_ids):
                episode_ids.append(i)
        
        return trajectories, task_descriptions, episode_ids
        
    def _process_trajectories(self, trajectories, task_descriptions, episode_ids=None):
        """
        Process trajectories and prepare samples.
        
        Args:
            trajectories: List of trajectory arrays with shape [num_timesteps, num_joints]
                          or a single array with shape [num_episodes, num_timesteps, num_joints]
            task_descriptions: List of task descriptions for emotion extraction
            episode_ids: List of episode IDs (optional)
        """
        # Handle different input formats
        if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
            # Single array with shape [num_episodes, num_timesteps, num_joints]
            traj_list = [trajectories[i] for i in range(trajectories.shape[0])]
        else:
            # List of arrays with shape [num_timesteps, num_joints]
            traj_list = trajectories
            
        # Create default episode IDs if not provided
        if episode_ids is None:
            episode_ids = list(range(len(traj_list)))
            
        # Process each trajectory
        for i, trajectory in enumerate(traj_list):
            if i >= len(task_descriptions):
                # Skip if no task description available
                continue
                
            # Extract emotion from task description
            emotion_str = extract_emotion(task_descriptions[i])
            
            # Skip unknown emotions
            if emotion_str not in self.emotions:
                continue
                
            # Get emotion index
            emotion_idx = self.emotions.index(emotion_str)
            
            # Get episode ID
            episode_id = episode_ids[i] if i < len(episode_ids) else i
            
            # Ensure trajectory is numpy array
            traj_array = np.asarray(trajectory)
            
            # Get number of joints
            num_joints = traj_array.shape[1] if traj_array.ndim > 1 else 1
            
            # Create windows with specified overlap
            n_samples = len(traj_array)
            
            for start in range(0, n_samples - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window = traj_array[start:end]
                
                # Get current joint state (input)
                state = traj_array[start].astype(np.float32)
                
                # Compute FFT features for the window (output)
                fft_features = self._compute_fft(window)
                
                # Store sample
                self.samples.append({
                    "emotion": emotion_str,
                    "emotion_idx": emotion_idx,
                    "state": state,
                    "fft_features": fft_features,
                    "episode_id": episode_id
                })
    
    def _compute_fft(self, window):
        """
        Compute FFT features for a window of joint positions.
        
        Args:
            window: Array of shape [window_size, num_joints]
            
        Returns:
            2D array of FFT amplitudes with shape [num_joints, num_freq_bins]
        """
        # Get frequency bins
        freqs = np.fft.rfftfreq(self.window_size, d=1/self.fps)
        
        # Apply max frequency filter if specified
        if self.max_freq is not None:
            freq_mask = freqs <= self.max_freq
        else:
            freq_mask = slice(None)  # Keep all frequencies
        
        # Transpose if needed to get [window_size, num_joints]
        if window.ndim == 1:
            # Handle single joint case
            window = window.reshape(-1, 1)
        
        num_joints = window.shape[1]
        fft_features = np.zeros((num_joints, np.sum(freq_mask) if isinstance(freq_mask, np.ndarray) else len(freqs)))
        
        # Compute FFT for each joint
        for j in range(num_joints):
            signal = window[:, j]
            fft_vals = np.fft.rfft(signal)
            fft_features[j] = np.abs(fft_vals)[freq_mask]
            
        return fft_features.astype(np.float32)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Returns dictionary with:
            - inputs: Tensor with current joint state
            - targets: Tensor with FFT features [num_joints, num_freq_bins]
            - emotion_idx: Emotion index for one-hot encoding later
        """
        sample = self.samples[idx]
        
        # Just use joint state for input
        input_vec = sample["state"].astype(np.float32)
        
        # Use FFT features as target (already in shape [num_joints, num_freq_bins])
        output_mat = sample["fft_features"]
        
        # Return tensors with proper keys for LeRobot compatibility
        return {
            "inputs": torch.tensor(input_vec),
            "targets": torch.tensor(output_mat),
            "emotion_idx": sample["emotion_idx"],
            "metadata": {
                "emotion": sample["emotion"],
                "episode_id": sample["episode_id"]
            }
        }
    
    def get_emotion_distribution(self):
        """Get the distribution of emotions in the dataset."""
        emotions = [s["emotion"] for s in self.samples]
        counts = {}
        for emotion in emotions:
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts
    
    def get_emotion_indices(self):
        """Get a mapping of emotion names to indices."""
        return {emotion: idx for idx, emotion in enumerate(self.emotions)}
    
    def get_input_dim(self):
        """Get input dimension (state)."""
        if len(self.samples) > 0:
            return len(self.samples[0]["state"])
        return 0
    
    def get_output_shape(self):
        """Get output shape (FFT features) as [num_joints, num_freq_bins]."""
        if len(self.samples) > 0:
            return self.samples[0]["fft_features"].shape
        return (0, 0)


# Example usage
if __name__ == "__main__":
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
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example batch iteration
    for batch in dataloader:
        inputs = batch["inputs"]
        targets = batch["targets"]
        emotion_idx = batch["emotion_idx"]
        metadata = batch["metadata"]

        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch targets shape: {targets.shape}")
        print(f"Batch emotion indices: {emotion_idx[:5]}")
        print(f"Batch emotions: {metadata['emotion'][:5]}")
        break