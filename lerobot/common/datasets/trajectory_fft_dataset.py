import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.data import LeRobotDataset

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

class TrajectoryFFTLabelDataset(LeRobotDataset):
    """Dataset for trajectory FFT features with emotion labels for LeRobot."""
    
    def __init__(self, 
                 repo_id,
                 root=None,
                 episodes=None,
                 delta_timestamps=None,
                 image_transforms=None,
                 revision=None,
                 video_backend=None,
                 fps=30, 
                 window_sec=1.5, 
                 overlap=0.5,
                 max_freq=5.0,
                 **kwargs):
        """
        Initialize the dataset.
        
        Args:
            repo_id: Repository ID for LeRobot dataset
            root: Root directory for LeRobot dataset
            episodes: Episodes to include in LeRobot dataset
            delta_timestamps: Delta timestamps for LeRobot dataset
            image_transforms: Image transforms for LeRobot dataset
            revision: Revision for LeRobot dataset
            video_backend: Video backend for LeRobot dataset
            fps: Frames per second of the trajectory data
            window_sec: Window size in seconds for FFT computation
            overlap: Overlap between consecutive windows (0.0-1.0)
            max_freq: Maximum frequency to keep in FFT features (Hz)
            **kwargs: Additional arguments passed to LeRobotDataset
        """
        # Initialize the parent LeRobotDataset with required parameters
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=revision,
            video_backend=video_backend,
            **kwargs
        )
        
        # Set FFT parameters
        self.fps = fps
        self.window_size = int(fps * window_sec)
        self.step_size = int(self.window_size * (1 - overlap))
        self.max_freq = max_freq
        
        # Define emotion categories
        self.emotions = ['sad', 'surprised', 'happy', 'angry', 'fearful', 'curious', 'playful']
        
        # Initialize samples list - will be populated when data is accessed
        self.samples = []
        self.processed = False
        
    def _process_data(self):
        """
        Process LeRobot data to extract trajectories and prepare FFT samples.
        This method is called lazily when data is first accessed.
        """
        if self.processed:
            return
            
        # Get episodes from the parent dataset
        episodes = super().episodes
        
        for episode_idx, episode in enumerate(episodes):
            # Extract task description
            task_desc = episode.get('task', '')
            if isinstance(task_desc, list) and len(task_desc) > 0:
                task_desc = task_desc[0]
                
            # Extract emotion from task description
            emotion_str = extract_emotion(task_desc)
            
            # Skip unknown emotions
            if emotion_str not in self.emotions:
                continue
                
            # Get emotion index
            emotion_idx = self.emotions.index(emotion_str)
            
            # Extract trajectory from episode
            trajectory = self._extract_trajectory_from_episode(episode)
            
            # Skip if not enough data points
            if len(trajectory) < self.window_size:
                continue
                
            # Create windows with specified overlap
            n_samples = len(trajectory)
            
            for start in range(0, n_samples - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window = trajectory[start:end]
                
                # Get current joint state (input)
                state = trajectory[start].astype(np.float32)
                
                # Compute FFT features for the window (output)
                fft_features = self._compute_fft(window)
                
                # Store sample
                self.samples.append({
                    "emotion": emotion_str,
                    "emotion_idx": emotion_idx,
                    "state": state,
                    "fft_features": fft_features,
                    "episode_idx": episode_idx
                })
                
        self.processed = True
        
    def _extract_trajectory_from_episode(self, episode):
        """
        Extract joint trajectory from an episode.
        Override this method based on your specific episode data structure.
        
        Args:
            episode: Episode data from LeRobotDataset
            
        Returns:
            Array of shape [num_timesteps, num_joints]
        """
        # This is a placeholder - implement based on your LeRobot episode structure
        # For example, if episodes contain an 'observation.state' key:
        if 'observations' in episode and len(episode['observations']) > 0:
            if 'state' in episode['observations'][0]:
                return np.array([obs['state'] for obs in episode['observations']])
                
        # Another possibility - directly accessing 'observation.state' if structured as in the uploaded code
        if 'observation.state' in episode:
            return np.array(episode['observation.state'])
            
        # If joint data is stored in episode['joint_positions']
        if 'joint_positions' in episode:
            return np.array(episode['joint_positions'])
            
        # If the episode itself is a sequence of observation states
        if hasattr(episode, 'shape') and len(episode.shape) == 2:
            return episode
            
        # Fallback - you'll need to implement this based on your data structure
        raise ValueError("Could not extract trajectory from episode. Please implement _extract_trajectory_from_episode method.")
    
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
        # Process data lazily if not already done
        if not self.processed:
            self._process_data()
            
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Returns dictionary with:
            - inputs: Tensor with current joint state
            - targets: Tensor with FFT features [num_joints, num_freq_bins]
            - emotion_idx: Emotion index for one-hot encoding later
        """
        # Process data lazily if not already done
        if not self.processed:
            self._process_data()
            
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
                "episode_idx": sample["episode_idx"]
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
    # Example usage with LeRobot configuration
    import argparse
    from omegaconf import OmegaConf
    
    # Create a sample configuration
    cfg = OmegaConf.create({
        "dataset": {
            "repo_id": "lerobot/emotion_trajectories",
            "root": "./data",
            "episodes": None,  # Use all episodes
            "revision": "main",
            "video_backend": "opencv"
        }
    })
    
    # Create dataset
    dataset = TrajectoryFFTLabelDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=None,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
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