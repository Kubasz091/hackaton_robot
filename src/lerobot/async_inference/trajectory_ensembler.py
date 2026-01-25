import torch

class TrajectoryEnsembler:
    """
    Implements temporal ensembling for trajectory predictions.
    Maintains a running average of overlapping action chunks predicted at different timesteps.
    """
    def __init__(self):
        # Stores cumulative sum of actions for each timestep: {timestep: tensor(action_dim)}
        self.action_sums = {}
        # Stores count of predictions for each timestep: {timestep: count}
        self.action_counts = {}
        # Track the latest timestep we've seen to help with cleanup
        self.latest_timestep = -1

    def update(self, action_chunk: torch.Tensor, start_timestep: int):
        """
        Update the ensembler with a new chunk of predicted actions.
        
        Args:
            action_chunk: Tensor of shape (chunk_size, action_dim)
            start_timestep: The timestep corresponding to the first action in the chunk
        """
        chunk_size = action_chunk.shape[0]
        
        # Determine device from input tensor to ensure compatibility
        device = action_chunk.device
        
        chunk_indices = range(start_timestep, start_timestep + chunk_size)
        
        for i, t in enumerate(chunk_indices):
            if t not in self.action_sums:
                self.action_sums[t] = torch.zeros_like(action_chunk[i], device=device)
                self.action_counts[t] = 0
            
            # If device mismatch (e.g. historical data on different device), move it
            if self.action_sums[t].device != device:
                self.action_sums[t] = self.action_sums[t].to(device)
            
            self.action_sums[t] += action_chunk[i]
            self.action_counts[t] += 1
            
        self.latest_timestep = max(self.latest_timestep, start_timestep + chunk_size - 1)
        
        # Cleanup old timesteps (optional optimization: remove steps older than start_timestep)
        # We can safely remove anything that strictly precedes start_timestep, as we are unlikely
        # to query it again for real-time control (assuming monotonic time progress).
        # However, to be safe and simple, we can just cleanup rarely or let it grow if memory isn't an issue.
        # For a long running server, cleanup is important.
        
        # Simple cleanup: remove timesteps older than (start_timestep - buffer)
        # Assuming we don't need history too far back.
        cleanup_threshold = start_timestep - 100 # Keep a buffer
        keys_to_remove = [k for k in self.action_sums.keys() if k < cleanup_threshold]
        for k in keys_to_remove:
            del self.action_sums[k]
            del self.action_counts[k]

    def get_action_chunk(self, start_timestep: int, length: int) -> torch.Tensor | None:
        """
        Retrieve the ensembled action chunk.
        
        Args:
            start_timestep: The starting timestep of the requested chunk.
            length: The number of steps to return.
            
        Returns:
            Tensor of shape (length, action_dim) containing the averaged actions.
            Returns None if any timestep in the requested range is missing (incomplete coverage).
            Alternatively, returns best-effort average for what is available, or waits?
            
            For a real-time policy server, we usually predict "future" actions. 
            If we don't have enough data for the full length (e.g. at the very end), we might return what we have.
            But typically, the policy ensures we have a chunk.
        """
        
        # Collect averaged actions
        averaged_actions = []
        
        for t in range(start_timestep, start_timestep + length):
            if t not in self.action_sums:
                # Missing data for this timestep. 
                # This might happen if 'update' hasn't been called enough times or logic is off.
                # For now, let's treat it as "cannot form full chunk" or return partial?
                # Ideally, the caller ensures they only ask for what's reasonable.
                # But if we just started, we might not have overlaps for everything.
                return None 
            
            avg_action = self.action_sums[t] / self.action_counts[t]
            averaged_actions.append(avg_action)
            
        if not averaged_actions:
            return None
            
        return torch.stack(averaged_actions)
