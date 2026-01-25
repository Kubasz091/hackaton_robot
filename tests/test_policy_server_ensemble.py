
import unittest
import torch
import time
import pickle
from collections import deque
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from lerobot.async_inference.policy_server import PolicyServer
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedObservation, RemotePolicyConfig

# Mock for services_pb2
class MockServices:
    class Empty:
        pass
    class Actions:
        def __init__(self, data):
            self.data = data

@dataclass
class MockConfig:
    image_features = {}
    
class MockPolicy:
    def __init__(self, pretrained_path=None):
        self.config = MockConfig()
        
    def to(self, device):
        pass
        
    def predict_action_chunk(self, batch):
        # Return a constant chunk for testing, or identifiable pattern
        # Shape: (B, Horizon, Dim)
        # batch is a dict
        
        # Let's say we return a chunk where all values are equal to the 'timestamp' input 
        # (if we could pass it) or just a counter.
        # Ideally we want to verify averaging.
        
        # If we return a chunk of all 1s, then all 2s.
        # Average should be 1.5.
        
        pass

class TestPolicyServerEnsembling(unittest.TestCase):
    def setUp(self):
        self.config = PolicyServerConfig(host="localhost", port=9999)
        with patch('lerobot.async_inference.policy_server.get_logger'):
            self.server = PolicyServer(self.config)
            self.server.action_buffer = deque(maxlen=4)
            
        # Manually setup server state as if "SendPolicyInstructions" was called
        self.server.policy_type = "mock"
        self.server.actions_per_chunk = 8
        self.server.device = "cpu"
        self.server.lerobot_features = {}
        
        # Mock pre/post processors
        self.server.preprocessor = MagicMock(return_value={})
        self.server.postprocessor = MagicMock(side_effect=lambda x: x) # Identity
        # property policy_image_features relies on self.policy.config.image_features
        
        # Mock policy
        self.server.policy = MagicMock()
        self.server.policy.config = MockConfig()

        
    @patch('lerobot.async_inference.policy_server.raw_observation_to_observation')
    def test_ensembling_logic(self, mock_raw_to_obs):
        """
        Verify that _predict_action_chunk correctly ensembles actions.
        We will manually inject observations and check the output actions.
        """
        # Mock raw_observation_to_observation to just return the observation dict from TimedObservation
        mock_raw_to_obs.side_effect = lambda obs, features, img_features: obs
        
        # We need to mock _get_action_chunk because that's where the raw prediction comes from
        # and we want to control it perfectly.
        
        # Scenario:
        # Horizon = 8
        # Actions per chunk = 2 (for simplicity)
        self.server.actions_per_chunk = 2
        
        # Observation 1 at t=0
        # Policy predicts all 10s.
        obs1 = TimedObservation(
             observation={"observation.state": torch.zeros(14)},
             timestamp=0.0,
             timestep=0,
             must_go=True
        )
        
        # Mock _get_action_chunk to return tensor of shape (B=1, Horizon=8, Dim=1)
        # For Obs 1, return all 10.0
        self.server._get_action_chunk = MagicMock(return_value=torch.full((1, 8, 1), 10.0))
        
        # Call _predict_action_chunk with ensembling
        # Since buffer is empty, it should just be 10.0
        actions1 = self.server._predict_action_chunk(obs1, use_ensemble=True)
        
        # actions1 should be list of TimedAction
        # We expect 2 actions (actions_per_chunk)
        self.assertEqual(len(actions1), 2)
        self.assertEqual(actions1[0].action.item(), 10.0) # t=0
        self.assertEqual(actions1[1].action.item(), 10.0) # t=1
        
        # Verify buffer
        self.assertEqual(len(self.server.action_buffer), 1)
        
        # Observation 2 at t=2 (since actions_per_chunk=2)
        # Policy predicts all 20.0
        obs2 = TimedObservation(
             observation={"observation.state": torch.zeros(14)},
             timestamp=0.1, # arbitrary
             timestep=2,
             must_go=True
        )
        self.server._get_action_chunk = MagicMock(return_value=torch.full((1, 8, 1), 20.0))
        
        actions2 = self.server._predict_action_chunk(obs2, use_ensemble=True)
        
        # Now we should have ensembling.
        # Current time t=2.
        # We output actions for t=2 and t=3.
        
        # From Buffer[0] (the previous one, started at t=0):
        # It covers t=0..7.
        # At t=2 (index 2), val is 10.0.
        # At t=3 (index 3), val is 10.0.
        
        # From Buffer[1] (current one, starts at t=2):
        # It covers t=2..9.
        # At t=2 (index 0), val is 20.0.
        # At t=3 (index 1), val is 20.0.
        
        # Average should be (10+20)/2 = 15.0
        
        self.assertEqual(len(actions2), 2)
        self.assertAlmostEqual(actions2[0].action.item(), 15.0) # t=2
        self.assertAlmostEqual(actions2[1].action.item(), 15.0) # t=3
        
        # Verify buffer
        self.assertEqual(len(self.server.action_buffer), 2)
        
        # Output info
        print("\nTest Passed: Ensembling correctly averaged 10.0 and 20.0 to 15.0")

if __name__ == '__main__':
    unittest.main()
