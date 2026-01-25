# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from collections import deque
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        self.action_buffer = deque(maxlen=4)  # For temporal ensembling
        
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def _temporal_ensemble_actions(self, window_size: int = 4) -> torch.Tensor:
        """
        Averages the overlapping action chunks stored in self.action_buffer.
        
        Algorithm:
        1. We have a buffer of up to `window_size` previous action chunks.
           Each chunk covers timesteps [t, t+horizon].
        2. We want to produce the averaged action for the *current* execution window.
           The 'latest' chunk in the buffer corresponds to the current inference at time t.
        3. Simple approach:
           - Align all chunks in time.
           - Average the predictions for the steps that we are about to execute.
           
        However, the simple "average overlapping parts" is complex because we just want the *next* k actions.
        Each chunk starts at a different timestep.
        
        Assumption: The client requests actions sequentially.
        If we just predicted for t=100 (horizon 64), we have actions for 100..163.
        Previous prediction was at t=92 (if interval=8), covering 92..155.
        
        To simplify, we will just average the *first* `actions_per_chunk` actions of the *latest* chunk 
        with the corresponding actions from previous chunks.
        
        Let's say actions_per_chunk = 8.
        - Chunk 0 (latest): Starts at T. Action[0] is for T.
        - Chunk 1 (prev): Starts at T-8. Action[8] is for T.
        - Chunk 2 (prev): Starts at T-16. Action[16] is for T.
        
        We need to average:
        Output[i] = Mean(Chunk_0[i], Chunk_1[i+8], Chunk_2[i+16], ...)
        for i in range(actions_per_chunk).
        """
        if not self.action_buffer:
            return None
            
        latest_chunk = self.action_buffer[-1] # Shape (Horizon, ActionDim)
        
        # We only need to return 'actions_per_chunk' steps.
        # Initialize accumulator for these steps.
        # Note: latest_chunk might have batch dim if not careful, but _predict_action_chunk returns list[TimedAction]
        # Wait, _predict_action_chunk returns list[TimedAction], which is already timed! 
        # But here we are intercepting INSIDE _predict_action_chunk or GetActions? 
        # The prompt plan said "Modify _predict_action_chunk".
        # But _predict_action_chunk returns TimedAction list.
        # Let's adjust where we do this. 
        # Ideally, we do this on Tensors before converting to TimedAction.
        
        # Let's look at _predict_action_chunk. It calls self._get_action_chunk(observation).
        # self._get_action_chunk returns a Tensor (B, Chunk, Dim).
        # Then it post-processes it.
        
        # ACTUALLY, sticking to the plan: modify PolicyServer to contain the deque. 
        # This function is a helper.
        pass

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            
            # --- MODIFIED: Temporal Ensembling ---
            # 1. Get the raw prediction (Chunk, Dim) - skipping post-processing for a moment?
            # No, _predict_action_chunk does a lot (prepare, preprocess, infer, postprocess, time).
            # To do ensembling cleanly, we should ideally ensemble the *postprocessed* tensors (in physical space),
            # OR the raw tensors (in normalized space). Normalized is better usually.
            # But _predict_action_chunk does everything in one go.
            
            # Let's modify _predict_action_chunk to handle ensembling internally or breakup the steps.
            # Or better, we can just ensemble the *final* TimedAction values? 
            # No, TimedAction is a list of objects.
            
            # Let's modify _predict_action_chunk to support ensembling.
            action_chunk = self._predict_action_chunk(obs, use_ensemble=True)
            
            # --- END MODIFIED ---
            
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        # IMPORTANT: For ensembling, we want the FULL horizon, not just the executed part.
        # But existing code slices it: return chunk[:, : self.actions_per_chunk, :]
        # We need to change this if we want to ensemble!
        # If we return the full chunk here, the postprocessor will process the full chunk.
        
        # However, modifying _get_action_chunk might break other things if they expect only executed actions?
        # But _predict_action_chunk uses it.
        
        # Let's return the FULL chunk here, and slice later after ensembling.
        # The existing code did: return chunk[:, : self.actions_per_chunk, :]
        # changing to:
        return chunk 

    def _predict_action_chunk(self, observation_t: TimedObservation, use_ensemble: bool = False) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        
        # Get FULL chunk (B, Horizon, Dim)
        action_tensor_full = self._get_action_chunk(observation) 
        
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor_full.shape}"
        )
    
        # --- ENSEMBLING LOGIC ---
        if use_ensemble and hasattr(self, 'action_buffer'):
            # action_tensor_full is (B, Horizon, Dim). We assume B=1.
            current_pred = action_tensor_full.detach().cpu() # Move to CPU for buffer storage if needed, or keep GPU
            
            # Store in buffer
            self.action_buffer.append(current_pred)
            
            # Perform Weighted Average
            # We need to align the overlapping predictions.
            # Current time: t. 
            # We want actions for [t, t + actions_per_chunk].
            
            # Buffer[-1] starts at t.
            # Buffer[-2] starts at t - actions_per_chunk (approx, assuming fixed rate).
            # Buffer[-3] starts at t - 2*actions_per_chunk.
            
            # Let k = actions_per_chunk.
            # We want output for t+0, t+1, ... t+k-1.
            
            # For a given step 'i' in the output (0 <= i < k):
            # It corresponds to time T = t + i.
            
            # From Buffer[-1] (start t): Index is i.
            # From Buffer[-2] (start t-k): Index is k + i.
            # From Buffer[-3] (start t-2k): Index is 2k + i.
            
            # General formula: For Buffer[-1-j], the index is j*k + i.
            # We check if index < Horizon.
            
            k = self.actions_per_chunk
            horizon = current_pred.shape[1]
            
            ensembled_actions = []
            
            # Iterate over the steps we want to EXECUTE (0 to k-1)
            for i in range(k):
                valid_preds = []
                # Look back in buffer
                for j in range(len(self.action_buffer)):
                    # self.action_buffer is a deque.
                    # self.action_buffer[-1] is latest (j=0 in formula above).
                    # self.action_buffer[-2] is previous (j=1).
                    # So we iterate backwards? Or just iterate the deque directly?
                    # Let's iterate index `j` where 0 is latest.
                    
                    chunk_idx = len(self.action_buffer) - 1 - j # index in deque (0 is oldest, -1 is latest)
                    chunk = self.action_buffer[chunk_idx] 
                    
                    # We want the prediction for time (t + i)
                    # This chunk started at time (t - j*k)
                    # So the time difference is (t+i) - (t-j*k) = i + j*k
                    
                    pred_idx = i + j * k
                    
                    if pred_idx < horizon:
                         # Append tensor: (B, Dim) -> (Dim) if B=1
                         valid_preds.append(chunk[0, pred_idx])
                
                # Average
                if valid_preds:
                    # Stack: (N, Dim) -> Mean -> (Dim)
                    mean_action = torch.stack(valid_preds).mean(dim=0)
                    ensembled_actions.append(mean_action)
                else:
                    # Should not happen if horizon >= k
                    ensembled_actions.append(current_pred[0, i])
            
            # Stack back to (1, k, Dim)
            # ensembled_actions is list of (Dim)
            action_tensor = torch.stack(ensembled_actions).unsqueeze(0) # (1, k, Dim)
            
            # Ensure it's on the right device for post-processing ???
            # Postprocessor usually handles device. But we detached to CPU potentially.
            action_tensor = action_tensor.to(self.device)

        else:
             # No ensembling, just take the first k steps
             action_tensor = action_tensor_full[:, : self.actions_per_chunk, :]

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
