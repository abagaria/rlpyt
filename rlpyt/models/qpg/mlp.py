
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class MuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            output_max=1,
            ):
        super().__init__()
        self._output_max = output_max
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        mu = self._output_max * torch.tanh(self.mlp(observation.view(T * B, -1)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class PiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

#implementation of inverse dynamics model
class InverseDynamicsMlpModel(torch.nn.Module):
    def __init__(self, observation_shape, hidden_sizes, action_size):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.mlp = MlpModel(
            input_size=2*int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward, next_observation):

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        #concatenate observation s and next_observation s'
        input_mlp = torch.cat([observation.view(T * B, -1), next_observation.view(T * B, -1)], dim=1)
        output = self.mlp(input_mlp)
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std

#implementation of transition model
class TransitionMlpModel(torch.nn.Module):
    def __init__(self, observation_shape, hidden_sizes, action_size, std=1e-5):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self._action_size = action_size
        self.log_std = np.log(std) * torch.ones(action_size, 1)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=int(np.prod(observation_shape))
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        T_input = torch.cat([observation.view(T * B, -1), prev_action.view(T * B, -1)], dim=1)
        t = self.mlp(T_input)
        t = restore_leading_dims(t, lead_dim, T, B)
        return t, self.log_std

class QofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class VMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size=None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)),
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        v = self.mlp(observation.view(T * B, -1)).squeeze(-1)
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
