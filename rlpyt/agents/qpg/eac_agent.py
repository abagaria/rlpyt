
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.models.qpg.mlp import QofMuMlpModel, VMlpModel, PiMlpModel, TransitionMlpModel, InverseDynamicsMlpModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v", "inv", "trans"])


class EacAgent(BaseAgent):

    def __init__(
            self,
            ModelCls=PiMlpModel,  # Pi model.
            QModelCls=QofMuMlpModel,
            VModelCls=VMlpModel,
            InverseModelCls=InverseDynamicsMlpModel, #Inverse Dynamics Model
            TransitionModelCls=TransitionMlpModel, #Transition Dynamics Model
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None,
            v_model_kwargs=None,
            inv_model_kwargs=None,
            trans_model_kwargs=None,
            initial_model_state_dict=None,  # All models.
            action_squash=1.,  # Max magnitude (or None).
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            ):
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[256, 256])
        if v_model_kwargs is None:
            v_model_kwargs = dict(hidden_sizes=[256, 256])
        if inv_model_kwargs is None:
            inv_model_kwargs = dict(hidden_sizes=[256, 256, 256])
        if trans_model_kwargs is None:
            trans_model_kwargs = dict(hidden_sizes=[256, 256, 256])
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)
        save__init__args(locals())
        self.state_distribution = None
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # Don't let base agent try to load.
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict
        self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.v_model = self.VModelCls(**self.env_model_kwargs, **self.v_model_kwargs)
        self.target_v_model = self.VModelCls(**self.env_model_kwargs,
            **self.v_model_kwargs)
        self.target_v_model.load_state_dict(self.v_model.state_dict())
        #inverse dynamics and transition dynamics initialization
        self.inv_model = self.InverseModelCls(**self.env_model_kwargs, **self.inv_model_kwargs)
        self.trans_model = self.TransitionModelCls(**self.env_model_kwargs, **self.trans_model_kwargs)
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1

        # Gaussian distribution for predicting actions
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

        # Gaussian distribution for predicting states
        self.state_distribution = Gaussian(dim=env_spaces.observation.shape[0])

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.v_model.to(self.device)
        self.target_v_model.to(self.device)
        self.inv_model.to(self.device) # inverse
        self.trans_model.to(self.device) #transition

    def data_parallel(self):
        super().data_parallel
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        self.q1_model = DDP_WRAP(self.q1_model)
        self.q2_model = DDP_WRAP(self.q2_model)
        self.v_model = DDP_WRAP(self.v_model)
        self.inv_model = DDP_WRAP(self.inv_model)
        self.trans_model = DDP_WRAP(self.inv_model)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        v = self.v_model(*model_inputs)
        return v.cpu()

    def pi(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    def target_v(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_v = self.target_v_model(*model_inputs)
        return target_v.cpu()

    # inverse dynamics
    def inv(self, observation, prev_action, prev_reward, next_observation, new_action):
        """ Compute the log-likelihood of the action sampled from the current policy. """
        model_inputs = buffer_to((observation, prev_action, prev_reward, next_observation), device=self.device)
        mean, log_std = self.inv_model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        log_inv = self.distribution.log_likelihood(new_action, dist_info)
        return log_inv.cpu(), dist_info

    # transition dynamics
    def transition(self, observation, prev_action, prev_reward, next_observation, next_action, next_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mean, log_std = self.trans_model(*model_inputs)
        log_std = log_std.to(mean.device)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        log_transition_prob = self.state_distribution.log_likelihood(next_observation, dist_info)
        return log_transition_prob.cpu(), dist_info

    def transition_sample(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, action, prev_reward), device=self.device)
        mean, log_std = self.trans_model(*model_inputs)
        log_std = log_std.to(mean.device)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        sample_state, sample_log_trans = self.state_distribution.sample_loglikelihood(dist_info)
        return sample_state, sample_log_trans, dist_info

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_v_model, self.v_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q1=self.q1_model, q2=self.q2_model,
            v=self.v_model,inv=self.inv_model, trans=self.trans_model)

    def pi_parameters(self):
        return self.model.parameters()

    def q1_parameters(self):
        return self.q1_model.parameters()

    def q2_parameters(self):
        return self.q2_model.parameters()

    def v_parameters(self):
        return self.v_model.parameters()

    def inv_parameters(self):
        return self.inv_model.parameters()

    def trans_parameters(self):
        return self.trans_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q1_model.train()
        self.q2_model.train()
        self.v_model.train()
        self.inv_model.train()
        self.trans_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        self.v_model.eval()
        self.inv_model.eval()
        self.trans_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        self.v_model.eval()
        self.inv_model.eval()
        self.trans_model.eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            q1_model=self.q1_model.state_dict(),
            q2_model=self.q2_model.state_dict(),
            v_model=self.v_model.state_dict(),
            target_v_model=self.target_v_model.state_dict(),
            inv_model=self.inv_model.state_dict(),
            trans_model=self.trans_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q1_model.load_state_dict(state_dict["q1_model"])
        self.q2_model.load_state_dict(state_dict["q2_model"])
        self.v_model.load_state_dict(state_dict["v_model"])
        self.target_v_model.load_state_dict(state_dict["target_v_model"])
        self.inv_model.load_state_dict(state_dict["inv_model"])
        self.trans_model.load_state_dict(state_dict["trans_model"])
