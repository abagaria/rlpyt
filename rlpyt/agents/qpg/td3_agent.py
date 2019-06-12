
from rlpyt.agents.dpg.ddpg_agent import DdpgAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.independent_gaussian import Gaussian
from rlpyt.models.utils import update_state_dict


class Td3Agent(DdpgAgent):

    def __init__(
            self,
            pretrain_std=2.,  # To make actions roughly uniform.
            target_noise_std=0.2,
            target_noise_clip=0.5,
            initial_q2_model_state_dict=None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.initial_q2_model_state_dict = initial_q2_model_state_dict
        self.pretrain_std = pretrain_std
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.q2_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        if self.initial_q1_model_state_dict is not None:
            self.q2_model.load_state_dict(self.initial_q2_model_state_dict)
        self.target_q2_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())
        self.target_distribution = Gaussian(dim=env_spec.action_space.size,
            std=self.target_noise_std, noise_clip=self.target_noise_clip,
            clip=env_spec.action_space.high)  # Assume symmetric low=-high.
        self.agent.give_min_itr_learn(self.min_itr_learn)

    def initialize_cuda(self, cuda_idx=None):
        super().initialize_cuda(cuda_idx)
        if cuda_idx is None:
            return
        self.q2_model.to(self.device)
        self.target_q2_model.to(self.device)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q1 = self.q_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def target_q_at_mu(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_mu = self.target_mu_model(*model_inputs)
        target_action = self.target_distribution.sample(target_mu)
        target_q1_at_mu = self.target_q_model(*model_inputs, target_action)
        target_q2_at_mu = self.target_q2_model(*model_inputs, target_action)
        return target_q1_at_mu.cpu(), target_q2_at_mu.cpu()

    def update_target(self, tau=1):
        super().update_target(tau)
        update_state_dict(self.target_q2_model, self.q2_model, tau)

    def q_parameters(self):
        yield from self.q_model.parameters()
        yield from self.q2_model.parameters()

    def set_target_noise(self, std, noise_clip=None):
        self.target_distribution.set_std(std)
        self.target_distribution.set_noise_clip(noise_clip)

    def train_mode(self, itr):
        super().train_mode()
        self.q2_model.train()

    def sample_mode(self, itr):
        super().sample_mode()
        self.q2_model.eval()
        std = self.action_std if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)

    def eval_mode(self, itr):
        super().eval_mode()
        self.q2_model.eval()
