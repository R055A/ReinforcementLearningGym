from abc import ABC, abstractmethod
from torch import tensor, cat


class ActionObservation(ABC):

    def __init__(self,
                 device):
        """
        Initiate action-observation (environment interaction)
        abstract class instance
        """
        self.device = device

    @abstractmethod
    def set_obs(self, gym_env):
        """
        Initialize observation space each episode
        """
        pass

    @abstractmethod
    def time_step(self, gym_env, action):
        """
        Perform time step in the true environment
        """
        pass

    @abstractmethod
    def update_obs(self, next_obs, obs=None):
        """
        Update observation space with incoming observation
        """
        pass

    def preprocess_obs(self, data):
        """
        Preprocess observation space; add to tensor
        """
        return tensor(data=data,
                      device=self.device).float().unsqueeze(0)


class ClassicControlActionObservation(ActionObservation):

    def __init__(self,
                 device):
        """
        Initiate action-observation subclass instance
        for Classic Control environments
        """
        super().__init__(device=device)

    def set_obs(self, gym_env):
        return self.preprocess_obs(data=gym_env.reset())

    def time_step(self, gym_env, action):
        return gym_env.step(action=action.item())

    def update_obs(self, next_obs, obs=None):
        return self.preprocess_obs(next_obs)


class AtariActionObservation(ActionObservation):

    def __init__(self,
                 device,
                 rescale,
                 obs_size,
                 actions_map):
        """
        Initiate action-observation subclass instance
        for Atari games environment
        """
        super().__init__(device=device)
        self.rescale = rescale
        self.obs_size = obs_size
        self.actions_map = actions_map

    def set_obs(self, gym_env):
        obs = self.preprocess_obs(data=gym_env.reset() / self.rescale)
        return cat(tensors=self.obs_size *
                           [obs]).unsqueeze(0).to(self.device)

    def time_step(self, gym_env, action):
        return gym_env.step(action=self.actions_map[action.item()])

    def update_obs(self, next_obs, obs=None):
        next_obs = self.preprocess_obs(data=next_obs / self.rescale)
        return cat(tensors=(obs[:, 1:, ...],
                            next_obs.unsqueeze(1)), dim=1).to(self.device)
