from torch.nn import Module, Sequential, Conv2d, \
    ReLU, Flatten, Linear, BatchNorm2d, Dropout
from random import random, randrange
from torch import tensor, argmax


class DQN(Module):

    def __init__(self,
                 env_config,
                 device):
        """
        Initiate DQN superclass instance
        """
        super(DQN, self).__init__()
        self.device = device

        # hyperparameters
        self.num_actions = env_config['num_actions']
        self.anneal_length = env_config['anneal_length']
        self.eps_start = env_config['eps_start']
        self.eps_end = env_config['eps_end']
        self.steps_done = 0

        # NN model - initialize in subclass
        self.model = None

    def forward(self, input_data):
        """
        Runs forward pass of NN
        """
        return self.model(input_data)

    def act(self,
            observation,
            exploit=False):
        """
        Selects an action using epsilon-greedy exploration strategy and DQN
        """
        eps_greedy = max(self.eps_start -
                         self.steps_done *
                         (self.eps_start - self.eps_end) /
                         self.anneal_length,
                         self.eps_end)
        if not exploit and random() < eps_greedy:
            # return random action using epsilon-greedy exploration
            return tensor(data=randrange(self.num_actions),
                          device=self.device)
        # return action using DQN selection
        return argmax(input=self.forward(input_data=observation)[0])


class ClassicControlDQN(DQN):

    def __init__(self,
                 env_config,
                 device):
        """
        Initiate DQN subclass instance for Classic Control environments
        """
        super().__init__(env_config=env_config,
                         device=device)
        self.model = Sequential(Linear(in_features=env_config['obs_size'],
                                       out_features=256),
                                ReLU(),
                                # Linear(in_features=256,
                                #        out_features=128),
                                # ReLU(),
                                # Linear(in_features=128,
                                #        out_features=64),
                                # ReLU(),
                                Linear(in_features=256,
                                       out_features=self.num_actions),
                                Flatten())


class AtariDQN(DQN):

    def __init__(self,
                 env_config,
                 device):
        """
        Initiate DQN subclass instance for Atari game environments
        """
        super().__init__(env_config=env_config,
                         device=device)
        self.obs_size = env_config['obs_stack_size']
        self.model = Sequential(Conv2d(in_channels=4,
                                       out_channels=32,
                                       kernel_size=8,
                                       stride=4),
                                # BatchNorm2d(num_features=32),
                                ReLU(),
                                # Dropout(),
                                Conv2d(in_channels=32,
                                       out_channels=64,
                                       kernel_size=4,
                                       stride=2),
                                # BatchNorm2d(num_features=64),
                                ReLU(),
                                # Dropout(),
                                Conv2d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=3),
                                # BatchNorm2d(num_features=64),
                                ReLU(),
                                Flatten(),
                                Linear(in_features=3136,
                                       out_features=512),
                                ReLU(),
                                Linear(in_features=512,
                                       out_features=self.num_actions))
