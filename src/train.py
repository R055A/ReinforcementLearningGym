from action_obs import ClassicControlActionObservation, AtariActionObservation
from torch import device, cuda, tensor, optim, float, save, bool, cat, zeros
from evaluate import EvaluateClassicControlPolicy, EvaluateAtariPolicy
from config import MODEL_PATH, get_hyperparameters, is_classic_control
from gym.wrappers import AtariPreprocessing
from dqn import ClassicControlDQN, AtariDQN
from torch.nn.functional import mse_loss
from parser import ParseTrainArguments
from replay import ReplayMemory
from os.path import exists
from os import makedirs
from gym import make
from math import inf


class TrainAgentUsingDQN:

    def __init__(self, args):
        """
        Initiate train agent superclass instance
        """
        cuda.empty_cache() if cuda.is_available() else None
        self.device = device('cuda' if cuda.is_available() else 'cpu')

        # initialize environment
        self.env_name = args.env
        self.gym_env = make(id=self.env_name)

        # configure hyperparameters
        self.env_config = get_hyperparameters(env_name=self.env_name)
        self.num_episodes = self.env_config['num_episodes']
        self.train_freq = self.env_config['train_freq']
        self.train_update_freq = self.env_config['target_update_freq']
        self.evaluate_freq = args.eval_freq
        self.eval_episodes = args.num_eval_episodes

        # initialize replay memory
        self.memory = ReplayMemory(memory_size=self.env_config['memory_size'])

        # initialize observation space
        self.obs = None

        # initialize in subclass
        self.dqn = self.target_network = self.optimizer = \
            self.eval = self.action_obs = self.batch_size = self.gamma = None

    def learn(self):
        """
        Train agent using DQN
        """
        best_avg_return = -inf

        for episode in range(self.num_episodes):
            done = False
            self.obs = self.action_obs.set_obs(gym_env=self.gym_env)

            while not done:
                self.dqn.steps_done += 1

                # get action from DQN, act in the true environment
                action = self.dqn.act(observation=self.obs)
                next_obs, reward, done, info = \
                    self.action_obs.time_step(gym_env=self.gym_env,
                                              action=action)

                # preprocess incoming observation
                if not done:
                    next_obs = self.action_obs.update_obs(next_obs=next_obs,
                                                          obs=self.obs)

                # add transition to replay memory
                self.memory.push(obs=self.obs.to(self.device),
                                 action=tensor([[action]]).to(self.device),
                                 next_obs=next_obs.to(self.device)
                                 if not done else None,
                                 reward=tensor([[reward]],
                                               dtype=float).to(self.device))
                self.obs = next_obs

                if episode % self.train_freq == 0:
                    self.optimize_network()

                if episode % self.train_update_freq == 0:
                    # update target network
                    self.target_network = self.dqn

            if episode % self.evaluate_freq == 0:
                # evaluate agent
                avg_return = self.eval.run(dqn_model=self.dqn,
                                           gym_env=self.gym_env)
                print(f'Episode { episode } | Score { avg_return }')

                if avg_return >= best_avg_return:
                    best_avg_return = avg_return
                    print('New best score!')
                    save(self.dqn, f'{ MODEL_PATH }/{ self.env_name }.pt')

        # close environment
        self.gym_env.close()

    def optimize_network(self):
        """
        Optimize Q-network using batch samples from replay buffer
        """
        if len(self.memory) < self.batch_size:
            return

        obs, action, next_obs, reward = \
            self.memory.sample_transitions(batch_size=self.batch_size)

        temp_mask = tensor(data=tuple(map(lambda s: s is not None,
                                          next_obs)),
                           device=self.device,
                           dtype=bool)
        obs_batch = cat(tensors=obs).to(self.device)
        action_batch = cat(tensors=action).to(self.device)
        temp_next_obs = cat(tensors=[s for s in next_obs
                                     if s is not None]).to(self.device)
        reward_batch = cat(tensors=reward).to(self.device)

        # estimate q-values for state-action pairs (s, a)
        estimated_q_vals = self.dqn(obs_batch).gather(1, action_batch)

        # compute q-value targets
        q_val_targets = zeros(self.batch_size,
                              device=self.device)
        q_val_targets[temp_mask] = \
            self.target_network(temp_next_obs).max(1)[0].detach()
        q_vals = q_val_targets * self.gamma + reward_batch.squeeze()

        loss = mse_loss(input=estimated_q_vals.squeeze(),
                        target=q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class TrainClassicControlAgent(TrainAgentUsingDQN):

    def __init__(self, args):
        """
        Initiate train agent subclass instance for Classic Control environments
        """
        super().__init__(args=args)
        self.batch_size = self.env_config['batch_size']
        self.gamma = self.env_config['gamma']

        # initialize deep Q-network
        self.dqn = ClassicControlDQN(env_config=self.env_config,
                                     device=self.device).to(self.device)

        # initialize target Q-network
        self.target_network = \
            ClassicControlDQN(env_config=self.env_config,
                              device=self.device).to(self.device)

        # initialize optimizer
        self.optimizer = optim.Adam(params=self.dqn.parameters(),
                                    lr=self.env_config['learning_rate'])

        # initialize evaluation policy
        self.eval = \
            EvaluateClassicControlPolicy(dqn_model=self.dqn,
                                         gym_env=self.gym_env,
                                         num_episodes=self.eval_episodes,
                                         device_type=self.device,
                                         env_name=self.env_name)

        # initialize action-observation interaction with the environment
        self.action_obs = ClassicControlActionObservation(device=self.device)


class TrainAtariAgent(TrainAgentUsingDQN):

    def __init__(self, args):
        """
        Initiate train agent subclass instance for Atari game environments
        """
        super().__init__(args=args)
        screen_size = self.env_config['screen_size']
        is_grayscale_obs = self.env_config['is_grayscale_obs']
        frame_skip = self.env_config['frame_skip']
        noop_max = self.env_config['noop_max']
        rescale = self.env_config['rescale']
        obs_size = self.env_config['obs_stack_size']
        actions_map = self.env_config['actions_map']
        self.batch_size = self.env_config['batch_size']
        self.gamma = self.env_config['gamma']

        # wrap environment with AtariPreprocessing
        self.gym_env = AtariPreprocessing(env=self.gym_env,
                                          screen_size=screen_size,
                                          grayscale_obs=is_grayscale_obs,
                                          frame_skip=frame_skip,
                                          noop_max=noop_max)

        # initialize deep Q-network
        self.dqn = AtariDQN(env_config=self.env_config,
                            device=self.device).to(self.device)

        # initialize target Q-network
        self.target_network = AtariDQN(env_config=self.env_config,
                                       device=self.device).to(self.device)

        # initialize optimizer
        self.optimizer = optim.Adam(params=self.dqn.parameters(),
                                    lr=self.env_config['learning_rate'])

        # initialize evaluation policy
        self.eval = EvaluateAtariPolicy(dqn_model=self.dqn,
                                        gym_env=self.gym_env,
                                        num_episodes=self.eval_episodes,
                                        device_type=self.device,
                                        env_name=self.env_name)

        # initialize action-observation interaction with the environment
        self.action_obs = AtariActionObservation(device=self.device,
                                                 rescale=rescale,
                                                 obs_size=obs_size,
                                                 actions_map=actions_map)


def train_model():
    args = ParseTrainArguments().get_args()
    makedirs(MODEL_PATH) if not exists(MODEL_PATH) else None

    if is_classic_control(args.env):
        train_agent = TrainClassicControlAgent(args)
    else:
        train_agent = TrainAtariAgent(args)
    train_agent.learn()


if __name__ == '__main__':
    train_model()
