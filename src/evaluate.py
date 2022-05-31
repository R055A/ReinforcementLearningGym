from action_obs import ClassicControlActionObservation, AtariActionObservation
from config import is_classic_control, get_hyperparameters
from gym.wrappers import AtariPreprocessing
from parser import ParseArguments
from torch import load
from gym import make


class EvaluatePolicy:

    def __init__(self,
                 dqn_model,
                 gym_env,
                 num_episodes,
                 device_type,
                 is_render_env=False,
                 is_verbose_returns=False):
        """
        Initiate evaluate agent superclass instance
        """
        self.dqn_model = dqn_model
        self.gym_env = gym_env
        self.num_episodes = num_episodes
        self.device = device_type
        self.is_render_env = is_render_env
        self.is_verbose_returns = is_verbose_returns

        # initialize in subclass
        self.action_obs = self.obs = None

    def run(self,
            dqn_model=None,
            gym_env=None):
        """
        Evaluate agent using trained DQN model
        """
        self.dqn_model = dqn_model if dqn_model else self.dqn_model
        self.gym_env = gym_env if gym_env else self.gym_env
        total_episodes_return = 0

        for i in range(self.num_episodes):
            self.obs = self.action_obs.set_obs(gym_env=self.gym_env)
            done = False
            episode_return = 0

            while not done:
                self.gym_env.render() if self.is_render_env else None
                action = self.dqn_model.act(observation=self.obs,
                                            exploit=True)
                next_obs, reward, done, info = \
                    self.action_obs.time_step(gym_env=self.gym_env,
                                              action=action)
                self.obs = self.action_obs.update_obs(next_obs=next_obs,
                                                      obs=self.obs)
                episode_return += reward
            total_episodes_return += episode_return

            if self.is_verbose_returns:
                print(f'Episode { i + 1 } | Score { episode_return }')
        return total_episodes_return / self.num_episodes


class EvaluateClassicControlPolicy(EvaluatePolicy):

    def __init__(self,
                 dqn_model,
                 gym_env,
                 num_episodes,
                 device_type,
                 is_render_env=False,
                 is_verbose_returns=False):
        """
        Initiate evaluate agent subclass instance
        for Classic Control environments
        """
        super().__init__(dqn_model,
                         gym_env,
                         num_episodes,
                         device_type,
                         is_render_env,
                         is_verbose_returns)

        # initialize action-observation interaction with the environment
        self.action_obs = ClassicControlActionObservation(device=self.device)


class EvaluateAtariPolicy(EvaluatePolicy):

    def __init__(self,
                 dqn_model,
                 gym_env,
                 num_episodes,
                 device_type,
                 env_name,
                 is_render_env=False,
                 is_verbose_returns=False):
        """
        Initiate evaluate agent subclass instance for Atari game environments
        """
        super().__init__(dqn_model,
                         gym_env,
                         num_episodes,
                         device_type,
                         is_render_env,
                         is_verbose_returns)
        env_config = get_hyperparameters(env_name)
        screen_size = env_config['screen_size']
        is_grayscale_obs = env_config['is_grayscale_obs']
        frame_skip = env_config['frame_skip']
        noop_max = env_config['noop_max']
        rescale = env_config['rescale']
        obs_size = env_config['obs_stack_size']
        actions_map = env_config['actions_map']

        # wrap environment with AtariPreprocessing
        self.gym_env = AtariPreprocessing(env=self.gym_env,
                                          screen_size=screen_size,
                                          grayscale_obs=is_grayscale_obs,
                                          frame_skip=frame_skip,
                                          noop_max=noop_max)

        # initialize action-observation interaction with the environment
        self.action_obs = AtariActionObservation(device=self.device,
                                                 rescale=rescale,
                                                 obs_size=obs_size,
                                                 actions_map=actions_map)


def evaluate_trained_model():
    args = ParseArguments().get_args()
    device = 'cpu'
    model = load(f=f'models/{ args.env }.pt',
                 map_location=device)

    model.eval()
    if is_classic_control(args.env):
        env = make(id=args.env)
        eval_policy = \
            EvaluateClassicControlPolicy(dqn_model=model,
                                         gym_env=env,
                                         num_episodes=args.num_eval_episodes,
                                         device_type=device,
                                         is_render_env=True,
                                         is_verbose_returns=True)
    else:
        env = make(id=args.env,
                   render_mode='human')
        eval_policy = EvaluateAtariPolicy(dqn_model=model,
                                          gym_env=env,
                                          num_episodes=args.num_eval_episodes,
                                          device_type=device,
                                          env_name=args.env,
                                          is_verbose_returns=True)
    mean_episodes_return = eval_policy.run()
    print(f'Total Episodes { args.num_eval_episodes } | '
          f'Mean Score { mean_episodes_return }')
    env.close()


if __name__ == '__main__':
    evaluate_trained_model()
