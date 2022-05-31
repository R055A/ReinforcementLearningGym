CartPole = {
    'memory_size': 65000,
    'num_episodes': 2000,
    'target_update_freq': 100,
    'train_freq': 1,
    'learning_rate': 0.00005,
    'batch_size': 32,
    'gamma': 0.95,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'obs_size': 4,
    'num_actions': 2,
}

Acrobot = {
    'memory_size': 65000,
    'num_episodes': 2000,
    'target_update_freq': 100,
    'train_freq': 1,
    'learning_rate': 0.00005,
    'batch_size': 32,
    'gamma': 0.95,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'obs_size': 6,
    'num_actions': 3,
}

MountainCar = {
    'memory_size': 65000,
    'num_episodes': 2000,
    'target_update_freq': 100,
    'train_freq': 1,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'gamma': 0.95,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'obs_size': 2,
    'num_actions': 3,
}

Pong = {
    'memory_size': 65000,
    'num_episodes': 10000,
    'target_update_freq': 1000,
    'train_freq': 4,
    'learning_rate': 0.00005,
    'batch_size': 32,
    'gamma': 0.99,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'anneal_length': 10**4,
    'num_actions': 2,
    'actions_map': [2, 3],
    'obs_stack_size': 4,
    'screen_size': 84,
    'is_grayscale_obs': True,
    'frame_skip': 1,
    'noop_max': 30,
    'rescale': 255
}

Breakout = {
    'memory_size': 65000,
    'num_episodes': 10000,
    'target_update_freq': 1000,
    'train_freq': 4,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'gamma': 0.99,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'anneal_length': 10**4,
    'num_actions': 3,
    'actions_map': [1, 2, 3],
    'obs_stack_size': 4,
    'screen_size': 84,
    'is_grayscale_obs': True,
    'frame_skip': 1,
    'noop_max': 30,
    'rescale': 255
}

cartpole_env_names = ['CartPole-v0',
                      'CartPole-v1']

mountain_car_env_names = ['MountainCar-v1']

acrobot_env_names = ['Acrobot-v1']

pong_env_names = ['Pong-v0',
                  'Pong-v4',
                  'ALE/Pong-v5']

breakout_env_names = ['Breakout-v0',
                      'Breakout-v4',
                      'ALE/Breakout-v5']


def get_env_names():
    return cartpole_env_names + \
           mountain_car_env_names + \
           acrobot_env_names + \
           pong_env_names + \
           breakout_env_names


def is_classic_control(env_name):
    return env_name in cartpole_env_names \
           or env_name in mountain_car_env_names \
           or env_name in acrobot_env_names


def get_hyperparameters(env_name):
    if env_name in cartpole_env_names:
        return CartPole
    elif env_name in mountain_car_env_names:
        return MountainCar
    elif env_name in acrobot_env_names:
        return Acrobot
    elif env_name in pong_env_names:
        return Pong
    elif env_name in breakout_env_names:
        return Breakout
