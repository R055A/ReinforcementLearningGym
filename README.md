# Reinforcement Learning Gym

Train and evaluate reinforcement learning agents in [OpenAI Gym environments](https://www.gymlibrary.ml/) using Deep Q-learning (DQN).

![Pong-v5 graph](img/pong-v5_graph_darkmode.png#gh-dark-mode-only)
![Pong-v5 graph](img/pong-v5_graph_lightmode.png#gh-light-mode-only)

## Environments

The following environments are implemented for training and evaluating:

#### Classic Control

>* CartPole
>* MountainCar
>* Acrobot

#### Atari

>* Pong
>* Breakout

more can be added, and hyperparameters can be tuned in `config.py`

## Trained Models

The following are trained models included in this repository for evaluation:

#### Classic Control

>* CartPole-v1
>* Acrobot-v1
 
#### Atari

>* Pong-v0
>* Pong-v5

Each trained model can be found in the `/models` directory.

Demonstration videos of trained models can be found in the [OpenAI Gym](https://youtube.com/playlist?list=PLcPfzo2p7brHwyAORic1_jfBMKp0InkPt) YouTube playlist.

# Setup

For training and evaluating the OpenAI Gym environments with a GPU, the following setup has been used:

>* Anaconda 2.11
>* CUDA 11.3
>* Python 3.10
>* PyTorch 1.11
>* OpenAI gym 0.24

# Install

```shell
pip install -r requirements.txt
```

# Run

## CartPole

[![CartPole-v1 video](img/CartPole-v1_thumbnail.png)](https://youtu.be/XoRg8VKQlk4)

### CartPole-v0

#### Train

```shell
python src/train.py --env CartPole-v0 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env CartPole-v0 --num_eval_episodes <int> --is_render --is_record
```

### CartPole-v1

#### Train

```shell
python src/train.py --env CartPole-v1 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env CartPole-v1 --num_eval_episodes <int> --is_render --is_record
```

## Acrobot

[![Acrobot-v1 video](img/Acrobot-v1_thumbnail.png)](https://youtube.com/shorts/aRY78lB6cfQ?feature=share)

### Acrobot-v1

#### Train

```shell
python src/train.py --env Acrobot-v1 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env Acrobot-v1 --num_eval_episodes <int> --is_render --is_record
```

## MountainCar

### MountainCar-v0

#### Train

```shell
python src/train.py --env MountainCar-v0 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env MountainCar-v0 --num_eval_episodes <int> --is_render --is_record
```

## Pong

[![Pong-v5 video](img/Pong-v5_thumbnail.png)](https://youtu.be/fve0Uf91ruo)

### Pong-v0

#### Train

```shell
python src/train.py --env Pong-v0 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env Pong-v0 --num_eval_episodes <int> --is_render --is_record
```

### Pong-v4

#### Train

```shell
python src/train.py --env Pong-v4 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env Pong-v4 --num_eval_episodes <int> --is_render --is_record
```

### Pong-v5

#### Train

```shell
python src/train.py --env ALE/Pong-v5 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env ALE/Pong-v5 --num_eval_episodes <int> --is_render --is_record
```

## Breakout

### Breakout-v0

#### Train

```shell
python src/train.py --env Breakout-v0 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env Breakout-v0 --num_eval_episodes <int> --is_render --is_record
```

### Breakout-v4

#### Train

```shell
python src/train.py --env Breakout-v4 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env Breakout-v4 --num_eval_episodes <int> --is_render --is_record
```

### Breakout-v5

#### Train

```shell
python src/train.py --env ALE/Breakout-v5 --eval_freq <int> --num_eval_episodes <int>
```

#### Evaluate

```shell
python src/evaluate.py --env ALE/Breakout-v5 --num_eval_episodes <int> --is_render --is_record
```
