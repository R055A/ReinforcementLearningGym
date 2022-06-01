from config import get_env_names
from argparse import ArgumentParser


class ParseArguments:

    def __init__(self):
        """
        Initiate argument parser superclass instance
        """
        self.parser = ArgumentParser()
        self.parser.add_argument('--env',
                                 choices=get_env_names())
        self.parser.add_argument('--num_eval_episodes',
                                 type=int,
                                 default=10,
                                 help='Number of evaluation episodes',
                                 nargs='?')

    def get_args(self):
        return self.parser.parse_args()


class ParseTrainArguments(ParseArguments):

    def __init__(self):
        """
        Initiate argument parser subclass instance for training agents
        """
        super().__init__()
        self.parser.add_argument('--eval_freq',
                                 type=int,
                                 default=25,
                                 help='Training evaluation frequency',
                                 nargs='?')


class ParseEvaluateArguments(ParseArguments):

    def __init__(self):
        """
        Initiate argument parser subclass instance for training agents
        """
        super().__init__()
        self.parser.add_argument('--is_render',
                                 dest='is_render',
                                 action='store_true',
                                 help='Render the environment')
        self.parser.set_defaults(is_render=False)
        self.parser.add_argument('--is_record',
                                 dest='is_record',
                                 action='store_true',
                                 help='Render the environment')
        self.parser.set_defaults(is_record=False)
