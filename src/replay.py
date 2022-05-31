from random import sample


class ReplayMemory:

    def __init__(self, memory_size):
        """
        Initiate replay memory class instance
        """
        self.__cap = memory_size
        self.__memory = []
        self.__pos = 0

    def __len__(self):
        return len(self.__memory)

    def push(self, obs, action, next_obs, reward):
        """
        Push transition to replay memory
        """
        self.__memory.append(None) if len(self.__memory) < self.__cap else None
        self.__memory[self.__pos] = (obs, action, next_obs, reward)
        self.__pos = (self.__pos + 1) % self.__cap

    def sample_transitions(self, batch_size):
        """
        Sample batch-size transitions from replay memory
        """
        return tuple(zip(*sample(self.__memory, batch_size)))
