from collections import defaultdict

import numpy as np
import torch

from connected_components import get_connected_components
from step import SQLilloEngine as Board
from step import format_actions
from utils import DotDic


class EpisodeStats():
    """This class represents the saved information for a batch of episodes.
    It is used to manage episode data.
    """

    def __init__(self, conf):
        self.reward = list()
        self.step_records = defaultdict(lambda: defaultdict(lambda: DotDic({})))

    def record_worker_step(self, t, worker, step_dic):
        self.step_records[t][worker] = DotDic(step_dic)

    def record_reward(self, step_reward):
        self.reward.append(step_reward)

    def get_data(self):
        """This function returns all the information of the episode in a
        dictionary format. It returns the information that will be saved,
        thus, saving all the episode data is not needed.
        """
        final_reward = max(self.reward[-11:])  # ugly sorry
        dictionary = {
            "reward": final_reward,
            # "loss": self.episode_loss,
        }
        return dictionary


def random_player(conf):
    """This function acts as a player that behaves randomly and independently.
    It returns the actions taken for each worker in a list.
    """
    decision = np.random.randint(conf.n_actions, size=conf.n_workers)
    return list(decision)


class SQLilloLearningEnv():
    """This class represents the environment in which the experiments will take
    place. It records the train and test stats.
    """

    def __init__(self, conf, seed=1234):
        np.random.seed(seed)
        self.player_id = 1
        self.conf = conf
        self.device = conf.device
        self.stats = []
        self.test_stats = []

    def train(self, agent):
        """This function trains the given agents running the number of episodes
        defined in self.conf. Also it saves the information for each episode;
        the variables that are saved are indicated in the get_data() function
        from the class EpisodeStats.
        """
        episode_stats = None
        # for each epoch
        for n_episode in range(self.conf.n_episodes):
            print("DEBUG n_episode: ", n_episode)
            episode_stats = self.run_episode(agent)
            ep_loss = agent.learn_from_episode(episode_stats, n_episode)

            episode_stats.episode_loss = ep_loss

            self.stats.append(episode_stats.get_data())
            print("DEBUG get_data: ", episode_stats.get_data())

            if (n_episode + 1) % self.conf.test_freq == 0:
                # if we perform a test episode
                test_episode = self.run_episode(agent, train_mode=False)
                self.test_stats.append(test_episode.get_data())
                if (n_episode + 1) % self.conf.show_results == 0:
                    print(
                        f"Mean Test Reward of {test_episode.final_reward.sum() / self.conf.bs:.3f} at episode {n_episode + 1}")

    def reset(self, seed=1234):
        """This function resets the stats and the seed."""
        self.stats = []
        self.test_stats = []
        np.random.seed(seed)

    def get_reward(self, board, tick):
        """This function returns the reward of corresponding to the action
        vector with respect the ground_truth considering this is happening in
        the step indicated.
        """
        if tick % 10 != 0:
            return 0
        else:
            connected_comps = get_connected_components(board)
            our_cc = connected_comps[1]
            other_cc = max(connected_comps[2], connected_comps[3], connected_comps[4])  #  very ugly

            # return self.conf.alpha * our_cc - self.conf.beta * other_cc
            return self.conf.alpha * our_cc

    def run_episode(self, agent, train_mode=True):
        """This function runs a single batch of episodes and records the
        states, communications, outputs of the episode and returns this record.
        """
        episode_stats = EpisodeStats(self.conf)
        game_board = Board()
        for tick in range(self.conf.n_ticks):
            actions = list()
            workers = game_board.get_player_pos(self.player_id)
            for worker_idx, worker_pos_dic in enumerate(workers):
                agent_inputs = {
                    'board': torch.tensor(game_board.get_board()).float().unsqueeze(0).unsqueeze(0),
                    'agent_pos': torch.tensor(list(worker_pos_dic.values())).float().unsqueeze(0),
                }

                Q = agent.model(**agent_inputs)
                Qt = agent.target(**agent_inputs)
                action, action_value = agent.select_action(Q, train_mode)

                # save worker decision
                actions.append(action.item())

                worker_step_dic = {  # TODO: see if we are missing something to record
                    "Qt": Qt,
                    "action_value": action_value,
                }
                episode_stats.record_worker_step(tick, worker_idx, worker_step_dic)

                if train_mode:
                    agent.eps = agent.eps * self.conf.eps_decay

            # submit actions to board
            # board.execute_actions(actions, player1)
            opponent1 = random_player(self.conf)
            opponent2 = random_player(self.conf)
            opponent3 = random_player(self.conf)

            actions_to_submit = format_actions([actions, opponent1, opponent2, opponent3])
            new_board = game_board.move_players(actions_to_submit)

            reward = self.get_reward(new_board, tick)

            episode_stats.record_reward(reward)

        return episode_stats
