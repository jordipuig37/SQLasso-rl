import numpy as np
import numpy.random as random
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def weight_reset(m):
    if isinstance(m, nn.Embedding) or isinstance(m, nn.RNN) or \
        isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        m.reset_parameters()


def init_xavi_uniform(m):
    if isinstance(m, nn.Embedding) or \
        isinstance(m, nn.Linear) or isinstance(m, nn.RNN):
            torch.nn.init.xavier_uniform_(m.weight)


class AgentNet(nn.Module):
    """This class represents the Network architecture of the agents."""
    def __init__(self, conf):
        super(AgentNet, self).__init__()
        self.conf = conf
        self.conv = nn.Sequential(
            nn.Conv2d(),
            nn.MaxPool2d(),
            nn.Linear(),
            nn.Conv2d()

        )
        self.id_embedder = nn.Embedding(conf.n_players, conf.emb_dim)

        self.rnn = nn.RNN(conf.emb_dim, conf.rnn_size, num_layers=conf.rnn_layers, batch_first=True)

        self.action = nn.Sequential(
            nn.Linear(conf.rnn_size, conf.action_space)
        )



    def forward(self, board, agent_id, agent_pos, hidden=None):
        features = self.conv(board)
        agent_id_emb = self.id_embedder(agent_id)
        midstate = self.rnn(..., hidden)
        output = self.action(midstate)

        return output


class PlayerSQLillo():
    """This is the class definition of the agent that will interact with the
    environment.
    """
    def __init__(self, idx, conf, model=None, target=None):
        self.conf = conf
        self.idx = idx
        self.eps = conf.eps
        self.device = conf.device
        if model is None:
            self.model = AgentNet(conf).to(conf.device)
        else:
            self.model = model.to(conf.device)

        if target is None:
            self.target = AgentNet(conf).to(conf.device)
        else:
            self.target = target.to(conf.device)

        self.optimizer = optim.RMSprop(self.model.parameters(),
            lr=conf.learningrate,
            momentum=conf.momentum)


        self.action_range = range(0, conf.n_actions)


    def reset(self):
        # reset the network
        self.model.apply(weight_reset)
        self.target.apply(weight_reset)


    def get_model_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def actualize_target_network(self):
        self.target.load_state_dict(self.model.state_dict())


    def _random_choice(self, n):
        return torch.randint(0,high=n, size=(self.conf.bs,))


    def _eps_flip(self, eps):
        return random.random() < eps


    def select_action(self, Q, eps=0, train_mode=True):
        """This function selects (following a epsilon greedy policy over Q)
        an action and a communication. It returns the action and comm as well
        as the Qvalue for that pair.
        """
        should_select_random_a = train_mode and self._eps_flip(self.eps)
        if should_select_random_a:
            action = self._random_choice(self.conf.n_actions).to(self.device)
            action_value = torch.take(Q[:,self.action_range], action)
        
        else:
            action_value, action    = torch.max(Q[:,self.action_range], dim=1)

        return action.long(), action_value


    def episode_loss(self, episode):
        """This function returns a loss that can be backpropageted through the
        agents' networks.
        """
        # compute the loss for each element of the batch
        total_loss = torch.zeros(self.conf.bs, device=self.device)
        for agent_idx in range(self.conf.n_players):
            for step in range(self.conf.steps):
                # L(.) = (r + gamma* max(Q_t(s+1, a+1)) - Q(s, a))**2
                r = episode.total_reward[step] 
                qsa = episode.step_records[step][agent_idx].action_value  # the q value for the action selected
                if step == self.conf.steps-1:  # if we are at the last step
                    td_action = r - qsa
                else:
                    q_target = episode.step_records[step+1][agent_idx].Qt[:,self.action_range]
                    td_action = r + self.conf.gamma * q_target.max(dim=1)[0] - qsa

                total_loss = total_loss + (td_action**2)  # accumulate loss

        loss = total_loss.sum()
        loss = loss/float((self.conf.bs * self.conf.n_players))
        return loss


    def learn_from_episode(self, episode_record):
        """This function computes the loss for a batch of epiosodes in 
        episode_record and actualizes the agent model wheights with it.
        """
        self.optimizer.zero_grad()
        loss = self.episode_loss(episode_record)
        loss.backward(retain_graph=not self.conf.model_know_share)  # retain graph because we don't share parameters
        clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()