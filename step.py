import numpy as np
from collections import Counter

class SQLilloEngine(object):

    def __init__(self, sep=3):
        # Create board
        self.board = np.zeros((40,40))

        # Randomize locations
        self.players = np.array([1,2,3,4])
        np.random.shuffle(self.players)
        self.inv_players = np.empty_like(self.players)
        for i in range(4):
            self.inv_players[self.players[i]-1] = i

        # Define initial separation from corners
        self.corn_sep = sep

        # Declare player positions
        self.players_pos = \
        [{'id': self.players[0], 
          'workers': [
              {'id': 0, 'pos': {'x': self.corn_sep,   'y': self.corn_sep}},
              {'id': 1, 'pos': {'x': self.corn_sep+1, 'y': self.corn_sep}},
              {'id': 2, 'pos': {'x': self.corn_sep+2, 'y': self.corn_sep}},
              {'id': 3, 'pos': {'x': self.corn_sep,   'y': self.corn_sep+1}},
              {'id': 4, 'pos': {'x': self.corn_sep+2, 'y': self.corn_sep+1}},
              {'id': 5, 'pos': {'x': self.corn_sep,   'y': self.corn_sep+2}},
              {'id': 6, 'pos': {'x': self.corn_sep+1, 'y': self.corn_sep+2}},
              {'id': 7, 'pos': {'x': self.corn_sep+2, 'y': self.corn_sep+2}}
            ]
          },
        {'id': self.players[1], 
         'workers': [
              {'id': 0, 'pos': {'x': self.corn_sep,   'y': 40-self.corn_sep-3}},
              {'id': 1, 'pos': {'x': self.corn_sep+1, 'y': 40-self.corn_sep-3}},
              {'id': 2, 'pos': {'x': self.corn_sep+2, 'y': 40-self.corn_sep-3}},
              {'id': 3, 'pos': {'x': self.corn_sep,   'y': 40-self.corn_sep-2}},
              {'id': 4, 'pos': {'x': self.corn_sep+2, 'y': 40-self.corn_sep-2}},
              {'id': 5, 'pos': {'x': self.corn_sep,   'y': 40-self.corn_sep-1}},
              {'id': 6, 'pos': {'x': self.corn_sep+1, 'y': 40-self.corn_sep-1}},
              {'id': 7, 'pos': {'x': self.corn_sep+2, 'y': 40-self.corn_sep-1}}
            ]
          },
        {'id': self.players[2], 
         'workers': [
              {'id': 0, 'pos': {'x': 40-self.corn_sep-3, 'y': self.corn_sep}},
              {'id': 1, 'pos': {'x': 40-self.corn_sep-2, 'y': self.corn_sep}},
              {'id': 2, 'pos': {'x': 40-self.corn_sep-1, 'y': self.corn_sep}},
              {'id': 3, 'pos': {'x': 40-self.corn_sep-3, 'y': self.corn_sep+1}},
              {'id': 4, 'pos': {'x': 40-self.corn_sep-1, 'y': self.corn_sep+1}},
              {'id': 5, 'pos': {'x': 40-self.corn_sep-3, 'y': self.corn_sep+2}},
              {'id': 6, 'pos': {'x': 40-self.corn_sep-2, 'y': self.corn_sep+2}},
              {'id': 7, 'pos': {'x': 40-self.corn_sep-1, 'y': self.corn_sep+2}}
            ]
          },
        {'id': self.players[3], 
         'workers': [
              {'id': 0, 'pos': {'x': 40-self.corn_sep-3, 'y': 40-self.corn_sep-3}},
              {'id': 1, 'pos': {'x': 40-self.corn_sep-2, 'y': 40-self.corn_sep-3}},
              {'id': 2, 'pos': {'x': 40-self.corn_sep-1, 'y': 40-self.corn_sep-3}},
              {'id': 3, 'pos': {'x': 40-self.corn_sep-3, 'y': 40-self.corn_sep-2}},
              {'id': 4, 'pos': {'x': 40-self.corn_sep-1, 'y': 40-self.corn_sep-2}},
              {'id': 5, 'pos': {'x': 40-self.corn_sep-3, 'y': 40-self.corn_sep-1}},
              {'id': 6, 'pos': {'x': 40-self.corn_sep-2, 'y': 40-self.corn_sep-1}},
              {'id': 7, 'pos': {'x': 40-self.corn_sep-1, 'y': 40-self.corn_sep-1}}
            ]
          }
        ]

        # Player 1
        self.board[self.corn_sep:self.corn_sep+3,
                   self.corn_sep:self.corn_sep+3] = self.players[0]
        self.board[self.corn_sep+1,self.corn_sep+1] = 0
        # Player 2
        self.board[self.corn_sep:self.corn_sep+3,
                   40-self.corn_sep-3:40-self.corn_sep] = self.players[1]
        self.board[self.corn_sep+1,40-self.corn_sep-2] = 0
        # Player 3
        self.board[40-self.corn_sep-3:40-self.corn_sep,
                   self.corn_sep:self.corn_sep+3] = self.players[2]
        self.board[40-self.corn_sep-2,self.corn_sep+1] = 0
        # Player 4        
        self.board[40-self.corn_sep-3:40-self.corn_sep,
                   40-self.corn_sep-3:40-self.corn_sep] = self.players[3]
        self.board[40-self.corn_sep-2,40-self.corn_sep-2] = 0

        for _player in self.players_pos:
            for _worker in _player['workers']:
                assert(self.board[_worker['pos']['x'], _worker['pos']['y']] == _player['id'])

    def check_actions(self):
        c = Counter([a[0] for a in self.priori_acts])
        self.post_acts = []
        for act in self.priori_acts:
            if c[act[0]] == 1:
                self.post_acts.append(act)

    def move(self, x, y, act):
        _x, _y = x, y
        if act == 0: # Up
            x += -1
        elif act == 2: # down
            x += 1
        elif act == 1: # right
            y += 1
        elif act == 3: # left
            y += -1
        if x < 0 or y < 0 or x > 39 or y > 39:
            x, y = _x, _y
        return x, y

    def set_action(self, worker, act, player):
        pos = self.players_pos[self.inv_players[player-1]]['workers'][worker]['pos']
        x, y = pos['x'], pos['y']
        x, y = self.move(x, y, act)
        self.priori_acts.append(((x,y), player, worker))

    def execute_actions(self):
        for act in self.post_acts:
            self.board[act[0][0], act[0][1]] = act[1]
            self.players_pos[self.inv_players[act[1]-1]]['workers'][act[2]]['pos'] = \
                {'x': act[0][0], 'y': act[0][1]}

    def move_players(self, actions):
        # Execute board dynamics
        self.priori_acts = []
        for _player in actions:
            for _worker in _player['workers']:
                self.set_action(act=_worker['action'], worker=_worker['id'], 
                                player=_player['id'])
        self.check_actions()
        self.execute_actions()
        return self.get_board()

    def get_board(self):
        return self.board

    def get_players_pos(self):
        return self.players_pos

    def get_player_pos(self, player_idx):
        return list(map(lambda x: x["pos"], self.players_pos[player_idx]["workers"]))


def format_actions_single_player(player_id, action_list):
    return dict({"id":player_id , 
        "workers": [{"id":i, "act":a} for i, a in enumerate(action_list)]})


def format_actions(actions):
    return list(map(lambda x: format_actions_single_player(*x), enumerate(actions)))
