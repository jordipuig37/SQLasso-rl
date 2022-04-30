import pandas as pd
import json
import argparse
import copy

import torch
from utils import DotDic

from agent import PlayerSQLillo
from agent import AgentNet
from agent import init_xavi_uniform
from environment import SQLilloLearningEnv


def create_agent(conf):
    cnet = AgentNet(conf)
    cnet.apply(init_xavi_uniform)
    cnet_target = copy.deepcopy(cnet)
    agent = PlayerSQLillo(conf, model=cnet, target=cnet_target)

    return agent


def read_conf(path):
    f = open(path, "r")
    conf = DotDic(json.load(f))
    f.close()
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return conf


def make_trial(conf, return_test=True):
    environment = SQLilloLearningEnv(conf)
    agent = create_agent(conf)
    environment.train(agent)
    df = pd.DataFrame(environment.stats)
    test_df = pd.DataFrame(environment.test_stats)

    if return_test:
        return df, test_df

    return df, None


def main(args, **kwargs):
    base_conf = read_conf(args.conf_file)
    train_results = make_trial(base_conf, False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf-file", type=str,
        help="The file in which the configuration for our experimets is stored.")
    parser.add_argument("-v", "--verbose", nargs='?', const=True, default=False)
    args = parser.parse_args()

    main(args)
