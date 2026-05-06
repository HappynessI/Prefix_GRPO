import os
import sys
import json
import glob
import random
import numpy as np

import textworld
import textworld.agents
import textworld.gym
from textworld.gym.envs import TextworldBatchGymEnv
import gym

from alfworld.agents.utils.misc import (
    Demangler,
    get_templated_task_desc,
    add_task_to_grammar,
)
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredInfos

from .utils import load_config


class SingleAlfredTWEnv(AlfredTWEnv):
    """
    Interface for Textworld Env
    Contains only one game_file per environment
    """

    def __init__(self, config, train_eval="eval_out_of_distribution"):
        if os.environ.get("ALFWORLD_VERBOSE_ENV_LOGS", "0").lower() in {"1", "true", "yes"}:
            print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval

        self.goal_desc_human_anns_prob = self.config["env"]["goal_desc_human_anns_prob"]
        self.get_game_logic()
        self.random_seed = 42

        self.game_files = []
        self.num_games = 0

    def init_env(self, batch_size):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = self.config["env"]["expert_type"]
        training_method = self.config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]

            expert_plan = True if self.train_eval == "train" else False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")
        else:
            raise NotImplementedError

        return TextworldBatchGymEnv(
            gamefiles=self.game_files,
            request_infos=request_infos,
            batch_size=batch_size,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers,
        )


def get_all_game_files(config, split="eval_out_of_distribution"):
    env = AlfredTWEnv(config, train_eval=split)
    game_files = env.game_files
    del env
    return game_files


if __name__ == "__main__":
    os.environ["ALFWORLD_DATA"] = "/Users/wang/.cache/alfworld"
    config = load_config("configs/base_config.yaml")
    game_files = get_all_game_files(config, "train")
    game_files = [game[len(os.environ["ALFWORLD_DATA"]) + 1 :] for game in game_files]
    with open("legacy/alfworld/client/games/new_file.json", "w") as f:
        f.write(json.dumps(game_files, indent=2))
        f.close()
    print(len(game_files))
    print(game_files[0])
