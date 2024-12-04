# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the CyberBattle Gym environment"""
from typing import List

from cyberbattle._env.option_wrapper import ContextWrapper, random_options
from cyberbattle._env.cyberbattle_env import AttackerGoal, CyberBattleEnv
import pytest
import gymnasium as gym
import numpy as np
from typing import cast
from cyberbattle.simulation import model as m
from cyberbattle.samples.toyctf.yangyang_toy_deception import nodes,global_vulnerability_library,ENV_IDENTIFIERS
from cyberbattle.samples.toyctf import yangyang_toy_deception

from ..samples.toyctf import yangyang_toy_deception
def test_total_realnodes():
    a=m.Environment(network=m.create_network(nodes), vulnerability_library=global_vulnerability_library, identifiers=ENV_IDENTIFIERS)
    b=a.get_nodes_with_real()
    print(b)



class TestCyberBattleEnv(CyberBattleEnv):
    def get_owned_real_nodes_indices(self) -> List[int]:
        # 通过继承的方式调用父类中的私有方法
        return self._CyberBattleEnv__get__owned_real_nodes_indices()

def test_getrealnodes() -> None:
    """Run a few iterations of the gym environment"""

    # 使用你的自定义环境类来创建环境实例，而不是 gym.make
    env = TestCyberBattleEnv(initial_environment=yangyang_toy_deception.new_environment())  # 直接创建 TestCyberBattleEnv 实例

    # 通过公有方法访问私有属性
    a = env.get_owned_real_nodes_indices()
    print("daying a", a)
    print("yy")
    print("adechangdu", len(a))

    env.close()




def test_few_gym_iterations() -> None:
    """Run a few iterations of the gym environment"""
    env = cast(CyberBattleEnv, gym.make("CyberBattleDeception-v0"))

    for _ in range(2):
        env.reset()
        action_mask = env.compute_action_mask()
        assert action_mask
        for t in range(12):
            # env.render()

            # sample a valid action
            action = env.sample_valid_action()

            observation, reward, done, truncated, info = env.step(action)
            if truncated:
                print("Episode truncated after {} timesteps".format(t + 1))
                break

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()
    pass


def test_step_after_done() -> None:
    actions = [
        {"local_vulnerability": np.array([0, 1])},  # done=False r=9.0
        {"remote_vulnerability": np.array([0, 1, 0])},  # done=False r=4.0
        {"connect": np.array([0, 1, 2, 0])},  # done=False r=100.0
        {"local_vulnerability": np.array([1, 3])},  # done=False r=9.0
        {"connect": np.array([0, 2, 3, 1])},  # done=False r=100.0
        {"remote_vulnerability": np.array([1, 2, 1])},  # done=False r=6.0
        {"remote_vulnerability": np.array([1, 2, 0])},  # done=False r=6.0
        {"remote_vulnerability": np.array([2, 1, 1])},  # done=False r=2.0
        {"local_vulnerability": np.array([1, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([1, 1])},  # done=False r=0.0
        {"local_vulnerability": np.array([2, 1])},  # done=False r=6.0
        {"remote_vulnerability": np.array([2, 3, 0])},  # done=False r=4.0
        {"local_vulnerability": np.array([2, 4])},  # done=False r=9.0
        {"connect": np.array([0, 3, 2, 2])},  # done=False r=100.0
        {"local_vulnerability": np.array([3, 3])},  # done=False r=9.0
        {"local_vulnerability": np.array([3, 0])},  # done=False r=6.0
        {"remote_vulnerability": np.array([0, 4, 1])},  # done=False r=8.0
        {"local_vulnerability": np.array([3, 1])},  # done=False r=0.0
        {"connect": np.array([2, 4, 3, 3])},  # done=False r=100.0
        {"remote_vulnerability": np.array([1, 3, 1])},  # done=False r=2.0
        {"remote_vulnerability": np.array([1, 4, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([4, 1])},  # done=False r=6.0
        {"remote_vulnerability": np.array([0, 5, 0])},  # done=False r=4.0
        {"local_vulnerability": np.array([4, 4])},  # done=False r=9.0
        {"connect": np.array([3, 5, 2, 4])},  # done=False r=100.0
        {"remote_vulnerability": np.array([2, 5, 1])},  # done=False r=2.0
        {"local_vulnerability": np.array([5, 3])},  # done=False r=9.0
        {"connect": np.array([2, 6, 3, 5])},  # done=False r=100.0
        {"remote_vulnerability": np.array([4, 6, 1])},  # done=False r=6.0
        {"local_vulnerability": np.array([5, 0])},  # done=False r=6.0
        {"remote_vulnerability": np.array([4, 6, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([5, 1])},  # done=False r=0.0
        {"local_vulnerability": np.array([6, 1])},  # done=False r=6.0
        {"remote_vulnerability": np.array([6, 7, 0])},  # done=False r=4.0
        {"remote_vulnerability": np.array([0, 7, 1])},  # done=False r=2.0
        {"local_vulnerability": np.array([6, 4])},  # done=False r=9.0
        {"connect": np.array([4, 7, 2, 6])},  # done=False r=100.0
        {"local_vulnerability": np.array([7, 3])},  # done=False r=9.0
        {"connect": np.array([0, 8, 3, 7])},  # done=False r=100.0
        {"remote_vulnerability": np.array([0, 8, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([7, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([8, 4])},  # done=False r=9.0
        {"remote_vulnerability": np.array([3, 9, 1])},  # done=False r=2.0
        {"connect": np.array([3, 9, 2, 8])},  # done=False r=100.0
        {"remote_vulnerability": np.array([4, 9, 0])},  # done=False r=2.0
        {"local_vulnerability": np.array([9, 0])},  # done=False r=6.0
        {"remote_vulnerability": np.array([3, 8, 1])},  # done=False r=6.0
        {"remote_vulnerability": np.array([6, 10, 0])},  # done=False r=6.0
        {"local_vulnerability": np.array([9, 1])},  # done=False r=0.0
        {"local_vulnerability": np.array([9, 3])},  # done=False r=9.0
        {"remote_vulnerability": np.array([8, 10, 1])},  # done=False r=8.0
        {"local_vulnerability": np.array([7, 1])},  # done=False r=0.0
        {"connect": np.array([8, 10, 3, 9])},  # done=False r=100.0
        {"local_vulnerability": np.array([10, 4])},  # done=False r=9.0
        {"local_vulnerability": np.array([8, 1])},  # done=False r=6.0
        {"connect": np.array([7, 11, 2, 10])},  # done=True r=5000.0
        # this is one too many (after done)
        {"connect": np.array([10, 5, 2, 4])},
    ]

    env = gym.make(
        "CyberBattleChain-v0",
        size=10,
        attacker_goal=AttackerGoal(own_atleast_percent=1.0),
    )
    env.reset()
    for a in actions[:-1]:
        observation, reward, done, truncated, info = env.step(a)
        print(f"{a}, # done={done} truncated={truncated} r={reward}")

    with pytest.raises(RuntimeError, match=r"new episode must be started with env\.reset\(\)"):
        env.step(actions[-1])


def test_option_wrapper():
    env = gym.make("CyberBattleChain-v0", size=10, attacker_goal=AttackerGoal(reward=4000))
    env = ContextWrapper(cast(CyberBattleEnv, env), options=random_options)

    s, _ = env.reset()
    for t in range(4):
        s, r, done, truncated, info = env.step()
        if r > 0:
            print(r, done, info["action"])
        if done:
            break
