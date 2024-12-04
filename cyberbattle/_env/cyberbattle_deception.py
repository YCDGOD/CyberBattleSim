# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from cyberbattle.samples.toyctf import yangyang_toy_deception

from ..samples.toyctf import yangyang_toy_deception
from . import cyberbattle_env


class CyberBattleDeception(cyberbattle_env.CyberBattleEnv):
    """自己创建一个测试网络"""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=yangyang_toy_deception.new_environment(), **kwargs)
"""CyberBattleEnv需要一个初始化环境——initial_environment: model.Environment,
然后进入yangyang_toy_deception.new_environment()这个函数将网络拓扑传给作为参数，
返回一个Environment类"""
