# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from cyberbattle.samples.toyctf import yangyang_toy_deception

from ..samples.toyctf import yangyang_toy_deception
from . import cyberbattle_env


class CyberBattleYang(cyberbattle_env.CyberBattleEnv):
    """自己创建一个测试网络"""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=yangyang_toy_deception.new_environment(), **kwargs)
