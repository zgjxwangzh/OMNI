"""action_symmetry_penalty 单测(恢复机制, 2026-08-11)。

对称罚只读 env.action_manager.action (N, 29), 不依赖 sim/其他数据:
  - pitch 类成对关节 (0,1)(9,10)(13,14)(11,12)(21,22)(25,26): L == R
  - roll/yaw 类镜像对 (3,4)(6,7)(17,18)(15,16)(19,20)(23,24)(27,28): L == -R
返回归一化不对称均值 → 1 - exp(-asym/std), 完全对称时 0。
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mdp", "jump_rewards.py")
)
spec = importlib.util.spec_from_file_location("jump_rewards", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
# 需要 mock 掉 isaaclab 依赖(纯函数测试不加载 sim)
sys.modules["isaaclab"] = SimpleNamespace()
sys.modules["isaaclab.assets"] = SimpleNamespace(Articulation=object)
sys.modules["isaaclab.managers"] = SimpleNamespace(
    SceneEntityCfg=lambda *a, **k: SimpleNamespace(_args=a, _kwargs=k)
)
sys.modules["isaaclab.sensors"] = SimpleNamespace(ContactSensor=object)
sys.modules["isaaclab.envs"] = SimpleNamespace()
sys.modules["whole_body_tracking"] = SimpleNamespace()
sys.modules["whole_body_tracking.tasks"] = SimpleNamespace()
sys.modules["whole_body_tracking.tasks.tracking"] = SimpleNamespace()
sys.modules["whole_body_tracking.tasks.tracking.mdp"] = SimpleNamespace()
sys.modules["whole_body_tracking.tasks.tracking.mdp.commands"] = SimpleNamespace(
    MotionCommand=object
)
spec.loader.exec_module(mod)

PITCH_PAIRS = [(0, 1), (9, 10), (13, 14), (11, 12), (21, 22), (25, 26)]
MIRROR_PAIRS = [(3, 4), (6, 7), (17, 18), (15, 16), (19, 20), (23, 24), (27, 28)]


def make_env(actions):
    N = len(actions)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.tensor(actions, dtype=torch.float32))
    )
    return env


def asym_value(actions):
    """手工按函数定义算不对称均值(对照用)。"""
    a = torch.tensor(actions, dtype=torch.float32)
    total = 0.0
    for i, j in PITCH_PAIRS:
        total += torch.abs(a[:, i] - a[:, j]).mean().item()
    for i, j in MIRROR_PAIRS:
        total += torch.abs(a[:, i] + a[:, j]).mean().item()
    return total / (len(PITCH_PAIRS) + len(MIRROR_PAIRS))


def penalty(actions, scale=0.25):
    return mod.action_symmetry_penalty(make_env(actions), scale=scale)


def test_zero_action_perfect_symmetry():
    # 全 0 动作 -> 完全对称 -> 罚 0
    out = penalty([[0.0] * 29])
    assert torch.allclose(out, torch.zeros(1), atol=1e-6)


def test_reference_mirror_action_zero():
    # 一组"镜像对称"动作(参考同款): pitch 对相等、mirror 对相反 -> 罚 0
    act = [0.0] * 29
    for i, j in PITCH_PAIRS:
        act[i] = act[j] = 0.4
    for i, j in MIRROR_PAIRS:
        act[i], act[j] = 0.3, -0.3
    out = penalty([act])
    assert torch.allclose(out, torch.zeros(1), atol=1e-6)


def test_one_pitch_pair_asymmetric_penalized():
    # 仅 hip_pitch(0,1) 不对称 -> 罚 > 0, 且幅度与归一化值匹配
    act = [0.0] * 29
    act[0], act[1] = 0.5, -0.5  # |L-R| = 1.0
    out = penalty([act])
    manual = asym_value([act])
    expect = (torch.tensor(manual) / 0.25) ** 2
    assert torch.allclose(out, expect.unsqueeze(0), atol=1e-4)
    # 归一化后 1.0/13≈0.077 -> (0.077/0.25)²≈0.095, 明确>0 即被罚
    assert 0.05 < out[0].item() < 0.2


def test_one_mirror_pair_asymmetric_penalized():
    # 仅 ankle_roll(17,18) 不对称(mirror: 应相反, 给了同号) -> 罚 > 0
    act = [0.0] * 29
    act[17], act[18] = 0.5, 0.5  # 应镜像为 -0.5/+0.5, 同号 = 不对称 1.0
    out = penalty([act])
    manual = asym_value([act])
    expect = (torch.tensor(manual) / 0.25) ** 2
    assert torch.allclose(out, expect.unsqueeze(0), atol=1e-4)
    assert out[0].item() > 0.0


def test_more_asymmetry_more_penalty():
    # 不对称幅度翻倍 -> 罚单调递增(平方, 永远不饱和)
    act_small = [0.0] * 29
    act_large = [0.0] * 29
    act_small[0], act_small[1] = 0.1, -0.1
    act_large[0], act_large[1] = 1.0, -1.0
    p_small = penalty([act_small])[0].item()
    p_large = penalty([act_large])[0].item()
    assert p_large > p_small


def test_batch_mean_across_envs():
    # 多 env: 一个对称一个不对称, 均值在两个结果之间
    act_sym = [0.0] * 29
    act_asym = [0.0] * 29
    act_asym[0], act_asym[1] = 0.5, -0.5
    out = penalty([act_sym, act_asym])
    p_sym = penalty([act_sym])[0]
    p_asym = penalty([act_asym])[0]
    assert torch.allclose(out, torch.stack([p_sym, p_asym]), atol=1e-6)


def test_custom_scale_scales():
    # scale 越大, 同幅度不对称罚越小(平方罚的分母)
    act = [0.0] * 29
    act[0], act[1] = 0.5, -0.5
    p20 = penalty([act], scale=0.20)[0].item()
    p40 = penalty([act], scale=0.40)[0].item()
    assert p40 < p20
