#!/usr/bin/env python3
"""
OMNI 29-DOF 机器人集成到 TienKung-Lab 的自动化脚本。

功能:
1. 读取 TienKung-Lab 现有的 tienkung 环境配置作为模板
2. 创建 OMNI 专属的环境配置 (omni_walk / omni_run)
3. 注册到 TienKung-Lab 的任务系统中
4. 处理 OMNI 模型的 Python 路径

用法:
    cd /root/autodl-tmp/TienKung-Lab
    python integrate_omni.py
"""

import os
import sys
import re
import shutil
from pathlib import Path

# ============================================================
# 配置
# ============================================================
TKL_ROOT = Path(__file__).parent.resolve()
LEGGED_LAB = TKL_ROOT / "legged_lab"
ENVS_DIR = LEGGED_LAB / "envs"
ASSETS_DIR = LEGGED_LAB / "assets"
OMNI_ASSET_DIR = ASSETS_DIR / "omni_29dof"

# OMNI 机器人的 29 个关节名 (按 URDF 顺序)
OMNI_JOINTS = [
    # 左腿 (6)
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    # 右腿 (6)
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    # 腰 (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂 (7)
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint",
    "wrist_pitch_l_joint", "wrist_roll_l_joint",
    # 右臂 (7)
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint",
    "wrist_pitch_r_joint", "wrist_roll_r_joint",
]

# OMNI 默认站立关节角
OMNI_DEFAULT_JOINT_POS = {
    "hip_pitch_l_joint": -0.26178,
    "hip_pitch_r_joint": -0.26178,
    "knee_pitch_l_joint": 0.52356,
    "knee_pitch_r_joint": 0.52356,
    "ankle_pitch_l_joint": -0.26178,
    "ankle_pitch_r_joint": -0.26178,
    "elbow_pitch_l_joint": -0.7,
    "elbow_pitch_r_joint": -0.7,
    "shoulder_pitch_l_joint": 0.3,
    "shoulder_pitch_r_joint": 0.3,
}

# 脚底 link 名称 (用于接触奖励)
OMNI_FOOT_LINKS = ["ankle_roll_l_link", "ankle_roll_r_link"]

# 身体主要 link (用于跟踪奖励)
OMNI_BODY_LINKS = [
    "base_link",
    "hip_pitch_l_link", "hip_pitch_r_link",
    "knee_pitch_l_link", "knee_pitch_r_link",
    "ankle_pitch_l_link", "ankle_pitch_r_link",
]


def check_prerequisites():
    """检查前置条件"""
    print("=" * 50)
    print("  OMNI 29-DOF 集成到 TienKung-Lab")
    print("=" * 50)
    print()

    # 检查 TienKung-Lab 结构
    if not LEGGED_LAB.exists():
        print(f"[错误] 找不到 {LEGGED_LAB}")
        print("  请确认在 TienKung-Lab 根目录下运行本脚本")
        sys.exit(1)

    # 检查 tienkung 环境 (作为模板)
    tienkung_env = ENVS_DIR / "tienkung"
    if not tienkung_env.exists():
        print(f"[错误] 找不到模板环境: {tienkung_env}")
        sys.exit(1)

    # 检查 OMNI 模型
    if not OMNI_ASSET_DIR.exists():
        print(f"[错误] OMNI 模型未部署: {OMNI_ASSET_DIR}")
        print("  请先运行 setup_tienkung.sh 或手动复制模型文件")
        sys.exit(1)

    urdf_path = OMNI_ASSET_DIR / "omni_29dof_nohead_noshoe" / "urdf" / \
        "omni_29dof_nohead_noshoe_merged_modify_feet.urdf"
    if not urdf_path.exists():
        print(f"[错误] URDF 文件缺失: {urdf_path}")
        sys.exit(1)

    print("[✓] 前置检查通过")
    print(f"  TienKung-Lab: {TKL_ROOT}")
    print(f"  OMNI 模型: {OMNI_ASSET_DIR}")
    print()


def create_omni_env_directory():
    """创建 OMNI 环境目录结构"""
    omni_env_dir = ENVS_DIR / "omni"
    omni_env_dir.mkdir(exist_ok=True)

    # 创建 datasets 目录
    (omni_env_dir / "datasets" / "motion_visualization").mkdir(parents=True, exist_ok=True)
    (omni_env_dir / "datasets" / "motion_amp_expert").mkdir(parents=True, exist_ok=True)

    print(f"[✓] 创建目录: {omni_env_dir}")
    return omni_env_dir


def copy_motion_data(omni_env_dir: Path):
    """复制 TienKung 的动作数据作为临时占位 (后续需替换为 OMNI 专属数据)"""
    tienkung_datasets = ENVS_DIR / "tienkung" / "datasets"
    omni_datasets = omni_env_dir / "datasets"

    for subdir in ["motion_visualization", "motion_amp_expert"]:
        src_dir = tienkung_datasets / subdir
        dst_dir = omni_datasets / subdir
        if src_dir.exists():
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
            print(f"[✓] 复制动作数据: {subdir}/ ({len(list(src_dir.iterdir()))} 个文件)")
        else:
            print(f"[提示] TienKung {subdir}/ 不存在，跳过")

    print("  注意: 这是 TienKung 的临时数据，正式训练需替换为 OMNI 专属重定向数据")


def adapt_tienkung_env(omni_env_dir: Path):
    """基于 tienkung_env.py 创建 omni_env.py"""
    src = ENVS_DIR / "tienkung" / "tienkung_env.py"
    dst = omni_env_dir / "omni_env.py"

    if not src.exists():
        print(f"[警告] 模板文件不存在: {src}")
        print("  将创建基础版本...")
        create_basic_omni_env(dst)
        return

    content = src.read_text()

    # 替换机器人相关引用
    content = content.replace("tienkung2_lite", "omni_29dof")
    content = content.replace("tienkung", "omni")
    content = content.replace("TienKung", "OMNI")
    content = content.replace("Tienkung", "Omni")
    content = content.replace("TIENKUNG", "OMNI")

    # 类名替换
    content = content.replace("TienkungEnv", "OmniEnv")
    content = content.replace("TienKungEnv", "OmniEnv")

    # 修复: 删除引用不存在的 WithSensor 配置文件的行
    lines = content.split('\n')
    filtered_lines = []
    skip_continuation = False
    for line in lines:
        if 'WithSensor' in line or 'with_sensor' in line:
            # 如果这行有未关闭的括号，后续行也要跳过
            if '(' in line and ')' not in line:
                skip_continuation = True
            continue
        if skip_continuation:
            if ')' in line:
                skip_continuation = False
            continue
        filtered_lines.append(line)
    content = '\n'.join(filtered_lines)

    # 修复: 替换 pelvis 为 base_link (OMNI 没有 pelvis link)
    content = content.replace("pelvis", "base_link")

    dst.write_text(content)
    print(f"[✓] 创建: {dst.name} (基于 tienkung_env.py 适配)")


def adapt_walk_cfg(omni_env_dir: Path):
    """基于 walk_cfg.py 创建 OMNI 走路配置"""
    src = ENVS_DIR / "tienkung" / "walk_cfg.py"
    dst = omni_env_dir / "walk_cfg.py"

    if not src.exists():
        print(f"[警告] 模板不存在: {src}，创建基础配置")
        create_basic_walk_cfg(dst)
        return

    content = src.read_text()

    # 替换机器人引用
    content = content.replace("tienkung2_lite", "omni_29dof")
    content = content.replace("tienkung", "omni")
    content = content.replace("TienKung", "OMNI")
    content = content.replace("Tienkung", "Omni")

    # 类名
    content = content.replace("TienkungWalkCfg", "OmniWalkCfg")
    content = content.replace("TienKungWalkCfg", "OmniWalkCfg")

    # 任务注册名
    content = content.replace('"walk"', '"omni_walk"')
    content = content.replace("'walk'", "'omni_walk'")

    # 修复: 替换 pelvis 为 base_link (OMNI 没有 pelvis link)
    content = content.replace("pelvis", "base_link")

    dst.write_text(content)
    print(f"[✓] 创建: {dst.name}")


def adapt_run_cfg(omni_env_dir: Path):
    """基于 run_cfg.py 创建 OMNI 跑步配置"""
    src = ENVS_DIR / "tienkung" / "run_cfg.py"
    dst = omni_env_dir / "run_cfg.py"

    if not src.exists():
        print(f"[提示] run_cfg.py 不存在，跳过跑步配置")
        return

    content = src.read_text()
    content = content.replace("tienkung2_lite", "omni_29dof")
    content = content.replace("tienkung", "omni")
    content = content.replace("TienKung", "OMNI")
    content = content.replace("Tienkung", "Omni")
    content = content.replace("TienkungRunCfg", "OmniRunCfg")
    content = content.replace("TienKungRunCfg", "OmniRunCfg")
    content = content.replace('"run"', '"omni_run"')
    content = content.replace("'run'", "'omni_run'")

    # 修复: 替换 pelvis 为 base_link
    content = content.replace("pelvis", "base_link")

    dst.write_text(content)
    print(f"[✓] 创建: {dst.name}")


def create_basic_omni_env(dst: Path):
    """当模板不可用时，创建基础 OMNI 环境文件"""
    content = '''"""
OMNI 29-DOF 人形机器人环境配置 (基础版)
基于 TienKung-Lab 框架，使用 OMNI 机器人模型
"""
# 注意: 这是自动生成的基础版本
# 如果 TienKung-Lab 的 tienkung_env.py 存在，请优先使用适配版本
# 本文件可能需要根据实际 TienKung-Lab 版本手动调整

import sys
from pathlib import Path

# 将 OMNI 模型目录加入 Python 路径
OMNI_DIR = Path(__file__).parent.parent.parent / "assets" / "omni_29dof"
if str(OMNI_DIR) not in sys.path:
    sys.path.insert(0, str(OMNI_DIR))
'''
    dst.write_text(content)
    print(f"[✓] 创建基础版: {dst.name}")


def create_basic_walk_cfg(dst: Path):
    """基础走路配置 (当模板不可用时)"""
    content = '''"""OMNI 走路任务配置 (基础版，需手动完善)"""
# TODO: 参考 tienkung/walk_cfg.py 补充完整配置
'''
    dst.write_text(content)


def create_init_file(omni_env_dir: Path):
    """创建 OMNI 环境的 __init__.py，注册任务"""
    init_content = '''"""
OMNI 29-DOF 环境注册
"""
import sys
from pathlib import Path

# 确保 OMNI 模型可被 import
OMNI_ASSET_DIR = str(Path(__file__).parent.parent / "assets" / "omni_29dof")
if OMNI_ASSET_DIR not in sys.path:
    sys.path.insert(0, OMNI_ASSET_DIR)

from legged_lab.envs.omni.omni_env import OMNIEnv
from legged_lab.envs.omni.walk_cfg import OMNIWalkFlatEnvCfg, OMNIWalkAgentCfg
from legged_lab.utils.task_registry import task_registry

task_registry.register("omni_walk", OMNIEnv, OMNIWalkFlatEnvCfg(), OMNIWalkAgentCfg())

try:
    from legged_lab.envs.omni.run_cfg import OMNIRunFlatEnvCfg, OMNIRunAgentCfg
    task_registry.register("omni_run", OMNIEnv, OMNIRunFlatEnvCfg(), OMNIRunAgentCfg())
except (ImportError, Exception):
    pass
'''
    (omni_env_dir / "__init__.py").write_text(init_content)
    print(f"[✓] 创建: __init__.py (任务注册)")


def register_in_envs_init():
    """在 envs/__init__.py 中注册 OMNI 环境"""
    envs_init = ENVS_DIR / "__init__.py"

    if not envs_init.exists():
        # 创建新的
        envs_init.write_text('from .omni import *\n')
        print(f"[✓] 创建 envs/__init__.py")
        return

    content = envs_init.read_text()

    # 检查是否已注册
    if "omni" in content.lower():
        print(f"[✓] envs/__init__.py 已包含 OMNI 注册")
        return

    # 追加 OMNI 导入
    content += "\n# OMNI 29-DOF 环境\nfrom .omni import *\n"
    envs_init.write_text(content)
    print(f"[✓] 已在 envs/__init__.py 注册 OMNI")


def fix_assets_import():
    """
    修复 OMNI 机器人配置文件中的 import 路径问题。
    原始文件用 'from assets import ASSET_DIR'，在新目录结构下不可用。
    同时创建 assets/omni_29dof/__init__.py 导出 OMNI_CFG。
    """
    robot_cfg_file = OMNI_ASSET_DIR / "robots" / "omni_29dof_nohead_noshoe_dcmotor_identified.py"

    if robot_cfg_file.exists():
        content = robot_cfg_file.read_text()
        # 修复: 替换 'from assets import ASSET_DIR' 为自动计算路径
        if "from assets import ASSET_DIR" in content:
            content = content.replace(
                "from assets import ASSET_DIR",
                "import os\nASSET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
            )
            robot_cfg_file.write_text(content)
            print(f"[✓] 修复 robot cfg 中的 'from assets import' 路径")
        else:
            print(f"[✓] robot cfg import 路径无需修复")
    else:
        print(f"[警告] 未找到: {robot_cfg_file}")

    # 创建 assets/omni_29dof/__init__.py 导出 OMNI_CFG
    assets_init = OMNI_ASSET_DIR / "__init__.py"
    init_content = '''"""
OMNI 29-DOF 机器人模型包
导出 OMNI_CFG 供 TienKung-Lab 环境配置引用
"""
import sys
from pathlib import Path

# 确保本目录下的 robots/ actuators/ 可被 import
sys.path.insert(0, str(Path(__file__).parent))

from robots.omni_29dof_nohead_noshoe_dcmotor_identified import (
    OMNI_DCMOTOR_IDENTIFIED_CFG as OMNI_CFG,
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE as OMNI_ACTION_SCALE,
)
'''
    assets_init.write_text(init_content)
    print(f"[✓] 创建: assets/omni_29dof/__init__.py (导出 OMNI_CFG)")


def create_robot_cfg_adapter(omni_env_dir: Path):
    """
    创建 OMNI 机器人配置适配器。
    将 omni_29dof 的 ArticulationCfg 转换为 TienKung-Lab 可用的格式。
    """
    adapter_content = '''"""
OMNI 29-DOF 机器人配置适配器
将 OMNI 模型的 ArticulationCfg 接入 TienKung-Lab 框架
"""
import sys
from pathlib import Path

# 确保 OMNI 模型包可被导入
OMNI_DIR = str(Path(__file__).parent.parent / "assets" / "omni_29dof")
if OMNI_DIR not in sys.path:
    sys.path.insert(0, OMNI_DIR)

from robots.omni_29dof_nohead_noshoe_dcmotor_identified import (
    OMNI_DCMOTOR_IDENTIFIED_CFG,
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
)

# ============================================================
# OMNI 机器人关键参数 (供环境配置引用)
# ============================================================

NUM_DOFS = 29
"""OMNI 总自由度数"""

JOINT_NAMES = [
    # 左腿 (6)
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    # 右腿 (6)
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    # 腰 (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂 (7)
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint",
    "wrist_pitch_l_joint", "wrist_roll_l_joint",
    # 右臂 (7)
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint",
    "wrist_pitch_r_joint", "wrist_roll_r_joint",
]
"""29 个关节名称列表"""

DEFAULT_JOINT_POS = {
    "hip_pitch_l_joint": -0.26178,
    "hip_pitch_r_joint": -0.26178,
    "knee_pitch_l_joint": 0.52356,
    "knee_pitch_r_joint": 0.52356,
    "ankle_pitch_l_joint": -0.26178,
    "ankle_pitch_r_joint": -0.26178,
    "elbow_pitch_l_joint": -0.7,
    "elbow_pitch_r_joint": -0.7,
    "shoulder_pitch_l_joint": 0.3,
    "shoulder_pitch_r_joint": 0.3,
}
"""默认站立姿态 (其余关节为 0)"""

ACTION_SCALE = OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE
"""动作缩放 (全部关节 0.25)"""

FOOT_LINK_NAMES = ["ankle_roll_l_link", "ankle_roll_r_link"]
"""脚底 link (用于接触奖励/终止条件)"""

BASE_LINK_NAME = "base_link"
"""基座 link (用于高度/姿态奖励)"""

ROBOT_CFG = OMNI_DCMOTOR_IDENTIFIED_CFG
"""完整的 ArticulationCfg，可直接用于场景生成"""

# 初始高度 (base_link 离地)
INIT_HEIGHT = 0.8
'''
    dst = omni_env_dir / "omni_robot_cfg.py"
    dst.write_text(adapter_content)
    print(f"[✓] 创建: omni_robot_cfg.py (机器人参数适配)")


def print_post_integration_guide():
    """打印集成后的操作指南"""
    print()
    print("=" * 50)
    print("  集成完成！后续操作指南")
    print("=" * 50)
    print("""
━━━ 第一步: 验证集成是否成功 ━━━

  cd /root/autodl-tmp/TienKung-Lab
  python -c "from legged_lab.envs.omni.omni_robot_cfg import ROBOT_CFG; print('OMNI 配置加载成功')"

━━━ 第二步: 检查并适配环境配置 ━━━

  自动适配可能不完美 (TienKung-Lab 版本差异)。
  如果训练报错，需要手动检查以下文件:

    legged_lab/envs/omni/omni_env.py    ← 环境主逻辑
    legged_lab/envs/omni/walk_cfg.py    ← 走路任务参数

  重点检查:
  1. 关节数量是否匹配 (OMNI = 29 DOF)
  2. 脚底 link 名称是否正确 (ankle_roll_l/r_link)
  3. 奖励函数中引用的 body 名称是否存在于 OMNI URDF
  4. 观测空间维度是否正确

━━━ 第三步: 尝试训练 ━━━

  # 先用少量环境验证
  python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=64

  # 如果报错，看错误信息，通常是:
  # - 关节名不匹配 → 修改 cfg 中的 joint_names
  # - link 名不存在 → 修改奖励/终止条件中的 body_names
  # - API 不兼容 → Isaac Lab 2.1→2.3 的接口变化

━━━ 第四步: 正式训练 ━━━

  python legged_lab/scripts/train.py --task=omni_walk --headless \\
      --logger=tensorboard --num_envs=4096

━━━ 关于动作数据 (AMP) ━━━

  TienKung-Lab 使用 AMP (对抗性动作先验) 训练自然步态。
  需要动作数据: legged_lab/envs/omni/datasets/motion_amp_expert/

  获取方式:
  1. 用 GMR 将人体动捕 (AMASS/SMPLX) 重定向到 OMNI 骨架
  2. 或暂时禁用 AMP 奖励，只用基础 RL 奖励训练
     (在 walk_cfg.py 中设置 amp_weight = 0)

━━━ 常见问题 ━━━

  Q: 报 "task omni_walk not found"
  A: 检查 legged_lab/envs/__init__.py 是否包含 "from .omni import *"

  Q: 报关节数量不匹配
  A: TienKung 和 OMNI 的 DOF 数可能不同，需修改 cfg 中的 num_actions

  Q: 报 API 不兼容 (如某个函数参数变了)
  A: Isaac Lab 2.1→2.3 有少量 API 变化，按报错信息调整即可
""")


def main():
    check_prerequisites()

    # 1. 创建目录结构
    omni_env_dir = create_omni_env_directory()

    # 2. 复制动作数据 (TienKung 临时占位)
    copy_motion_data(omni_env_dir)

    # 3. 修复 assets import 路径 + 创建 __init__.py
    fix_assets_import()

    # 4. 创建机器人配置适配器
    create_robot_cfg_adapter(omni_env_dir)

    # 5. 适配环境文件 (已内置 WithSensor 删除 + pelvis 替换)
    adapt_tienkung_env(omni_env_dir)
    adapt_walk_cfg(omni_env_dir)
    adapt_run_cfg(omni_env_dir)

    # 6. 创建 __init__.py (包含 task_registry.register)
    create_init_file(omni_env_dir)

    # 7. 注册到 envs
    register_in_envs_init()

    # 8. 打印后续指南
    print_post_integration_guide()


if __name__ == "__main__":
    main()
