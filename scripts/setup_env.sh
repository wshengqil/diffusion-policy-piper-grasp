#!/usr/bin/env bash
# 环境搭建（WSL2 Ubuntu 实测可用）。按需修改 cuda 版本。
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-diffusion}
WORKDIR=${WORKDIR:-$HOME/robotics}

conda create -n "$CONDA_ENV" python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# PyTorch（按你的 cuda 版本调整；要求 1.0 < torch < 2.3）
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# MuJoCo / 渲染依赖
sudo apt-get update
sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
pip install mujoco==3.1.6 dm-control==1.0.10

# 上游框架
mkdir -p "$WORKDIR" && cd "$WORKDIR"
git clone https://github.com/ARISE-Initiative/robomimic.git
git clone https://github.com/NoneJou072/robopal.git
pip install -e ./robomimic
pip install -r ./robopal/requirements.txt
pip install numpy==1.26.4 d4rl==1.1

# 覆盖本仓库的改动（在仓库根目录执行本脚本时可用）
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cp "$REPO_ROOT"/src/robopal/scripts/*.py               "$WORKDIR"/robopal/robopal/scripts/
cp "$REPO_ROOT"/src/robopal/assets/scenes/grasping.xml "$WORKDIR"/robopal/robopal/assets/scenes/
cp "$REPO_ROOT"/src/robomimic/algo/diffusion_policy.py "$WORKDIR"/robomimic/robomimic/algo/
cp "$REPO_ROOT"/src/robomimic/scripts/run_trained_agent.py "$WORKDIR"/robomimic/robomimic/scripts/
cp "$REPO_ROOT"/src/robomimic/utils/lang_utils.py      "$WORKDIR"/robomimic/robomimic/utils/
cp "$REPO_ROOT"/config/diffusion_policy.json           "$WORKDIR"/robomimic/robomimic/exps/templates/

echo "done. 记得把 grasping.xml 里的 table.xml / table.STL 路径改成本机绝对路径。"
