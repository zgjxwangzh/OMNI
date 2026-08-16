# AutoDL 日常使用手册（每次开机后看这个）

## 一、登录实例

SSH 登录后（或 AutoDL 网页终端），执行以下初始化：

```bash
# 1. 激活 conda 环境（每次新终端都要）
export CONDA_ENVS_PATH="/root/autodl-tmp/conda_envs"
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# 2. 确保 Vulkan 配置存在（一般装一次就永久生效，保险起见检查一下）
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    mkdir -p /usr/share/glvnd/egl_vendor.d
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
    echo "Vulkan EGL 配置已修复"
fi

# 3. 进入项目目录
cd /root/autodl-tmp/TienKung-Lab
```

> 可以把上面 1-2 步加到 `~/.bashrc` 里一劳永逸：
> ```bash
> echo 'export CONDA_ENVS_PATH="/root/autodl-tmp/conda_envs"' >> ~/.bashrc
> echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
> echo 'conda activate env_isaaclab' >> ~/.bashrc
> source ~/.bashrc
> ```

---

## 二、启动训练

```bash
cd /root/autodl-tmp/TienKung-Lab

# 快速验证（5分钟，确认环境正常）
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=64 --max_iterations=100

# 正式训练走路（几小时~一天）
python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard

# 正式训练跑步
python legged_lab/scripts/train.py --task=omni_run --headless --num_envs=4096 --logger=tensorboard
```

> 训练会在后台持续运行。如果 SSH 断了训练会中断。
> 建议用 `tmux` 或 `nohup` 保持训练不断：
> ```bash
> # 方法1: tmux（推荐，断开SSH后训练继续）
> tmux new -s train
> python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard
> # 按 Ctrl+B 然后按 D 可以脱离 tmux，训练继续跑
> # 重新连接: tmux attach -t train
>
> # 方法2: nohup
> nohup python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard > train.log 2>&1 &
> # 查看进度: tail -f train.log
> ```

---

## 三、启动 TensorBoard（看训练曲线）

**另开一个终端**（或 tmux 新窗口）：

```bash
export CONDA_ENVS_PATH="/root/autodl-tmp/conda_envs"
eval "$(conda shell.bash hook)"
conda activate env_isaaclab
cd /root/autodl-tmp/TienKung-Lab
tensorboard --logdir=logs --host 0.0.0.0 --port 6006
```

---

## 四、访问 TensorBoard 网页

1. 打开 **AutoDL 网页控制台** → 找到你的实例
2. 点击 **「自定义服务」** 按钮
3. 会看到一个外网访问地址，类似：`https://xxxxx.autodl.fun`
4. 浏览器打开这个地址，就能看到训练曲线

> 如果「自定义服务」没有自动识别 6006 端口：
> - 确认 TensorBoard 已经在运行（终端显示 `TensorBoard 2.x.x at http://0.0.0.0:6006/`）
> - 刷新一下自定义服务页面
> - AutoDL 会自动检测已开放的端口并生成外网链接

---

## 五、查看/使用训练结果

```bash
cd /root/autodl-tmp/TienKung-Lab

# 训练日志和 checkpoint 在这里
ls logs/

# 回放训练好的策略（需要知道 run 文件夹名和 checkpoint 文件名）
python legged_lab/scripts/play.py --task=omni_walk --num_envs=1 \
    --load_run=logs/omni_walk/<时间戳文件夹> \
    --checkpoint=model_<步数>.pt

# 导出到 MuJoCo 验证
python legged_lab/scripts/sim2sim.py --task omni_walk \
    --policy logs/omni_walk/<时间戳>/exported/policy.pt --duration 100
```

---

## 六、常用命令速查

| 操作 | 命令 |
|------|------|
| 激活环境 | `export CONDA_ENVS_PATH="/root/autodl-tmp/conda_envs" && eval "$(conda shell.bash hook)" && conda activate env_isaaclab` |
| 进入项目 | `cd /root/autodl-tmp/TienKung-Lab` |
| 训练走路 | `python legged_lab/scripts/train.py --task=omni_walk --headless --num_envs=4096 --logger=tensorboard` |
| 看 GPU 状态 | `nvidia-smi` |
| 看训练进度 | `tail -f train.log`（如果用 nohup） |
| 恢复 tmux | `tmux attach -t train` |
| 停训练 | `Ctrl+C` |
| TensorBoard | `tensorboard --logdir=logs --host 0.0.0.0 --port 6006` |

---

## 七、注意事项

- **关机前**：训练会中断，checkpoint 已保存的不受影响（在 `logs/` 里）
- **数据盘持久化**：`/root/autodl-tmp/` 里的所有东西关机后还在
- **系统盘**：Vulkan 配置（`/usr/share/glvnd/`）一般也在，但保险起见开机检查一下
- **显存不够**：把 `--num_envs` 从 4096 降到 2048 或 1024
- **AutoDL 费用**：训练时 GPU 持续满载，注意账户余额
