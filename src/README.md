# `src/`：对上游框架的覆盖文件（drop-in overrides）

本目录里的文件**不是**独立可运行的项目，而是对两个开源框架的**局部改动**。按下面的相对路径覆盖到上游仓库即可还原我本地的运行环境。

- [robopal](https://github.com/NoneJou072/robopal)（Apache-2.0）— 仿真与演示采集
- [robomimic](https://github.com/ARISE-Initiative/robomimic)（MIT）— 训练与推理框架

## 覆盖规则

```text
本仓库                                          →  上游仓库
src/robopal/scripts/collect_human_demonstrations.py → robopal/robopal/scripts/collect_human_demonstrations.py
src/robopal/scripts/play_back_demonstrations.py     → robopal/robopal/scripts/play_back_demonstrations.py
src/robopal/assets/scenes/grasping.xml              → robopal/robopal/assets/scenes/grasping.xml
src/robomimic/algo/diffusion_policy.py              → robomimic/robomimic/algo/diffusion_policy.py
src/robomimic/scripts/run_trained_agent.py          → robomimic/robomimic/scripts/run_trained_agent.py
src/robomimic/utils/lang_utils.py                   → robomimic/robomimic/utils/lang_utils.py
config/diffusion_policy.json                        → robomimic/robomimic/exps/templates/diffusion_policy.json
```

## 每个文件改了什么

| 文件 | 改动点 |
| :--- | :--- |
| `robopal/assets/scenes/grasping.xml` | 补全 `TODO`：`<include file=...table.xml>` 与 `<mesh file=...table.STL>` 指向本地真实路径（**请改成你自己的绝对路径**） |
| `robopal/scripts/play_back_demonstrations.py` | 补全 `TODO`：默认回放目录指向本机 `collections/collections_*`，并在回放前加载 episode 的 model string |
| `robomimic/algo/diffusion_policy.py` | 推理时对单帧观测补 batch / `observation_horizon` 维，兼容 `[D]`、`[B,D]`、`[B,T,D]` |
| `robomimic/scripts/run_trained_agent.py` | 成功率统计同时参考 `env.is_success()`、`info` 与 `reward >= 1.0`；兼容 gym / gymnasium 的 4/5 元组 `step` 与 tuple / 非 tuple `reset` |
| `robomimic/utils/lang_utils.py` | 低维训练不需要 CLIP，改为本地缓存加载，避免导入期强制联网拉取 `openai/clip-vit-large-patch14` |

## 注意

- 上游版本会迭代，覆盖前请先 `git diff` 确认没有冲突；本仓库记录的是课程提供版本上的改动。
- 这些文件保留了上游的版权与许可证约束，仅用于说明我在其上的修改。
