# 第三方组件与许可

本仓库的 MIT 许可只覆盖我编写的代码。以下组件来自开源社区，**各自遵循其原始许可**：

| 组件 | 用途 | 许可 | 地址 |
| :--- | :--- | :--- | :--- |
| robopal | MuJoCo 机械臂仿真框架（PiPER 环境、键盘遥操采集） | Apache-2.0 | [NoneJou072/robopal](https://github.com/NoneJou072/robopal) |
| robomimic | 模仿学习训练/推理框架（Diffusion Policy 实现） | MIT | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| MuJoCo | 物理引擎 | Apache-2.0 | [mujoco.org](https://mujoco.org) |

## 关于 `src/` 与 `config/`

`src/` 下的文件是上述框架的**修改副本**，`config/diffusion_policy.json` 对应 robomimic 的 `exps/templates/diffusion_policy.json`。它们保留原项目的版权声明，在此仅用于说明我的改动。

## 关于 `data/demo.hdf5`

这是我在该项目中**自己采集**的 50 条 PiPER 抓取演示数据，随仓库一并提供，便于复现训练结果。
