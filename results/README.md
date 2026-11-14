# `results/`：原始数据出处

| 文件 | 说明 |
| :--- | :--- |
| `train_log.txt` | Epoch 100 完整训练日志（含每 10 epoch 的 5 次内部 rollout 统计） |
| `run_config.json` | 该次训练自动保存的完整配置快照 |
| `tensorboard/` | TensorBoard event 文件，可直接 `tensorboard --logdir results` 复看 |
| `train_curve.png` | Train/Loss、学习率、Policy_Grad_Norms |
| `rollout_metrics.png` | Rollout/Horizon、Rollout/Return（内部 rollout，仅看趋势） |
| `system_timing.png` | 内存占用与耗时 |
| `eval_run_trained_agent.json` | `run_trained_agent.py` 独立评估结果（见下） |

## 关于 `eval_run_trained_agent.json`

数值抄录自实验报告 §8 的终端输出，用于让数字可追溯：

```json
{
  "Return": 1.0,
  "Horizon": 315.6,
  "Success_Rate": 1.0,
  "Num_Success": 5.0
}
```

即 5 次 rollout 全部成功。同一文件里还记录了修复评估脚本后、用初始策略复现到的基线（2/5 = 0.4）。

## 三个容易误读的点

1. **训练日志里的 `Success_Rate: 0.0` 不能当结论。** 上游 `train.py` 的内部 rollout 统计存在 bug（任务完成但 `is_success()["task"]` 未同步），所以它一直为 0；正式口径是 `run_trained_agent.py` 的输出。
2. **`Return` 与 `Success_Rate` 不是一回事。** 修复前出现过 `Return=1.0` 但 `Success_Rate=0.0` 的自相矛盾输出，这正是本项目定位并修掉的问题。
3. **40% → 100% 不是"改了个 bug"。** 修复只让成功率变得可测；真正的提升来自重采演示数据 + 时序超参（`prediction_horizon 16` / `action_horizon 4`）+ 100 epoch 训练。详见根目录 README「4.5」。

## 权重

`last.pth`（约 1.0 GB）未入库，用 `README` 的训练命令可复现。
