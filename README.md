# nature2

## 2025.10.25 First start.
## 2025.10.26 Second start.


dqn_nature2015/

│

├── setup.sh                     # 环境一键安装脚本

├── requirements.txt             # Python 依赖列表

│

├── model.py                     # DQN 网络架构（3 Conv + 2 FC）

├── agent.py                     # DQN 代理（含 Experience Replay + Target Network + TensorBoard）

├── utils.py                     # Atari 预处理（Max over 2 frames, gray, crop, resize, stack）

│

├── train.py                     # 单游戏训练脚本（50M 帧，frame skip=4）

├── evaluate.py                  # 评估脚本（30 episodes, ε=0.05, 5分钟限制）

├── train_all.py                 # 多游戏训练入口（49个Atari游戏）

│

├── runs/                        # TensorBoard 日志目录（自动生成）

├── models/                      # 模型保存目录（自动生成）

│

└── README.md                    # 项目说明（含复现验证指南）