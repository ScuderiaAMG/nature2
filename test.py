
# test.py
import gymnasium as gym

# 直接使用 ALE/...-v5，无需任何额外命令
env = gym.make('ALE/Breakout-v5', frameskip=4, repeat_action_probability=0.0)
print("✅ 环境创建成功！")

obs, info = env.reset()
print(f"✅ 观测形状: {obs.shape}")

# 执行一个动作
obs, reward, terminated, truncated, info = env.step(1)
print(f"✅ 执行动作后，奖励: {reward}")

env.close()
print("✅ 环境关闭。一切正常！")