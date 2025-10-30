# test.py
import gymnasium as gym

env = gym.make('ALE/Breakout-v5')
print("环境创建成功")

obs, info = env.reset()
print(f"观测形状: {obs.shape}")

env.close()