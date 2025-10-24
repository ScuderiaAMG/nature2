# test.py
import gymnasium as gym

# 这个应该能成功了！
env = gym.make('ALE/Breakout-v5')
print("✅ 环境创建成功！")

obs, info = env.reset()
print(f"✅ 观测形状: {obs.shape}")

env.close()