
# test.py
import gymnasium as gym

env = gym.make('ALE/Breakout-v5', frameskip=4, repeat_action_probability=0.0)
print("环境创建")

obs, info = env.reset()
print(f"观测形状: {obs.shape}")

obs, reward, terminated, truncated, info = env.step(1)
print(f"执行动作后，奖励: {reward}")

env.close()
print("环境关闭")