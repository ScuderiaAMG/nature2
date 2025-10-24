# train.py
import gymnasium as gym
import random
import numpy as np
import torch
from agent import DQNAgent
from utils import AtariPreprocessor, FrameStack

def train(game_name, save_path):
    # env = gym.make(game_name)
    # 将 "BreakoutNoFrameskip-v4" 转换为 "ALE/Breakout-v5"
    ale_game_name = f"ALE/{game_name.split('NoFrameskip')[0]}-v5"
    env = gym.make(
    ale_game_name,
    frameskip=4,                  # 严格对齐论文的 frame skip=4
    repeat_action_probability=0.0, # 禁用 sticky actions，确保行为确定
    full_action_space=False       # 使用游戏的最小动作集
)
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(n_actions, device, log_dir=f"runs/{game_name}")
    total_env_frames = 0
    TARGET_FRAMES = 50_000_000
    FRAME_SKIP = 4

    while total_env_frames < TARGET_FRAMES:
        obs, _ = env.reset()
        preprocessor = AtariPreprocessor()
        frame_stack = FrameStack()

        # Random no-op start
        for _ in range(random.randint(1, 30)):
            obs, _, _, _, _ = env.step(0)

        frame_stack.add_frame(preprocessor.process(obs))
        episode_reward = 0

        while total_env_frames < TARGET_FRAMES:
            # if total_env_frames % FRAME_SKIP == 0:
            #     state = frame_stack.get_state()
            #     action = agent.select_action(state)
            state = frame_stack.get_state()
            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clipped_reward = np.sign(reward)

            processed_frame = preprocessor.process(next_obs)
            frame_stack.add_frame(processed_frame)
            next_state = frame_stack.get_state()

            agent.memory.push(state, action, clipped_reward, next_state, done)
            agent.optimize_model()

            episode_reward += reward
            total_env_frames += 4

            if done:
                break

        if total_env_frames // 1_000_000 > (total_env_frames - 1) // 1_000_000:
            print(f"[{game_name}] Frames: {total_env_frames/1e6:.1f}M, Reward: {episode_reward}")

    torch.save(agent.policy_net.state_dict(), save_path)
    agent.writer.close()
    env.close()
    print(f"Training for {game_name} completed. Model saved to {save_path}.")

if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "BreakoutNoFrameskip-v4"
    train(game, f"models/dqn_{game.split('NoFrameskip')[0].lower()}.pth")