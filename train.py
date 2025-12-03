# train.py
import gymnasium as gym
import random
import numpy as np
import torch
import os
from agent import DQNAgent
from utils import AtariPreprocessor, FrameStack

def train(game_name, save_path):
    os.makedirs("models", exist_ok=True)
    os.makedirs(f"runs/{game_name}", exist_ok=True)
    ale_game_name = f"ALE/{game_name.split('NoFrameskip')[0]}-v5"
    env = gym.make(
        ale_game_name,
        frameskip=4,                 
        repeat_action_probability=0.0, 
        full_action_space=False       
    )
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(n_actions, device, log_dir=f"runs/{game_name}")
    total_env_frames = 0
    TARGET_FRAMES = 50_000_000  # 保持5000万帧（论文标准）
    FRAME_SKIP = 4
    REPLAY_START_SIZE = 50_000  # 论文要求：预填充5万帧经验
    
    # ========== 新增：预填充经验回放缓冲区 ==========
    print(f"开始预填充经验（{REPLAY_START_SIZE}帧）...")
    obs, _ = env.reset()
    preprocessor = AtariPreprocessor()
    frame_stack = FrameStack()
    while total_env_frames < REPLAY_START_SIZE:
        # 随机选择动作（不学习）
        action = random.randrange(n_actions)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        clipped_reward = np.sign(reward)
        # 预处理帧
        processed_frame = preprocessor.process(next_obs)
        frame_stack.add_frame(processed_frame)
        state = frame_stack.get_state()
        next_state = frame_stack.get_state()  # 预填充阶段简化，实际和训练一致
        # 存入缓冲区
        agent.memory.push(state, action, clipped_reward, next_state, done)
        total_env_frames += FRAME_SKIP
        if done:
            obs, _ = env.reset()
            frame_stack.reset()
            # 初始随机noop（和训练一致）
            for _ in range(random.randint(1, 30)):
                obs, _, _, _, _ = env.step(0)
    print(f"预填充完成！当前总帧数：{total_env_frames/1e6:.1f}M")
    
    # ========== 原有训练逻辑（保持不变） ==========
    while total_env_frames < TARGET_FRAMES:
        obs, _ = env.reset()
        preprocessor = AtariPreprocessor()
        frame_stack = FrameStack()
        for _ in range(random.randint(1, 30)):
            obs, _, _, _, _ = env.step(0)
        frame_stack.add_frame(preprocessor.process(obs))
        episode_reward = 0
        while total_env_frames < TARGET_FRAMES:
            state = frame_stack.get_state()
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clipped_reward = np.sign(reward)
            processed_frame = preprocessor.process(next_obs)
            frame_stack.add_frame(processed_frame)
            next_state = frame_stack.get_state()
            agent.memory.push(state, action, clipped_reward, next_state, done)
            # 只有缓冲区足够大才优化（已满足，保险起见保留）
            if len(agent.memory) >= 32:
                agent.optimize_model()
            episode_reward += reward
            total_env_frames += FRAME_SKIP
            if done:
                break
        # 每100万帧打印日志+保存中间模型（新增：保存最优模型）
        if total_env_frames // 1_000_000 > (total_env_frames - FRAME_SKIP) // 1_000_000:
            print(f"[{game_name}] Frames: {total_env_frames/1e6:.1f}M, Reward: {episode_reward}")
            # 保存中间模型（避免只保存最后未收敛模型）
            torch.save(agent.policy_net.state_dict(), f"models/dqn_{game_name.split('NoFrameskip')[0].lower()}_epoch_{total_env_frames//1e6:.0f}M.pth")
    
    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), save_path)
    agent.writer.close()
    env.close()
    print(f"Training completed. Model saved to {save_path}.")

# def train(game_name, save_path):
#     os.makedirs("models", exist_ok=True)
#     os.makedirs(f"runs/{game_name}", exist_ok=True)
#     ale_game_name = f"ALE/{game_name.split('NoFrameskip')[0]}-v5"
#     env = gym.make(
#     ale_game_name,
#     frameskip=4,                 
#     repeat_action_probability=0.0, 
#     full_action_space=False       
# )
#     n_actions = env.action_space.n
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     agent = DQNAgent(n_actions, device, log_dir=f"runs/{game_name}")
#     total_env_frames = 0
#     TARGET_FRAMES = 50_000_000
#     FRAME_SKIP = 4

#     while total_env_frames < TARGET_FRAMES:
#         obs, _ = env.reset()
#         preprocessor = AtariPreprocessor()
#         frame_stack = FrameStack()

#         for _ in range(random.randint(1, 30)):
#             obs, _, _, _, _ = env.step(0)

#         frame_stack.add_frame(preprocessor.process(obs))
#         episode_reward = 0

#         while total_env_frames < TARGET_FRAMES:
#             # if total_env_frames % FRAME_SKIP == 0:
#             #     state = frame_stack.get_state()
#             #     action = agent.select_action(state)
#             state = frame_stack.get_state()
#             action = agent.select_action(state)

#             next_obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             clipped_reward = np.sign(reward)

#             processed_frame = preprocessor.process(next_obs)
#             frame_stack.add_frame(processed_frame)
#             next_state = frame_stack.get_state()

#             agent.memory.push(state, action, clipped_reward, next_state, done)
#             agent.optimize_model()

#             episode_reward += reward
#             total_env_frames += 4

#             if done:
#                 break

#         if total_env_frames // 1_000_000 > (total_env_frames - 1) // 1_000_000:
#             print(f"[{game_name}] Frames: {total_env_frames/1e6:.1f}M, Reward: {episode_reward}")

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     torch.save(agent.policy_net.state_dict(), save_path)
#     agent.writer.close()
#     env.close()
#     print(f"Training for {game_name} completed. Model saved to {save_path}.")

if __name__ == "__main__":
    import sys
    game = sys.argv[1] if len(sys.argv) > 1 else "BreakoutNoFrameskip-v4"
    train(game, f"models/dqn_{game.split('NoFrameskip')[0].lower()}.pth")