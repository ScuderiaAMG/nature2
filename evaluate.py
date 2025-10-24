# evaluate.py
import gymnasium as gym
import numpy as np
import torch
from model import DQN
from utils import AtariPreprocessor, FrameStack

def evaluate(model_path, game_name, n_episodes=30, max_frames=18000):
    env = gym.make(game_name)
    n_actions = env.action_space.n
    device = torch.device("cuda")
    policy_net = DQN(n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        preprocessor = AtariPreprocessor()
        frame_stack = FrameStack()

        for _ in range(np.random.randint(1, 31)):
            obs, _, _, _, _ = env.step(0)

        frame_stack.add_frame(preprocessor.process(obs))
        episode_reward = 0
        frames = 0

        while frames < max_frames:
            if np.random.random() > 0.05: # ε=0.05
                state = frame_stack.get_state()
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame_stack.add_frame(preprocessor.process(obs))
            episode_reward += reward
            frames += 1
            if done:
                break

        scores.append(episode_reward)
    env.close()
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    import sys
    model = sys.argv[1]
    game = sys.argv[2] if len(sys.argv) > 2 else "BreakoutNoFrameskip-v4"
    mean, std = evaluate(model, game)
    print(f"Evaluation Result for {game}: {mean:.2f} ± {std:.2f}")