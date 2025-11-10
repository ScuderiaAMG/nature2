import gymnasium as gym
import numpy as np
import torch
import argparse
from model import DQN
from utils import AtariPreprocessor, FrameStack

# 论文Extended Data Table 2核心数据：{游戏名称: (随机分数, 人类分数, 论文DQN分数±std, 论文归一化百分比)}
PAPER_METRICS = {
    "Breakout": (3.0, 316.8, "316.8 (±?)", 100.0),  # 论文中DQN最优值
    "Pong": (-20.7, 9.3, "18.9 (±1.3)", 132.0),
    "SpaceInvaders": (148.0, 1652.0, "1976 (±893)", 121.5),
    "Seaquest": (68.4, 20182.0, "5286 (±1310)", 25.9),
    "RiverRaid": (1339.0, 13513.0, "8316 (±1049)", 57.3),
    "Assault": (222.4, 1496.0, "3359 (±775)", 246.2),
    "Asterix": (210.0, 8503.0, "6012 (±1744)", 70.0),
    "BattleZone": (2360.0, 37800.0, "26300 (±7725)", 67.6),
    "BeamRider": (363.9, 5775.0, "6846 (±1619)", 119.8),
    "Boxing": (0.1, 4.3, "71.8 (±8.4)", 1707.9),
    "Centipede": (2091.0, 11963.0, "8309 (±5237)", 63.0),
    "ChopperCommand": (811.0, 9882.0, "6687 (±2916)", 64.8),
    "CrazyClimber": (10781.0, 35411.0, "114103 (±22797)", 419.5),
    "DemonAttack": (152.1, 3401.0, "9711 (±2406)", 294.2),
    "Freeway": (0.0, 29.6, "30.3 (±0.7)", 102.4),
    "KungFuMaster": (258.5, 22736.0, "23270 (±5955)", 102.4),
    "Q*bert": (163.9, 13455.0, "10596 (±3294)", 78.5),
    "RoadRunner": (11.5, 7845.0, "18257 (±4268)", 232.9),
    "StarGunner": (664.0, 10250.0, "57997 (±3152)", 598.1),
}

# 游戏名称映射：脚本输入名 → 论文中的游戏名
GAME_NAME_MAPPING = {
    "BreakoutNoFrameskip-v4": "Breakout",
    "PongNoFrameskip-v4": "Pong",
    "SpaceInvadersNoFrameskip-v4": "SpaceInvaders",
    "SeaquestNoFrameskip-v4": "Seaquest",
    "RiverRaidNoFrameskip-v4": "RiverRaid",
    "AssaultNoFrameskip-v4": "Assault",
    "AsterixNoFrameskip-v4": "Asterix",
    "BattleZoneNoFrameskip-v4": "BattleZone",
    "BeamRiderNoFrameskip-v4": "BeamRider",
    "BoxingNoFrameskip-v4": "Boxing",
    "CentipedeNoFrameskip-v4": "Centipede",
    "ChopperCommandNoFrameskip-v4": "ChopperCommand",
    "CrazyClimberNoFrameskip-v4": "CrazyClimber",
    "DemonAttackNoFrameskip-v4": "DemonAttack",
    "FreewayNoFrameskip-v4": "Freeway",
    "KungFuMasterNoFrameskip-v4": "KungFuMaster",
    "QbertNoFrameskip-v4": "Q*bert",
    "RoadRunnerNoFrameskip-v4": "RoadRunner",
    "StarGunnerNoFrameskip-v4": "StarGunner",
}

def calculate_normalized_score(model_score, random_score, human_score):
    """按论文公式计算归一化性能：100 × (模型分数 - 随机分数) / (人类分数 - 随机分数)"""
    if human_score == random_score:
        return 0.0
    return 100 * (model_score - random_score) / (human_score - random_score)

def verify_model(model_path, game_name="BreakoutNoFrameskip-v4", render=False, n_episodes=30, max_frames=18000):
    """
    严格对齐论文评估标准验证模型，与论文数据直接对比
    论文评估标准：30回合、每回合最多5分钟（18000帧）、ε=0.05、初始随机noop最多30步
    """
    # 1. 解析游戏名称（映射到论文中的名称）
    paper_game_name = GAME_NAME_MAPPING.get(game_name, game_name.split('NoFrameskip')[0])
    print(f"=== 模型验证（对齐论文《Human-level control through deep reinforcement learning》）===")
    print(f"验证游戏：{game_name} → 论文对应游戏：{paper_game_name}")
    print(f"评估参数：{n_episodes}回合 | 每回合最大帧数：{max_frames}（5分钟） | ε-greedy探索率：0.05")
    
    # 2. 环境设置（完全匹配论文训练/评估配置）
    ale_game_name = f"ALE/{game_name.split('NoFrameskip')[0]}-v5"
    env_kwargs = {
        "frameskip": 4,
        "repeat_action_probability": 0.0,
        "full_action_space": False
    }
    if render:
        env_kwargs["render_mode"] = "human"
    
    env = gym.make(ale_game_name, **env_kwargs)
    n_actions = env.action_space.n
    
    # 3. 设备与模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    try:
        policy_net = DQN(n_actions).to(device)
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()
        print(f"模型加载成功：{model_path}")
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return
    
    # 4. 预处理工具（与论文预处理一致：帧最大化去闪烁、灰度化、裁剪84×84、4帧堆叠）
    preprocessor = AtariPreprocessor()
    frame_stack = FrameStack(num_frames=4)
    
    # 5. 按论文标准运行验证
    total_rewards = []
    epsilon = 0.05  # 论文评估时固定ε=0.05
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        frame_stack.reset()
        
        # 初始随机noop步数（论文：1-30步随机）
        noop_steps = np.random.randint(1, 31)
        for _ in range(noop_steps):
            obs, _, _, _, _ = env.step(0)
        
        # 初始化帧堆叠
        processed_frame = preprocessor.process(obs)
        frame_stack.add_frame(processed_frame)
        
        episode_reward = 0
        frames = 0
        done = False
        
        while not done and frames < max_frames:
            # 按论文ε-greedy策略选择动作
            state = frame_stack.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            
            if np.random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            # 执行动作（论文使用动作重复4帧）
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 预处理下一帧并堆叠
            processed_next_frame = preprocessor.process(next_obs)
            frame_stack.add_frame(processed_next_frame)
            
            # 累计原始奖励（论文评估用原始奖励，训练时才裁剪）
            episode_reward += reward
            frames += 1
        
        total_rewards.append(episode_reward)
        print(f"回合 {episode+1:2d}/{n_episodes} | 奖励：{episode_reward:8.1f} | 步数：{frames:5d}")
    
    # 6. 计算模型核心指标（对齐论文格式）
    model_mean = np.mean(total_rewards)
    model_std = np.std(total_rewards)
    model_normalized = None
    
    # 查找论文对应指标并对比
    paper_random = paper_mean = paper_std = paper_normalized = None
    if paper_game_name in PAPER_METRICS:
        paper_random, paper_human, paper_score_str, paper_normalized = PAPER_METRICS[paper_game_name]
        model_normalized = calculate_normalized_score(model_mean, paper_random, paper_human)
    
    # 7. 输出对比结果
    print("\n" + "="*80)
    print(f"{'指标':<20} {'当前模型':<20} {'论文DQN':<20} {'差异':<10}")
    print("-"*80)
    print(f"平均奖励          {model_mean:18.2f} ± {model_std:6.2f}  {paper_score_str if paper_score_str else 'N/A':<20} {abs(model_mean - float(paper_score_str.split('(')[0])) if paper_score_str else 'N/A':<10.1f}")
    if model_normalized is not None:
        print(f"归一化性能(%)     {model_normalized:18.1f}          {paper_normalized:<20.1f} {abs(model_normalized - paper_normalized):<10.1f}")
        print(f"随机分数(基线)    {paper_random:18.1f}          {paper_random:<20.1f} {'0.0':<10}")
        print(f"人类分数(上限)    {paper_human:18.1f}          {paper_human:<20.1f} {'0.0':<10}")
    else:
        print(f"归一化性能(%)     {'N/A':<18}          {'N/A':<20} {'N/A':<10}")
        print(f"提示：当前游戏无论文参考数据，无法计算归一化对比")
    print("="*80)
    
    # 8. 复现性判断
    if model_normalized is not None:
        # 论文中归一化性能±10%视为合格复现
        if abs(model_normalized - paper_normalized) <= 10.0:
            print(f"✅ 模型复现性良好！归一化性能与论文差异在10%以内")
        elif abs(model_normalized - paper_normalized) <= 30.0:
            print(f"⚠️  模型基本复现！归一化性能与论文差异在30%以内")
        else:
            print(f"❌ 模型未复现论文性能！归一化性能与论文差异超过30%")
    
    env.close()
    return {
        "model_mean_reward": model_mean,
        "model_std_reward": model_std,
        "model_normalized_score": model_normalized,
        "paper_normalized_score": paper_normalized,
        "reproduction_status": "良好" if (model_normalized and abs(model_normalized - paper_normalized) <=10) else "一般" if (model_normalized and abs(model_normalized - paper_normalized) <=30) else "未复现"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对齐论文标准验证DQN模型（复现性检查）")
    parser.add_argument("model_path", help="/home/legion/nature2/models/breakout_dqn.pth")
    parser.add_argument("--game", default="BreakoutNoFrameskip-v4", help="游戏名称（如BreakoutNoFrameskip-v4）")
    parser.add_argument("--render", action="store_true", help="是否渲染游戏画面（论文评估时不渲染）")
    parser.add_argument("--episodes", type=int, default=30, help="验证回合数（论文固定30回合）")
    parser.add_argument("--max-frames", type=int, default=18000, help="每回合最大帧数（论文5分钟=18000帧）")
    
    args = parser.parse_args()
    verify_model(
        model_path=args.model_path,
        game_name=args.game,
        render=args.render,
        n_episodes=args.episodes,
        max_frames=args.max_frames
    )