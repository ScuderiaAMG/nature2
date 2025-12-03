# agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
from model import DQN

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# class ReplayBuffer:
#     def __init__(self, capacity=1_000_000):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, *args):
#         self.buffer.append(Experience(*args))

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def __len__(self):
#         return len(self.buffer)
# class ReplayBuffer:
#     def __init__(self, capacity=1_000_000):
#         self.capacity = capacity
#         self.size = 0        
#         self.pos = 0        
#         # 预分配 NumPy 数组（大幅节省内存）
#         self.states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
#         self.actions = np.empty((capacity,), dtype=np.int64)
#         self.rewards = np.empty((capacity,), dtype=np.float32)
#         self.next_states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
#         self.dones = np.empty((capacity,), dtype=np.bool_)

#     def push(self, state, action, reward, next_state, done):
#         self.states[self.pos] = state
#         self.actions[self.pos] = action
#         self.rewards[self.pos] = reward
#         self.next_states[self.pos] = next_state
#         self.dones[self.pos] = done
        
#         self.pos = (self.pos + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size):
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         return (
#             self.states[idxs],
#             self.actions[idxs],
#             self.rewards[idxs],
#             self.next_states[idxs],
#             self.dones[idxs]
#         )

#     def __len__(self):
#         return self.size
# agent.py 中 ReplayBuffer 初始化
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):  # 改为100万帧（论文标准）
        self.capacity = capacity
        self.size = 0        
        self.pos = 0  
        # 保持其他参数不变（状态维度4×84×84）
        self.states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.bool_)
# class ReplayBuffer:
#     def __init__(self, capacity=210_000):
#         self.capacity = capacity
#         self.size = 0        
#         self.pos = 0  
        
#         self.states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
#         self.actions = np.empty((capacity,), dtype=np.int64)
#         self.rewards = np.empty((capacity,), dtype=np.float32)
#         self.next_states = np.empty((capacity, 4, 84, 84), dtype=np.float32)
#         self.dones = np.empty((capacity,), dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)  

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )
    
    def __len__(self):
        return self.size
# class DQNAgent:
#     def __init__(self, n_actions, device, log_dir="runs"):
#         self.device = device
#         self.n_actions = n_actions
#         self.steps_done = 0
#         self.epsilon = 1.0
#         self.policy_net = DQN(n_actions).to(device)
#         self.target_net = DQN(n_actions).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
#         self.optimizer = optim.RMSprop(
#             self.policy_net.parameters(),
#             lr=0.00025,
#             alpha=0.95,
#             eps=0.01,
#             momentum=0.0,
#             centered=False
#         )
#         self.memory = ReplayBuffer(capacity=210_000)
#         self.writer = SummaryWriter(log_dir=log_dir)
#         self.log_step = 0

        

#     def select_action(self, state):
#         if random.random() > self.epsilon:
#             with torch.no_grad():
#                 state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
#                 return self.policy_net(state_tensor).max(1)[1].item()
#         else:
#             return random.randrange(self.n_actions)

#     # def optimize_model(self, batch_size=32, gamma=0.99):
#     #     if len(self.memory) < batch_size:
#     #         return

#     #     experiences = self.memory.sample(batch_size)
#     #     #batch = Experience(*zip(*experiences))

#     #     # state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
#     #     # action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
#     #     # reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
#     #     # next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
#     #     # done_batch = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
#     #     states, actions, rewards, next_states, dones = zip(*experiences)

#     #     state_batch = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
#     #     action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
#     #     reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
#     #     next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
#     #     done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
#     def optimize_model(self, batch_size=32, gamma=0.99):
#         if len(self.memory) < batch_size:
#             return

#         states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

#         state_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
#         action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
#         reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
#         next_state_batch = torch.tensor(next_states, dtype=torch.float32, device=self.device)
#         done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)

#         current_q = self.policy_net(state_batch).gather(1, action_batch)
#         with torch.no_grad():
#             next_q = self.target_net(next_state_batch).max(1)[0].detach()
#             target_q = reward_batch + (gamma * next_q * (~done_batch))

#         loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         for param in self.policy_net.parameters():
#             param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()

#         if self.steps_done < 1_000_000:
#             self.epsilon = 1.0 - 0.9 * (self.steps_done / 1_000_000)
#         else:
#             self.epsilon = 0.1

#         self.writer.add_scalar('Loss', loss.item(), self.log_step)
#         self.writer.add_scalar('Epsilon', self.epsilon, self.log_step)
#         self.log_step += 1

#         if self.steps_done % 10_000 == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())
class DQNAgent:
    def __init__(self, n_actions, device, log_dir="runs"):
        self.device = device
        self.n_actions = n_actions
        self.steps_done = 0
        self.epsilon = 1.0
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=0.00025,
            alpha=0.95,
            eps=0.01,
            momentum=0.0,
            centered=False
        )
        self.memory = ReplayBuffer(capacity=210_000)  # 已改为100万帧
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_step = 0
        self.optimize_count = 0  # 新增：计数优化次数
        self.target_update_freq = 1_000  # 论文：每10000次优化更新目标网络
    
    def optimize_model(self, batch_size=32, gamma=0.99):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        state_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].detach()
            target_q = reward_batch + (gamma * next_q * (~done_batch))
        
        loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 梯度裁剪（论文要求）
        self.optimizer.step()
        
        # Epsilon衰减（保持不变，和论文一致）
        if self.steps_done < 1_000_000:
            self.epsilon = 1.0 - 0.9 * (self.steps_done / 210_000)
        else:
            self.epsilon = 0.1
        
        # 日志记录
        self.writer.add_scalar('Loss', loss.item(), self.log_step)
        self.writer.add_scalar('Epsilon', self.epsilon, self.log_step)
        self.log_step += 1
        self.optimize_count += 1  # 优化次数+1
        
        # 按优化次数更新目标网络（原按steps_done，改为按optimize_count）
        if self.optimize_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"目标网络更新！当前优化次数：{self.optimize_count}")

    def select_action(self, state):
        """
        实现论文中的ε-greedy动作选择：
        - 以ε概率随机选择动作
        - 以1-ε概率选择当前Q值最大的动作
        """
        self.steps_done += 1  # 累计步数（用于ε衰减）
        # 生成随机数判断是否探索
        if np.random.random() > self.epsilon:
            # 贪婪选择：使用当前策略网络预测最优动作
            state_tensor = torch.tensor(
                state, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # 增加批次维度 (1, 4, 84, 84)
            with torch.no_grad():
                # 选择Q值最大的动作索引
                action = self.policy_net(state_tensor).max(1)[1].item()
            return action
        else:
            # 随机探索：从动作空间中随机选择
            return np.random.randint(self.n_actions)
        

    