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

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

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
        self.memory = ReplayBuffer()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_step = 0

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self, batch_size=32, gamma=0.99):
        if len(self.memory) < batch_size:
            return

        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].detach()
            target_q = reward_batch + (gamma * next_q * (~done_batch))

        loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Epsilon decay over first 1M env frames
        if self.steps_done < 1_000_000:
            self.epsilon = 1.0 - 0.9 * (self.steps_done / 1_000_000)
        else:
            self.epsilon = 0.1

        # Log to TensorBoard
        self.writer.add_scalar('Loss', loss.item(), self.log_step)
        self.writer.add_scalar('Epsilon', self.epsilon, self.log_step)
        self.log_step += 1

        # Update target network every 10k agent steps
        if self.steps_done % 10_000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())