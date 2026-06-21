# Nature2: A PyTorch Reproduction of Deep Q-Networks for Atari 2600

## Abstract

This repository presents a faithful reimplementation of the Deep Q-Network (DQN) algorithm introduced by Mnih et al. (2015) in the seminal work *"Human-level control through deep reinforcement learning,"* published in *Nature*. The DQN architecture represents a landmark achievement in artificial intelligence, demonstrating that a single neural network architecture, trained end-to-end from high-dimensional pixel inputs via a novel variant of Q-learning, can achieve human-level or superhuman performance across 49 distinct Atari 2600 games. This implementation faithfully reproduces the canonical algorithmic components --- experience replay, target network stabilization, frame preprocessing pipelines, and Huber loss optimization --- using PyTorch and the Gymnasium Arcade Learning Environment (ALE). The codebase provides training, evaluation, and paper-aligned verification tools, enabling rigorous reproduction and benchmarking of the original experimental results.

---

## 1. Theoretical Foundations

### 1.1 Markov Decision Processes and the Optimal Action-Value Function

We model each Atari 2600 game as a Markov Decision Process (MDP) defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$, where $\mathcal{S}$ denotes the set of environment states (represented as sequences of preprocessed game frames), $\mathcal{A}$ is the discrete set of available actions (ranging from 4 to 18 depending on the game), $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the state transition probability kernel, $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the bounded reward function, and $\gamma \in [0, 1)$ is the discount factor that controls the present value of future rewards.

The objective of reinforcement learning is to discover an optimal policy $\pi^*: \mathcal{S} \to \mathcal{A}$ that maximizes the expected discounted cumulative return. Formally, the optimal action-value function $Q^*(s, a)$ satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}_{s^{\prime} \sim \mathcal{P}(\cdot \mid s,a)}\left[\mathcal{R}(s, a)+\gamma \max_{a^{\prime} \in \mathcal{A}} Q^*(s^{\prime}, a^{\prime}) \right]$$

### 1.2 Q-Learning with Function Approximation

In high-dimensional state spaces such as raw pixel observations, tabular representations of Q-values become computationally intractable. The DQN approach employs a deep convolutional neural network $Q(s, a; \theta)$ parameterized by weights $\theta$ to approximate the optimal action-value function. The network is trained by iteratively minimizing the temporal difference (TD) error via stochastic gradient descent on minibatches drawn uniformly from an experience replay buffer $\mathcal{D} = \{e_1, e_2, \ldots, e_N\}$, where each experience tuple $e_t = (s_t, a_t, r_t, s_{t+1})$ records a single agent-environment interaction.

The loss function at iteration $i$ is defined as:

$$\mathcal{L}_i(\theta_i) = \mathbb{E}_{(s, a, r, s^{\prime}) \sim U(\mathcal{D})}\left[ \left( r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime}; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$

where $\theta_i^-$ denotes the parameters of a separate *target network*, which is held fixed during optimization and periodically synchronized with the online network $\theta_i$. This decoupling of action selection from target computation mitigates the harmful positive feedback loop (i.e., the *moving target problem*) inherent in bootstrapping methods.

### 1.3 Huber Loss for Robust Optimization

Following the original paper, we replace the standard mean squared error (MSE) with the Huber loss (smooth L1), which exhibits reduced sensitivity to outlier TD errors and prevents gradient explosion. The Huber loss is defined as:

$$\mathcal{L}_{\delta}(x) = \begin{cases} \frac{1}{2}x^2 & \text{for } |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

where $x = r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime}; \theta_i^-) - Q(s, a; \theta_i)$ denotes the TD error, and $\delta = 1$ (the default in PyTorch's `smooth_l1_loss`). The Huber loss behaves quadratically near zero, providing the smoothness of MSE for small errors, while decaying linearly for large errors, thereby bounding the gradient magnitude.

### 1.4 Exploration-Exploitation via Epsilon-Greedy Annealing

The agent selects actions according to an $\epsilon$-greedy policy that anneals linearly from full exploration to predominantly exploitative behavior:

$$\epsilon(\text{steps}) = \begin{cases} 1.0 - 0.9 \cdot \frac{\text{steps}}{N_\epsilon} & \text{if } \text{steps} < N_\epsilon \\ 0.1 & \text{otherwise} \end{cases}$$

where $N_\epsilon = 1,\!000,\!000$ frames. The action selection rule is:

$$a_t = \begin{cases} \arg\max_{a} Q(s_t, a; \theta) & \text{with probability } 1 - \epsilon \\ \text{Uniform}(\mathcal{A}) & \text{with probability } \epsilon \end{cases}$$

---

## 2. Neural Network Architecture

The DQN architecture follows the canonical three-layer convolutional encoder design specified in the 2015 *Nature* paper. Given an input tensor $\mathbf{X} \in \mathbb{R}^{4 \times 84 \times 84}$ representing a stack of four consecutive preprocessed grayscale frames, the network computes:

| Layer | Type | Parameters | Activation | Output Shape |
|-------|------|-----------|------------|--------------|
| Conv1 | 2D Convolution | 32 filters, $8 \times 8$ kernel, stride 4 | ReLU | $32 \times 20 \times 20$ |
| Conv2 | 2D Convolution | 64 filters, $4 \times 4$ kernel, stride 2 | ReLU | $64 \times 9 \times 9$ |
| Conv3 | 2D Convolution | 64 filters, $3 \times 3$ kernel, stride 1 | ReLU | $64 \times 7 \times 7$ |
| FC1 | Fully Connected | $3136 \to 512$ units | ReLU | $512$ |
| FC2 | Fully Connected (Output) | $512 \to n_{\text{actions}}$ units | Linear (identity) | $n_{\text{actions}}$ |

The convolutional encoder maps raw pixel observations to a compact latent feature representation, from which the fully connected layers compute per-action Q-value estimates. Notably, no pooling layers are employed; spatial downsampling is accomplished exclusively through strided convolutions. The network is reinitialized independently for each Atari game, as the original work demonstrates that a single architecture suffices across the entire suite without per-game hyperparameter tuning.

---

## 3. Preprocessing Pipeline

Raw Atari 2600 frames are emitted at $210 \times 160$ pixels in the RGB color space (128-color palette). The preprocessing pipeline transforms each frame through the following stages:

1. **Luminance Extraction (Max-over-Frames)**: To address the flickering artifact inherent in Atari 2600 rendering --- where sprites are displayed only on alternating frames --- each raw frame $\mathbf{F}_t^{\text{raw}} \in \mathbb{R}^{210 \times 160 \times 3}$ is first combined with its immediate predecessor via element-wise maximum:

   $$\mathbf{F}_t^{\text{max}} = \max(\mathbf{F}_t^{\text{raw}}, \mathbf{F}_{t-1}^{\text{raw}})$$

2. **Grayscale Conversion**: The maximum-frame is converted to a single luminance channel via standard RGB-to-grayscale transformation.

3. **Center Cropping**: The playing area is extracted by cropping to the region $[34:194, :]$ (removing $34$ pixels from the top and $26$ from the bottom), yielding dimensions $160 \times 160$.

4. **Downsampling**: The cropped frame is resized to $84 \times 84$ using area interpolation, which provides anti-aliasing for downscaling operations.

5. **Normalization**: Pixel intensities are scaled to the interval $[0, 1]$ by division by $255$.

6. **Frame Stacking**: The last $k = 4$ preprocessed frames are concatenated along the channel dimension to form the state representation $s_t \in \mathbb{R}^{4 \times 84 \times 84}$, encoding short-term temporal dynamics essential for velocity and direction inference. During initialization, the first frame is duplicated to populate the stack.

---

## 4. Algorithm Description

### 4.1 Experience Replay Buffer

The replay memory $\mathcal{D}$ is implemented as a preallocated circular buffer of fixed capacity $C = 1,\!000,\!000$ transitions, with each transition stored as a typed NumPy array for memory efficiency. The buffer maintains a position pointer and size counter: when the buffer is at capacity, new experiences overwrite the oldest entries. At each learning step, a minibatch of $B = 32$ experiences is uniformly sampled:

$$\{e^{(i)}\}_{i=1}^{B} \sim \text{Uniform}(\mathcal{D})$$

Experience replay serves two critical purposes: (i) it breaks the strong temporal correlations present in consecutive observations, making the training distribution closer to the i.i.d. assumption of stochastic gradient methods; and (ii) it enables each experience to contribute to multiple parameter updates, substantially improving sample efficiency.

### 4.2 Action Selection and Frame Skipping

To reduce computational burden without sacrificing effective control, the agent selects an action every $k = 4$ frames and repeats the chosen action for the intervening $k - 1 = 3$ frames. This frame-skipping technique yields a roughly fourfold reduction in per-episode decision steps while maintaining the temporal granularity necessary for competent play across diverse game dynamics.

### 4.3 Training Loop

The complete training procedure is summarized as follows:

1. **Pre-filling Phase**: Execute a random policy for $N_{\text{prefill}} = 50,\!000$ environment frames, storing every transition $(s_t, a_t, r_t, s_{t+1})$ in the replay buffer $\mathcal{D}$. Each episode begins with a uniformly random number of no-operation actions (between 1 and 30) to decorrelate episode-initial states.

2. **Main Loop**: For $T_{\text{total}} = 50,\!000,\!000$ frames:

   - With probability $\epsilon$, select a uniformly random action $a_t \sim \text{Uniform}(\mathcal{A})$; otherwise, select $a_t = \arg\max_a Q(s_t, a; \theta)$.
   - Execute $a_t$ in the emulator, observing reward $r_t$ and the subsequent raw frame; clip reward to $\text{sign}(r_t) \in \{-1, 0, +1\}$.
   - Apply the preprocessing pipeline to obtain the next state $s_{t+1}$ and store the transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$.
   - Sample a minibatch $\{(s_j, a_j, r_j, s_j')\}_{j=1}^{B}$ from $\mathcal{D}$.
   - Compute the target value for each sample:

     $$y_j = \begin{cases} r_j & \text{if episode terminates at step } j+1 \\ r_j + \gamma \max_{a^{\prime}} Q(s_j^{\prime}, a^{\prime}; \theta^-) & \text{otherwise} \end{cases}$$

   - Perform one gradient descent step on the Huber loss with respect to $\theta$.
   - Clip gradients elementwise to the interval $[-1, 1]$.
   - Every $C_{\text{target}} = 1,\!000$ optimization steps, synchronize the target network: $\theta^- \leftarrow \theta$ (hard update).

3. **Checkpointing**: Save intermediate model weights every $1,\!000,\!000$ frames.

### 4.4 Gradient Clipping

Error clipping: all gradients are clamped to the range $[-1, 1]$ before the optimizer step, following the original paper. This prevents occasional large TD errors -- common in reinforcement learning due to the bootstrapping formulation and non-stationary data distribution -- from causing destabilizing parameter updates:

$$\nabla_\theta \mathcal{L} \leftarrow \text{clip}\left(\nabla_\theta \mathcal{L},\ -1,\ 1\right)$$

---

## 5. Optimization Configuration

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | RMSProp | Momentum-free adaptive learning rates for non-stationary RL |
| Learning rate $\eta$ | $2.5 \times 10^{-4}$ | Paper-specified |
| RMSProp $\alpha$ (smoothing) | $0.95$ | Exponential moving average of squared gradients |
| RMSProp $\epsilon$ | $0.01$ | Numerical stability constant |
| RMSProp momentum | $0.0$ | No classical momentum (as per paper) |
| Discount factor $\gamma$ | $0.99$ | Standard for episodic tasks with moderate horizon |
| Minibatch size $B$ | $32$ | Balances gradient variance and computational cost |
| Replay buffer capacity $C$ | $1 \times 10^6$ | Stores approximately 20,000 episodes of experience |
| Target network update interval $C_{\text{target}}$ | $1,\!000$ | Optimization steps between hard target syncs |
| Total training frames $T_{\text{total}}$ | $5 \times 10^7$ | Approximately 38 days of real-time gameplay at 60 fps |
| Replay start size $N_{\text{prefill}}$ | $5 \times 10^4$ | Frames collected before the first learning step |
| Epsilon decay frames $N_\epsilon$ | $1 \times 10^6$ | Exploration annealing horizon |
| Final epsilon $\epsilon_{\text{final}}$ | $0.1$ | Persistent exploratory noise floor |
| Frame skip $k$ | $4$ | Action repeat interval |

---

## 6. Evaluation Protocol

### 6.1 Standard Evaluation

Model performance is assessed over $N_{\text{eval}} = 30$ episodes with an $\epsilon = 0.05$ exploration rate (mirroring the paper's protocol). Each episode is capped at $18,\!000$ frames (approximately 5 minutes of gameplay at 60 fps), with evaluation beginning after a stochastic number of initial no-operation actions (1--30). The reported metrics are the mean and standard deviation of the undiscounted episode returns.

### 6.2 Normalized Score and Paper Alignment

To enable cross-game comparison independent of absolute reward scale, we compute a **normalized score** that maps raw episode returns onto a scale where $0\%$ corresponds to a naive random agent and $100\%$ corresponds to a professional human game tester:

$$\text{score}_{\text{normalized}} = 100 \times \frac{\text{score}_{\text{agent}} - \text{score}_{\text{random}}}{\text{score}_{\text{human}} - \text{score}_{\text{random}}}$$

The random and human baselines are taken from Table 2 of Mnih et al. (2015). The script `verify_model_paper.py` implements this protocol for 19 Atari games with published reference metrics, categorizing reproduction fidelity as follows:

- $|\Delta_{\text{normalized}}| \leq 10\%$: Excellent reproduction
- $10\% < |\Delta_{\text{normalized}}| \leq 30\%$: Acceptable reproduction
- $|\Delta_{\text{normalized}}| > 30\%$: Not adequately reproduced

---

## 7. Supported Atari 2600 Games

The training pipeline supports all 49 games from the original DQN evaluation suite:

`Alien`, `Amidar`, `Assault`, `Asterix`, `Asteroids`, `Atlantis`, `BankHeist`, `BattleZone`, `BeamRider`, `Bowling`, `Boxing`, `Breakout`, `Centipede`, `ChopperCommand`, `CrazyClimber`, `DemonAttack`, `DoubleDunk`, `Enduro`, `FishingDerby`, `Freeway`, `Frostbite`, `Gopher`, `Gravitar`, `Hero`, `IceHockey`, `JamesBond`, `Kangaroo`, `Krull`, `KungFuMaster`, `MontezumaRevenge`, `MsPacman`, `NameThisGame`, `Phoenix`, `Pitfall`, `Pong`, `PrivateEye`, `Qbert`, `Riverraid`, `RoadRunner`, `Robotank`, `Seaquest`, `SpaceInvaders`, `StarGunner`, `Tennis`, `TimePilot`, `Tutankham`, `UpNDown`, `Venture`, `VideoPinball`, `WizardOfWor`, `Zaxxon`.

Among these, 19 games (Breakout, Pong, Space Invaders, Seaquest, River Raid, Assault, Asterix, Battle Zone, Beam Rider, Boxing, Centipede, Chopper Command, Crazy Climber, Demon Attack, Freeway, Kung-Fu Master, Q*bert, Road Runner, Star Gunner) have published human and random baseline scores, enabling formal normalized-score comparison.

---

## 8. Installation and Environment Setup

### 8.1 System Requirements

The development environment used for this project consists of:

- **Operating System**: Ubuntu 20.04 LTS (or compatible Linux distribution)
- **CPU**: Intel Core i7-14700HX (20 cores, 28 threads)
- **RAM**: 64 GB DDR5
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, CUDA 12.x)
- **Python**: 3.10

### 8.2 Environment Configuration

```bash
# Create and activate a conda environment
conda create -n nature2 python=3.10 -y
conda activate nature2

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install Gymnasium with Atari support and ALE bindings
pip install gymnasium[atari,accept-rom-license]==0.28.1
pip install ale-py==0.8.1

# Install auxiliary dependencies
pip install opencv-python numpy tqdm tensorboard
```

A convenience script, `test.sh`, automates the above environment setup steps.

### 8.3 Environment Verification

```bash
python test.py
```

This script initializes an ALE Breakout environment and executes a single random step, verifying that all dependencies are correctly installed and the ROM files are properly registered.

---

## 9. Usage

### 9.1 Single-Game Training

Train a DQN agent on a specified Atari game for 50 million frames:

```bash
python train.py BreakoutNoFrameskip-v4
```

The trained model weights are saved to `models/dqn_breakout.pth`, with intermediate checkpoints written every 1 million frames. TensorBoard logs are recorded under `runs/BreakoutNoFrameskip-v4/` and may be visualized with:

```bash
tensorboard --logdir runs/
```

### 9.2 Multi-Game Training

To train agents sequentially across all 49 Atari games:

```bash
python train_all.py
```

Games whose ROMs are not available are gracefully skipped with a diagnostic message.

### 9.3 Evaluation

Evaluate a trained model using the paper's standardized protocol (30 episodes, $\epsilon = 0.05$):

```bash
python evaluate.py models/dqn_breakout.pth BreakoutNoFrameskip-v4
```

Output includes the mean episode return and standard deviation.

### 9.4 Paper-Aligned Verification

Conduct a formal comparison against the published DQN results using normalized scores:

```bash
python verify_model_paper.py models/dqn_breakout.pth --game BreakoutNoFrameskip-v4
```

Optional arguments include `--render` for visual inspection, `--episodes` to modify the evaluation count, and `--max-frames` to adjust the per-episode frame cap.

---

## 10. Repository Structure

```
nature2/
├── model.py                  # DQN convolutional neural network definition
├── agent.py                  # DQNAgent (replay buffer, target network, RMSProp, Huber loss)
├── utils.py                  # Atari frame preprocessing and frame stacking
├── train.py                  # Single-game training loop (50M frames, frame skip = 4)
├── train_all.py              # Sequential multi-game training driver (49 Atari games)
├── evaluate.py               # Standardized evaluation (30 episodes, epsilon = 0.05)
├── verify_model_paper.py     # Paper-aligned verification with normalized score comparison
├── test.py                   # ALE environment sanity check
├── register_roms.py          # Atari ROM registration utility
├── test.sh                   # Automated environment setup script
├── demo                      # Reference usage commands
├── runs/                     # TensorBoard event log directory (auto-generated)
├── models/                   # Trained model weight storage (auto-generated)
├── LICENSE                   # MIT License
└── README.md                 # This document
```

---

## 11. Key Design Decisions and Implementation Notes

**Memory-Efficient Replay Buffer.** Rather than using Python `deque` objects with `namedtuple` wrapping, the replay buffer is implemented as preallocated NumPy arrays of fixed dtype (`float32`, `int64`, `bool_`). This design eliminates per-element object overhead, reducing RAM consumption by approximately an order of magnitude relative to naive Python container approaches, and enables the full 1-million-transition capacity to fit comfortably within 64 GB of system memory.

**Hard Target Network Update.** In contrast to Polyak-averaging (soft update) approaches popularized by subsequent DRL algorithms, this implementation adheres to the paper's hard update scheme: the target network parameters $\theta^-$ are replaced wholesale by the online network parameters $\theta$ every $C_{\text{target}}$ optimization steps. This creates a piecewise-constant target function, providing a stable learning signal for the duration of each update interval.

**Reward Clipping.** All rewards are clipped to the set $\{-1, 0, +1\}$ via the sign function. This normalization across games with vastly different reward scales (from single-digit scores in Pong to hundred-thousand-point scores in Crazy Climber) allows a single set of hyperparameters to be applied uniformly across the entire Atari suite without per-game adaptation.

---

## 12. Limitations and Future Work

The current implementation replicates the single-GPU, single-game training paradigm of the original work. Several extensions warrant consideration:

- **Double DQN** (van Hasselt et al., 2016): Decouple action selection from action evaluation to mitigate Q-value overestimation bias.
- **Dueling Network Architecture** (Wang et al., 2016): Separate the state-value and advantage streams within the network to improve policy evaluation in states where action choice has negligible impact.
- **Prioritized Experience Replay** (Schaul et al., 2016): Sample transitions with probability proportional to TD error magnitude, accelerating learning on salient experiences.
- **Distributed Training** (Nair et al., 2015; Horgan et al., 2018): Parallelize data collection across multiple actor processes with a shared replay buffer to reduce wall-clock training time.

Additionally, the games `MontezumaRevenge` and `Pitfall` --- which require long-term planning and exploration in sparse-reward settings --- remain challenging for the vanilla DQN algorithm and represent open problems in deep reinforcement learning.

---

## 13. License

This project is distributed under the MIT License. See `LICENSE` for the full text.

Copyright (c) 2025 Escherichia.

---

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540):529--533.

2. Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. (2013). The Arcade Learning Environment: An evaluation platform for general agents. *Journal of Artificial Intelligence Research*, 47:253--279.

3. van Hasselt, H., Guez, A., and Silver, D. (2016). Deep reinforcement learning with double Q-learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 30.

4. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In *Proceedings of the International Conference on Machine Learning (ICML)*, pages 1995--2003.

5. Schaul, T., Quan, J., Antonoglou, I., and Silver, D. (2016). Prioritized experience replay. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

6. Tieleman, T. and Hinton, G. (2012). Lecture 6.5---RMSProp: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*.

---

*Project initiated: 25 October 2025. Last revision: 20 June 2026.*
