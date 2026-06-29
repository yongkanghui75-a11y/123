# rsl_rl 代码框架学习笔记

## 1. rsl_rl 是什么

`rsl_rl` 是一个强化学习算法库。

在你现在的项目里，它和 `legged_gym` 配合使用：

```text
legged_gym 负责机器人环境
rsl_rl 负责强化学习算法训练
```

也就是说：

```text
legged_gym:
负责 observation、action、reward、reset、Isaac Gym 仿真

rsl_rl:
负责 actor、critic、rollout、return、advantage、PPO update、模型保存
```

一句话：

```text
legged_gym 是训练场；
rsl_rl 是训练算法；
PPO 是 rsl_rl 里面目前最重要的算法。
```

---

## 2. rsl_rl 最外层目录结构

你的路径大概是：

```bash
~/rl_ws/rsl_rl
```

核心结构大概是：

```text
rsl_rl/
├── rsl_rl/
│   ├── algorithms/
│   ├── modules/
│   ├── runners/
│   ├── storage/
│   └── utils/
├── setup.py
└── README.md
```

主要含义：

```text
algorithms/   强化学习算法主体，比如 PPO
modules/      神经网络模块，比如 ActorCritic
runners/      训练总调度器
storage/      rollout 数据存储
utils/        辅助工具
setup.py      安装配置文件
README.md     项目说明
```

你现在最需要关注这几个目录：

```text
runners/
algorithms/
modules/
storage/
```

---

## 3. rsl_rl 和 legged_gym 的关系

你平时运行训练命令：

```bash
python legged_gym/scripts/train.py \
  --task=anymal_c_flat \
  --headless \
  --num_envs=256 \
  --max_iterations=1000
```

表面上运行的是：

```text
legged_gym/scripts/train.py
```

但内部会调用 `rsl_rl`。

完整关系是：

```text
train.py
↓
task_registry.make_env()
↓
创建 legged_gym 环境
↓
task_registry.make_alg_runner()
↓
创建 rsl_rl 的 OnPolicyRunner
↓
OnPolicyRunner.learn()
↓
PPO 开始训练
```

所以：

```text
legged_gym 提供环境；
rsl_rl 提供训练器和算法。
```

---

## 4. runners：训练总调度器

路径：

```text
rsl_rl/rsl_rl/runners/
```

最重要文件：

```text
on_policy_runner.py
```

---

### 4.1 on_policy_runner.py

`OnPolicyRunner` 是训练总指挥。

它负责把环境、算法、网络、数据仓库连接起来。

它主要做：

```text
创建 ActorCritic 网络
创建 PPO 算法对象
创建 RolloutStorage
调用 env.step()
调用 PPO.act()
调用 PPO.update()
保存模型 checkpoint
打印训练日志
记录 TensorBoard 数据
```

核心函数是：

```python
learn()
```

理解：

```text
OnPolicyRunner = rsl_rl 的训练总调度器
```

它不负责具体 reward 怎么算。

reward 在 `legged_gym` 里算。

它也不负责 action 怎么变 torque。

action 变 torque 也在 `legged_gym` 里。

它主要负责：

```text
让算法和环境循环起来。
```

---

## 5. OnPolicyRunner.learn()：训练主循环

`learn()` 是整个强化学习训练的大循环。

大概逻辑：

```python
for it in range(num_learning_iterations):

    for i in range(num_steps_per_env):
        actions = alg.act(obs, critic_obs)
        obs, rewards, dones, infos = env.step(actions)
        alg.process_env_step(rewards, dones, infos)

    alg.compute_returns(critic_obs)
    alg.update()
```

它做三件核心事：

```text
1. 收集 rollout 数据
2. 计算 return / advantage
3. 更新 actor / critic
```

完整流程：

```text
当前 observation
↓
PPO.act()
↓
得到 action
↓
env.step(action)
↓
得到 reward / done / next_obs
↓
process_env_step()
↓
存进 RolloutStorage
↓
收集够一段 rollout
↓
compute_returns()
↓
PPO.update()
↓
actor 和 critic 被更新
```

一句话：

```text
OnPolicyRunner.learn() = 采样 → 存数据 → 算优势 → 更新网络。
```

---

## 6. algorithms：强化学习算法主体

路径：

```text
rsl_rl/rsl_rl/algorithms/
```

最重要文件：

```text
ppo.py
```

---

### 6.1 ppo.py

`ppo.py` 是 PPO 算法主体。

它负责：

```text
让 actor 选 action
记录 action 的 log_prob
记录 critic 的 value
处理环境返回的 reward / done
计算 return 和 advantage
计算 PPO loss
更新 actor 和 critic
```

里面主要函数：

```text
act()
process_env_step()
compute_returns()
update()
```

理解：

```text
ppo.py = PPO 算法核心逻辑
```

---

### 6.2 PPO.act()

作用：

```text
根据当前 observation 选择 action，并记录训练需要的数据。
```

它会保存：

```text
observations
critic_observations
actions
values
actions_log_prob
action_mean
action_sigma
```

理解：

```text
PPO.act() = 选动作 + 保存动作发生前的数据。
```

---

### 6.3 PPO.process_env_step()

作用：

```text
处理环境执行 action 后返回的结果。
```

它主要保存：

```text
reward
done
timeout 信息
```

然后调用：

```text
storage.add_transitions()
```

理解：

```text
process_env_step() = 保存动作发生后的结果。
```

所以：

```text
PPO.act()
保存动作前的数据

process_env_step()
保存动作后的结果
```

两者合起来就是一条完整经验：

```text
obs
action
reward
done
value
log_prob
```

---

### 6.4 PPO.compute_returns()

作用：

```text
计算 return 和 advantage。
```

其中：

```text
return = 未来累计奖励
advantage = 实际结果比 critic 预测好多少
```

理解：

```text
compute_returns() = 计算后面训练 actor / critic 的信号。
```

---

### 6.5 PPO.update()

作用：

```text
真正更新神经网络。
```

它会：

```text
从 storage 取 mini-batch
重新计算 action log_prob
计算 ratio
计算 actor loss
计算 critic loss
计算 entropy
反向传播
optimizer.step()
```

理解：

```text
PPO.update() = 真正让 actor 和 critic 变强的地方。
```

---

## 7. modules：神经网络模块

路径：

```text
rsl_rl/rsl_rl/modules/
```

最重要文件：

```text
actor_critic.py
```

---

### 7.1 actor_critic.py

这个文件定义 actor 和 critic 网络。

核心类：

```text
ActorCritic
```

它里面包含：

```text
actor 网络
critic 网络
动作分布 distribution
动作标准差 std
```

理解：

```text
ActorCritic = actor + critic + 动作分布。
```

---

## 8. actor 是什么

actor 是决策网络。

它负责：

```text
输入 observation
↓
输出 action mean
```

在 `anymal_c_flat` 里：

```text
observation = 48 维
action = 12 维
```

因为 ANYmal C 有 12 个关节。

actor 的作用：

```text
根据机器人当前状态，决定机器人应该怎么动。
```

理解：

```text
actor = 负责做动作的大脑。
```

---

## 9. critic 是什么

critic 是评估网络。

它负责：

```text
输入 critic_observation
↓
输出 value
```

value 表示：

```text
从当前状态开始，未来大概能拿多少 reward。
```

critic 不直接控制机器人。

它的作用是帮助 PPO 判断：

```text
当前 action 比预期好，还是比预期差。
```

理解：

```text
critic = 给状态打分的人。
```

---

## 10. ActorCritic.act()

训练时，actor 不会直接输出最终 action。

它先输出动作均值 `mean`。

然后用 `mean` 和 `std` 构造正态分布。

最后从分布里采样 action。

流程：

```text
observation
↓
actor 网络
↓
action mean
↓
Normal(mean, std)
↓
sample()
↓
action
```

核心代码大概是：

```python
self.update_distribution(observations)
return self.distribution.sample()
```

理解：

```text
训练时 action 是采样出来的，不是固定死的。
```

为什么要采样？

```text
因为强化学习需要探索。
```

---

## 11. update_distribution()

核心逻辑：

```python
mean = self.actor(observations)
self.distribution = Normal(mean, mean * 0. + self.std)
```

含义：

```text
mean = actor 认为比较合适的动作中心
std = 探索幅度
Normal(mean, std) = 动作分布
sample = 实际执行的动作
```

理解：

```text
update_distribution() = 根据当前 observation 构造动作分布。
```

---

## 12. evaluate()

`evaluate()` 是 critic 的前向计算。

流程：

```text
critic_observations
↓
critic 网络
↓
value
```

理解：

```text
evaluate() = critic 给当前状态打分。
```

---

## 13. storage：rollout 数据仓库

路径：

```text
rsl_rl/rsl_rl/storage/
```

最重要文件：

```text
rollout_storage.py
```

---

### 13.1 rollout_storage.py

`RolloutStorage` 是 PPO 的临时经验仓库。

它负责存一段 rollout 数据。

比如：

```text
num_steps_per_env = 24
num_envs = 256
```

那么一次 rollout 有：

```text
24 × 256 = 6144 条样本
```

它保存的数据包括：

```text
observations
critic_observations
actions
rewards
dones
values
actions_log_prob
mu
sigma
returns
advantages
```

理解：

```text
RolloutStorage = PPO 更新前临时存经历的地方。
```

---

### 13.2 add_transitions()

每执行一步环境，都会把 transition 存进 storage。

流程：

```text
PPO.act()
↓
得到 obs / action / value / log_prob
↓
env.step()
↓
得到 reward / done
↓
process_env_step()
↓
storage.add_transitions()
```

理解：

```text
add_transitions() = 把一整步训练数据存进仓库。
```

如果有 256 个并行环境，那么一次 add 存的是：

```text
256 个机器人当前这一步的数据。
```

---

### 13.3 compute_returns()

收集完一段 rollout 后，需要计算：

```text
returns
advantages
```

核心逻辑：

```text
delta = reward + gamma * next_value - value
advantage = delta + gamma * lam * next_advantage
return = advantage + value
```

其中：

```text
gamma = 折扣因子，决定未来 reward 的重要程度
lam = GAE 参数，让 advantage 估计更稳定
```

理解：

```text
RolloutStorage.compute_returns() = 根据 reward 和 value 计算训练信号。
```

---

### 13.4 mini_batch_generator()

PPO 更新时，不会一次性使用全部 rollout 数据。

它会把数据打乱并切成 mini-batch。

原始数据形状：

```text
[num_steps, num_envs, ...]
```

展开后：

```text
[num_steps * num_envs, ...]
```

例如：

```text
24 steps × 256 envs = 6144 samples
```

然后分成多个 mini-batch。

理解：

```text
mini_batch_generator() = 把 rollout 数据打乱分批交给 PPO.update()。
```

---

## 14. utils：辅助工具

路径：

```text
rsl_rl/rsl_rl/utils/
```

这个目录里一般放辅助函数。

它不是主线核心，但会辅助训练过程。

可能包含：

```text
日志工具
模型导出工具
辅助函数
配置处理工具
```

你前期不用重点看。

等你学会主线后，再看这里。

---

## 15. setup.py：安装配置

路径：

```text
rsl_rl/setup.py
```

这个文件用于把 `rsl_rl` 安装成 Python 包。

你之前安装时可能执行过：

```bash
pip install -e .
```

`-e` 的意思是 editable mode。

也就是：

```text
以开发模式安装。
```

这样你修改 `rsl_rl` 里的源码后，不需要重新安装，Python 就能直接用到修改后的代码。

理解：

```text
setup.py = 让 Python 能找到并导入 rsl_rl。
```

---

## 16. rsl_rl 的完整代码流

当你运行：

```bash
python legged_gym/scripts/train.py \
  --task=anymal_c_flat \
  --headless \
  --num_envs=256 \
  --max_iterations=1000
```

内部和 `rsl_rl` 相关的流程是：

```text
legged_gym train.py
↓
task_registry.make_alg_runner()
↓
创建 OnPolicyRunner
↓
OnPolicyRunner 创建 ActorCritic
↓
OnPolicyRunner 创建 PPO
↓
OnPolicyRunner 创建 RolloutStorage
↓
OnPolicyRunner.learn()
↓
PPO.act()
↓
ActorCritic.act()
↓
actor 输出动作分布
↓
采样 action
↓
legged_gym env.step(action)
↓
环境返回 reward / done / next_obs
↓
PPO.process_env_step()
↓
RolloutStorage.add_transitions()
↓
收集一段 rollout
↓
PPO.compute_returns()
↓
RolloutStorage.compute_returns()
↓
PPO.update()
↓
mini_batch_generator()
↓
计算 loss
↓
optimizer.step()
↓
更新 actor / critic
↓
保存 model_xxx.pt
```

一句话：

```text
rsl_rl 负责让 actor 试动作、存经历、算好坏、更新网络。
```

---

## 17. rsl_rl 和 legged_gym 的分工

### legged_gym 负责：

```text
创建机器人
加载 URDF
推进 Isaac Gym 仿真
把 action 变成 torque
计算 reward
生成 observation
判断 reset
```

### rsl_rl 负责：

```text
根据 observation 选 action
记录 rollout 数据
计算 return
计算 advantage
计算 PPO loss
更新 actor / critic
保存模型
```

连接关系：

```text
rsl_rl 输出 action
↓
legged_gym 执行 action
↓
legged_gym 返回 obs / reward / done
↓
rsl_rl 用这些数据更新策略
```

---

## 18. rsl_rl 各目录一句话总结

```text
runners/
训练总调度器，负责把环境、算法、网络、数据连接起来。

algorithms/
算法主体，目前最重要的是 PPO。

modules/
神经网络模块，主要是 ActorCritic。

storage/
rollout 数据仓库，负责存训练经历。

utils/
辅助工具。

setup.py
安装配置，让 Python 能导入 rsl_rl。
```

---

## 19. rsl_rl 各核心文件一句话总结

```text
on_policy_runner.py
训练总指挥，负责 learn() 主循环。

ppo.py
PPO 算法主体，负责 act、compute_returns、update。

actor_critic.py
actor 和 critic 网络定义。

rollout_storage.py
存 rollout 数据，计算 returns 和 advantages。
```

---

## 20. 最重要的一句话

`rsl_rl` 的本质：

```text
用 actor 根据 observation 选 action，
把 action 交给 legged_gym 执行，
把 reward 和 done 存起来，
计算哪些动作好、哪些动作差，
然后用 PPO 更新 actor 和 critic。
```

压缩版：

```text
rsl_rl = 选动作 + 存经历 + 算优势 + 更新网络。
```

最终记忆：

```text
runners 是训练总指挥；
algorithms 是算法核心；
modules 是神经网络；
storage 是经验仓库；
legged_gym 给环境结果；
rsl_rl 负责学习。
```

