# rsl_rl PPO 主线完整学习笔记

## 1. 当前学习目标

这一部分主要看懂 `rsl_rl` 里 PPO 是怎么训练策略网络的。

环境部分已经知道：

```text
obs → actor → action → env.step(action) → reward / done
```

PPO 主线要补上：

```text
reward / done / value / log_prob
↓
存进 rollout storage
↓
计算 return 和 advantage
↓
切成 mini-batch
↓
PPO.update()
↓
更新 actor 和 critic
```

最终完整闭环是：

```text
actor 负责输出 action
critic 负责估计 value
storage 负责保存 rollout 数据
compute_returns() 负责计算 advantage
mini_batch_generator() 负责切训练数据
update() 负责更新网络参数
```

---

## 2. PPO 总入口：`OnPolicyRunner.learn()`

文件位置：

```text
rsl_rl/rsl_rl/runners/on_policy_runner.py
```

核心流程：

```text
获取 obs
↓
调用 PPO.act(obs, critic_obs)
↓
得到 actions
↓
env.step(actions)
↓
得到新 obs、reward、done、infos
↓
process_env_step()
↓
存入 rollout storage
↓
重复 num_steps_per_env 步
↓
compute_returns()
↓
update()
```

一句话：

```text
learn() = 采样 rollout + 用 PPO 更新网络。
```

其中 rollout 可以理解为：

```text
让当前策略控制机器人跑一小段时间，
把这段交互数据全部记录下来。
```

---

## 3. ActorCritic.act()：actor 生成 action

文件位置：

```text
rsl_rl/rsl_rl/modules/actor_critic.py
```

代码：

```python
def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()
```

作用：

```text
先根据 observation 生成动作分布，
再从动作分布里随机采样 action。
```

流程：

```text
observations
↓
update_distribution()
↓
得到动作分布 Normal(mean, std)
↓
sample()
↓
得到 action
```

训练时使用 `sample()`，是为了探索。

---

## 4. update_distribution()：生成动作分布

文件位置：

```text
rsl_rl/rsl_rl/modules/actor_critic.py
```

代码：

```python
def update_distribution(self, observations):
    mean = self.actor(observations)
    self.distribution = Normal(mean, mean*0. + self.std)
```

作用：

```text
把 observation 输入 actor 网络，
得到动作均值 mean，
再用 mean 和 std 构造正态分布。
```

核心理解：

```text
mean
策略认为比较合理的动作中心。

std
动作探索的随机幅度。

Normal(mean, std)
动作分布。

sample()
从动作分布里采样真正执行的 action。
```

对于 ANYmal C：

```text
如果有 256 个并行机器人，每个机器人 12 个 action，
那么 mean 大概是 [256, 12]。
```

---

## 5. evaluate()：critic 估计 value

文件位置：

```text
rsl_rl/rsl_rl/modules/actor_critic.py
```

代码：

```python
def evaluate(self, critic_observations, **kwargs):
    value = self.critic(critic_observations)
    return value
```

作用：

```text
把 critic_observations 输入 critic 网络，
输出当前状态的 value。
```

`value` 的含义：

```text
不是当前这一步的 reward，
而是 critic 估计：
从当前状态开始，未来大概还能拿多少总奖励。
```

actor 和 critic 的区别：

```text
actor：
看当前状态，决定怎么动。

critic：
看当前状态，判断这个状态大概有多好。
```

对应代码：

```python
# actor
mean = self.actor(observations)

# critic
value = self.critic(critic_observations)
```

---

## 6. PPO.act()：动作发生前保存数据

文件位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

核心代码：

```python
def act(self, obs, critic_obs):
    if self.actor_critic.is_recurrent:
        self.transition.hidden_states = self.actor_critic.get_hidden_states()

    self.transition.actions = self.actor_critic.act(obs).detach()
    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()

    self.transition.observations = obs
    self.transition.critic_observations = critic_obs
    return self.transition.actions
```

作用：

```text
PPO.act()
不只是输出 action，
还会把 PPO 后面 update 需要的数据提前保存起来。
```

它保存：

```text
observations
当前 actor 看到的 obs。

critic_observations
当前 critic 看到的 obs。

actions
actor 采样出来的动作。

values
critic 对当前状态的估值。

actions_log_prob
旧策略下这个 action 的 log probability。

action_mean
动作分布均值。

action_sigma
动作分布标准差。
```

这里保存的是动作发生前的数据。

后面 `env.step(actions)` 之后，才会得到 reward 和 done。

---

## 7. get_actions_log_prob()：计算 action 的 log 概率

文件位置：

```text
rsl_rl/rsl_rl/modules/actor_critic.py
```

代码：

```python
def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)
```

作用：

```text
计算某个 action 在当前动作分布下的 log probability。
```

为什么要 `.sum(dim=-1)`？

因为 action 是多维的，比如 ANYmal C 有 12 个 action。

`distribution.log_prob(actions)` 会得到：

```text
每个关节动作各自的 log_prob
```

形状大概是：

```text
[num_envs, 12]
```

PPO 需要的是整个 action 向量的 log_prob，所以要把 12 个维度加起来：

```text
12 个关节 action 的 log_prob
↓
sum(dim=-1)
↓
整个 action 的 log_prob
```

这个函数后面用于 PPO 的核心 ratio：

```text
ratio = 新策略概率 / 旧策略概率
```

---

## 8. process_env_step()：动作发生后补 reward 和 done

文件位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

代码：

```python
def process_env_step(self, rewards, dones, infos):
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if 'time_outs' in infos:
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
        )

    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)
```

作用：

```text
把 env.step() 返回的 rewards 和 dones，
补进刚才保存的 transition 里，
然后把完整 transition 存进 rollout storage。
```

流程：

```text
PPO.act()
保存 obs、action、value、log_prob
↓
env.step(action)
得到 reward、done
↓
process_env_step()
补 reward、done
↓
storage.add_transitions()
正式存入 storage
```

### timeout 特殊处理

代码：

```python
if 'time_outs' in infos:
    self.transition.rewards += self.gamma * ...
```

含义：

```text
如果 episode 是因为正常超时结束，
不要把它当成摔倒失败。
```

所以会用 critic 的 value 补一点未来价值。

大白话：

```text
机器人不是摔倒了，只是时间到了。
不能因为 done=True 就认为未来价值为 0。
```

---

## 9. RolloutStorage.add_transitions()：把数据存进表格

文件位置：

```text
rsl_rl/rsl_rl/storage/rollout_storage.py
```

代码：

```python
def add_transitions(self, transition: Transition):
    if self.step >= self.num_transitions_per_env:
        raise AssertionError("Rollout buffer overflow")
    self.observations[self.step].copy_(transition.observations)
    if self.privileged_observations is not None:
        self.privileged_observations[self.step].copy_(transition.critic_observations)
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    self.mu[self.step].copy_(transition.action_mean)
    self.sigma[self.step].copy_(transition.action_sigma)
    self._save_hidden_states(transition.hidden_states)
    self.step += 1
```

作用：

```text
把当前这一步的训练数据，按时间顺序存进 rollout storage。
```

RolloutStorage 可以理解为 PPO 的训练数据表格。

它每一步存：

```text
observations
critic_observations / privileged_observations
actions
rewards
dones
values
actions_log_prob
mu
sigma
hidden_states
```

如果 `num_envs=256`，每次 `add_transitions()` 存的是：

```text
256 个机器人在同一个时间步的数据。
```

如果 `num_steps_per_env=24`，那么一次 rollout 大概有：

```text
24 × 256 = 6144 条 transition
```

这就是 GPU 并行训练快的原因。

---

## 10. PPO.compute_returns()：计算 returns 前的入口

文件位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

代码：

```python
def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

作用：

```text
先用 critic 估计 rollout 最后一个状态的 value，
然后交给 storage 计算 returns 和 advantages。
```

为什么需要 `last_values`？

因为 rollout 只是一小段，不是完整 episode。

比如只采样 24 步：

```text
step 0 → step 1 → ... → step 23
```

第 23 步之后机器人可能还没有结束，所以要用 critic 估计：

```text
第 24 步之后大概还能拿多少未来 reward。
```

这就是 `last_values` 的作用。

---

## 11. RolloutStorage.compute_returns()：计算 return 和 advantage

文件位置：

```text
rsl_rl/rsl_rl/storage/rollout_storage.py
```

代码：

```python
def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = self.values[step + 1]
        next_is_not_terminal = 1.0 - self.dones[step].float()
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        self.returns[step] = advantage + self.values[step]

    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

作用：

```text
用 reward、done、value 和 last_values，
倒着计算每一步的 return 和 advantage。
```

### return 是什么

```text
return = 从当前这一步开始，未来一段时间大概能拿到的总奖励。
```

### advantage 是什么

```text
advantage = 实际结果比 critic 原本预期好多少。
```

大概关系：

```text
advantage = return - value
```

如果：

```text
advantage > 0
```

说明这个 action 比预期好，应该加强。

如果：

```text
advantage < 0
```

说明这个 action 比预期差，应该削弱。

### 为什么倒着算

代码：

```python
for step in reversed(range(self.num_transitions_per_env)):
```

因为当前这一步好不好，要看后面发生了什么。

所以要从最后一步往前传：

```text
step 23 → step 22 → step 21 → ... → step 0
```

### delta 是什么

代码：

```python
delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
```

含义：

```text
delta =
当前 reward
+
下一步未来价值
-
当前 critic 估值
```

它表示：

```text
实际结果比 critic 当前预测好多少。
```

### advantage 递推

代码：

```python
advantage = delta + next_is_not_terminal * gamma * lam * advantage
```

含义：

```text
当前 action 好不好，
不仅看当前一步，
还看后面几步带来的连续影响。
```

如果 done=True：

```text
next_is_not_terminal = 0
```

未来价值会被切断，防止把下一个 episode 的数据算进来。

### 标准化 advantage

代码：

```python
self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

作用：

```text
让 advantage 的均值接近 0，标准差接近 1，
训练更稳定。
```

---

## 12. mini_batch_generator()：把 rollout 数据切成 mini-batch

文件位置：

```text
rsl_rl/rsl_rl/storage/rollout_storage.py
```

代码：

```python
def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)
    if self.privileged_observations is not None:
        critic_observations = self.privileged_observations.flatten(0, 1)
    else:
        critic_observations = observations

    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_mu = self.mu.flatten(0, 1)
    old_sigma = self.sigma.flatten(0, 1)

    for epoch in range(num_epochs):
        for i in range(num_mini_batches):
            start = i*mini_batch_size
            end = (i+1)*mini_batch_size
            batch_idx = indices[start:end]

            obs_batch = observations[batch_idx]
            critic_observations_batch = critic_observations[batch_idx]
            actions_batch = actions[batch_idx]
            target_values_batch = values[batch_idx]
            returns_batch = returns[batch_idx]
            old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
            advantages_batch = advantages[batch_idx]
            old_mu_batch = old_mu[batch_idx]
            old_sigma_batch = old_sigma[batch_idx]

            yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                  old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
```

作用：

```text
把 rollout storage 里的大批训练数据打乱，
再切成 mini-batch，
一批一批喂给 PPO.update()。
```

### batch_size

代码：

```python
batch_size = self.num_envs * self.num_transitions_per_env
```

含义：

```text
一轮 rollout 一共有多少条 transition。
```

例如：

```text
num_envs = 256
num_transitions_per_env = 24

batch_size = 256 × 24 = 6144
```

### mini_batch_size

代码：

```python
mini_batch_size = batch_size // num_mini_batches
```

如果：

```text
batch_size = 6144
num_mini_batches = 4
```

那么：

```text
mini_batch_size = 1536
```

意思是：

```text
每个 mini-batch 有 1536 条 transition。
```

### 打乱索引

代码：

```python
indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
```

作用：

```text
把样本顺序打乱，
避免训练时总按时间顺序喂数据。
```

### flatten：合并时间维和环境维

代码：

```python
observations = self.observations.flatten(0, 1)
```

storage 原来的形状大概是：

```text
[num_steps, num_envs, obs_dim]
```

例如：

```text
[24, 256, 48]
```

`flatten(0, 1)` 后变成：

```text
[24 × 256, 48]
=
[6144, 48]
```

意思是：

```text
不再区分第几步、哪个机器人，
统一看成 6144 条训练样本。
```

其他数据也一样：

```text
actions: [24, 256, 12] → [6144, 12]
advantages: [24, 256, 1] → [6144, 1]
returns: [24, 256, 1] → [6144, 1]
```

### num_epochs

代码：

```python
for epoch in range(num_epochs):
```

含义：

```text
同一批 rollout 数据，会重复训练 num_epochs 轮。
```

比如：

```text
num_epochs = 5
num_mini_batches = 4
```

那么 update 总共训练：

```text
5 × 4 = 20 个 mini-batch
```

### yield

代码：

```python
yield obs_batch, critic_observations_batch, actions_batch, ...
```

`yield` 可以理解为：

```text
每次吐出一个 mini-batch，
交给 PPO.update()；
下一次循环时，再吐出下一个 mini-batch。
```

在 `PPO.update()` 里：

```python
for obs_batch, critic_obs_batch, actions_batch, ... in generator:
```

每循环一次，就会从这里拿到一个 mini-batch。

最后的：

```python
(None, None), None
```

对应：

```text
hid_states_batch
masks_batch
```

因为普通 MLP 策略没有 RNN hidden states，所以这里返回 None。

---

## 13. PPO.update()：真正更新 actor 和 critic

文件位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

作用：

```text
用 rollout storage 里收集到的数据，
更新 actor 和 critic 网络参数。
```

整体流程：

```text
从 storage 取 mini-batch
↓
用当前 actor 重新计算 action 的 log_prob
↓
用当前 critic 重新计算 value
↓
计算 ratio = 新策略概率 / 旧策略概率
↓
用 advantage 计算 actor loss
↓
用 return 计算 critic loss
↓
加上 entropy 探索项
↓
loss.backward()
↓
optimizer.step()
↓
actor 和 critic 参数更新
```

---

## 14. update() 第一步：取 mini-batch

代码：

```python
if self.actor_critic.is_recurrent:
    generator = self.storage.reccurent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
    )
else:
    generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
    )
```

作用：

```text
把 rollout storage 里的数据切成一小批一小批 mini-batch。
```

普通 MLP 策略主要看：

```python
self.storage.mini_batch_generator(...)
```

mini-batch 里包含：

```text
obs_batch
critic_obs_batch
actions_batch
target_values_batch
advantages_batch
returns_batch
old_actions_log_prob_batch
old_mu_batch
old_sigma_batch
hid_states_batch
masks_batch
```

其中最重要的是：

```text
obs_batch
当时 actor 看到的 obs。

actions_batch
旧策略当时执行过的 action。

advantages_batch
这个 action 比预期好还是差。

returns_batch
critic 要学习的目标。

old_actions_log_prob_batch
旧策略下 action 的 log_prob。
```

---

## 15. update() 第二步：用当前策略重新计算

代码：

```python
self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
mu_batch = self.actor_critic.action_mean
sigma_batch = self.actor_critic.action_std
entropy_batch = self.actor_critic.entropy
```

作用：

```text
用当前 actor 和 critic，
重新计算同一批数据下的 log_prob、value、mean、std、entropy。
```

关键点：

```text
actions_batch 是旧策略当时执行过的 action。

现在重新计算 actions_log_prob_batch，
是在问：

新策略现在面对同样的 obs，
还愿不愿意做当时那个 action？
```

---

## 16. KL 自适应学习率

代码中 KL 部分作用：

```text
检查新策略和旧策略差得远不远。
```

如果 KL 太大：

```text
说明新策略变化太猛，降低学习率。
```

如果 KL 太小：

```text
说明新策略变化太慢，提高学习率。
```

这部分可以先知道作用，不需要第一轮深抠公式。

---

## 17. PPO ratio：新策略概率 / 旧策略概率

代码：

```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
```

含义：

```text
ratio = 新策略概率 / 旧策略概率
```

因为用的是 log probability：

```text
exp(new_log_prob - old_log_prob)
=
new_prob / old_prob
```

理解：

```text
ratio > 1
说明新策略比旧策略更愿意做这个 action。

ratio < 1
说明新策略比旧策略更不愿意做这个 action。
```

---

## 18. surrogate loss：actor 的损失

代码：

```python
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

作用：

```text
根据 advantage 决定旧 action 是该加强还是削弱，
同时用 clip 限制策略变化幅度。
```

理解：

```text
advantage > 0
说明 action 比预期好，新策略应该更愿意做它。

advantage < 0
说明 action 比预期差，新策略应该更不愿意做它。
```

为什么有负号？

```text
因为优化器默认最小化 loss。
PPO 本来想最大化好动作概率，
所以代码写成负的目标。
```

为什么要 clip？

如果：

```text
clip_param = 0.2
```

那么 ratio 会被限制在：

```text
[0.8, 1.2]
```

含义：

```text
策略可以变好，
但不能一下子变化太猛。
```

这就是 PPO 的核心思想。

---

## 19. value loss：critic 的损失

代码：

```python
value_loss = (returns_batch - value_batch).pow(2).mean()
```

或者带 clipped value loss 的版本。

作用：

```text
训练 critic，让 critic 的 value 更接近 returns。
```

也就是：

```text
critic 预测的 value
和
实际计算出来的 return
差得越远，value_loss 越大。
```

critic 学习目标：

```text
value_batch → returns_batch
```

---

## 20. entropy：鼓励探索

总 loss 中有：

```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
```

其中：

```text
entropy_batch
表示动作分布的随机程度。
```

entropy 越大：

```text
策略越随机，探索越强。
```

为什么是负号？

```text
优化器最小化 loss。
写成 -entropy，
等价于鼓励 entropy 变大。
```

所以总 loss 可以理解为：

```text
总 loss =
actor 损失
+
critic 损失
-
探索奖励
```

---

## 21. 梯度更新：网络参数真正变化

代码：

```python
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
self.optimizer.step()
```

含义：

```text
zero_grad()
清空旧梯度。

loss.backward()
根据 loss 计算参数应该怎么改。

clip_grad_norm_()
限制梯度大小，防止训练爆炸。

optimizer.step()
真正更新 actor 和 critic 参数。
```

最重要的一句：

```python
self.optimizer.step()
```

这句执行后：

```text
actor 和 critic 网络参数真的变了。
```

---

## 22. update() 收尾

代码：

```python
num_updates = self.num_learning_epochs * self.num_mini_batches
mean_value_loss /= num_updates
mean_surrogate_loss /= num_updates
self.storage.clear()

return mean_value_loss, mean_surrogate_loss
```

作用：

```text
计算平均 loss
清空 storage
返回日志信息
```

为什么清空 storage？

因为 PPO 是 on-policy 算法。

```text
当前 rollout 数据来自旧策略。
update 之后策略变了，
旧数据不能一直拿来用，
需要重新采集新 rollout。
```

---

## 23. PPO 主线完整闭环

现在 PPO 主线可以完整串起来：

```text
OnPolicyRunner.learn()
↓
获取 obs 和 critic_obs
↓
PPO.act()
↓
ActorCritic.act()
↓
update_distribution()
↓
actor 输出 mean
↓
Normal(mean, std)
↓
sample action
↓
critic evaluate() 输出 value
↓
get_actions_log_prob() 记录旧策略 log_prob
↓
env.step(action)
↓
环境返回 reward、done、infos
↓
process_env_step()
↓
补 reward 和 done
↓
RolloutStorage.add_transitions()
↓
存 obs、action、reward、done、value、log_prob、mean、std
↓
重复 num_steps_per_env 步
↓
PPO.compute_returns()
↓
critic 估最后状态 last_values
↓
RolloutStorage.compute_returns()
↓
计算 returns 和 advantages
↓
mini_batch_generator()
↓
flatten 时间维和环境维
↓
打乱样本顺序
↓
切成 mini-batch
↓
PPO.update()
↓
重新计算新策略 log_prob 和新 value
↓
ratio = 新策略概率 / 旧策略概率
↓
surrogate_loss 更新 actor
↓
value_loss 更新 critic
↓
entropy 保持探索
↓
optimizer.step()
↓
actor 和 critic 变得更好
```

---

## 24. 当前阶段最重要的一句话

```text
actor 根据 obs 输出动作分布并采样 action；
critic 根据 critic_obs 估计 value；
PPO 把 obs、action、reward、done、value、log_prob 存进 storage；
compute_returns() 用 reward 和 value 算 advantage；
mini_batch_generator() 把 rollout 数据切成 mini-batch；
update() 用 advantage 更新 actor，用 returns 更新 critic。
```

---

## 25. PPO 里几个关键词总结

### actor

```text
负责决定怎么动。
```

### critic

```text
负责估计当前状态未来能拿多少 reward。
```

### value

```text
critic 对当前状态的估值。
```

### return

```text
从当前这一步开始，未来大概能拿到的总奖励。
```

### advantage

```text
实际结果比 critic 预期好多少。
```

### log_prob

```text
某个 action 在当前策略下出现的 log 概率。
```

### ratio

```text
新策略选择这个 action 的概率 / 旧策略选择这个 action 的概率。
```

### surrogate_loss

```text
actor 的损失，用来决定 action 是该加强还是削弱。
```

### value_loss

```text
critic 的损失，用来让 value 更接近 return。
```

### entropy

```text
动作分布的随机程度，用来鼓励探索。
```

### clip

```text
限制策略更新幅度，防止新策略一步变化太猛。
```

### rollout storage

```text
PPO 的训练数据表格，保存 obs、action、reward、done、value、log_prob 等数据。
```

### mini-batch

```text
从 rollout 大表里切出来的一小批训练数据。
```

---

## 26. PPO 第一轮主线总结

PPO 的训练可以用一句话概括：

```text
先让当前策略控制机器人跑一段，
把每一步的 obs、action、reward、done、value、log_prob 存起来；
然后根据 reward 和 value 算出 advantage；
再把数据切成 mini-batch；
最后用 PPO loss 更新 actor 和 critic。
```

更直白地说：

```text
actor 先尝试动作；
环境给结果；
critic 估计这个状态值不值；
advantage 判断动作比预期好还是差；
PPO.update() 根据这个判断修改 actor 和 critic。
```

