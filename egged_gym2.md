# legged_gym 学习笔记更新：Commands、Reset、PPO 主线

## 1. Commands 是怎么来的

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

相关函数：

```python
_post_physics_step_callback()
_resample_commands()
```

### `_post_physics_step_callback()`

这个函数在每次物理仿真后、计算 reward 和 observation 前执行。

核心代码：

```python
env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
self._resample_commands(env_ids)
```

含义：

```text
每隔 resampling_time 秒，
找出需要换命令的机器人，
然后重新生成 commands。
```

对于 `anymal_c_flat`：

```python
resampling_time = 4.
heading_command = False
```

意思是：

```text
每 4 秒给机器人换一次目标速度。
不使用目标朝向模式，而是直接随机目标 yaw 角速度。
```

---

## 2. `_resample_commands()`：随机生成训练命令

核心作用：

```text
训练时需要的目标命令 commands，不是人手动输入的，
而是环境代码自动随机生成的。
```

commands 的含义：

```text
commands[:, 0]：目标前进/后退速度
commands[:, 1]：目标左右平移速度
commands[:, 2]：目标转向速度
```

它的作用不是直接探索 action，而是随机生成训练任务目标。

准确理解：

```text
command 随机 = 随机出题
action 探索 = 随机试答案
```

也就是说：

```text
_resample_commands()
随机生成“你要怎么走”

actor / PPO
探索“我该怎么动”
```

完整关系：

```text
_resample_commands()
随机生成目标速度 commands
↓
compute_observations()
把 commands 放进 observation，让策略知道目标
↓
actor 输出 action
↓
env.step(action)
机器人执行动作
↓
compute_reward()
检查实际速度是否跟上 commands
```

---

## 3. 为什么要随机生成 commands

如果 commands 永远固定，比如一直是：

```text
向前走 1.0 m/s
```

机器人最后可能只会这一种走法。

随机 commands 是为了让机器人学会多种任务：

```text
前进
后退
横移
左转
右转
站立
边走边转
```

所以：

```text
commands 解决“学什么”
actions 解决“怎么做”
reward 评价“做得好不好”
```

---

## 4. 终止判断：`check_termination()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

函数：

```python
check_termination()
```

作用：

```text
判断哪些机器人需要 reset。
```

主要有两种 reset 原因：

```text
1. 不该接触地面的身体部位接触了地面
   通常代表摔倒或碰撞失败

2. episode 时间太长，超过 max_episode_length
   这是正常超时结束，不算失败
```

核心变量：

```text
reset_buf
表示这个环境是否需要 reset

time_out_buf
表示是否因为超时而 reset
```

---

## 5. 终止奖励：`_reward_termination()`

代码：

```python
def _reward_termination(self):
    return self.reset_buf * ~self.time_out_buf
```

含义：

```text
只有因为摔倒/碰撞导致 reset，才触发 termination 惩罚。
如果只是正常超时结束，不给 termination 惩罚。
```

逻辑：

```text
摔倒：
reset_buf = True
time_out_buf = False
结果 = True，触发惩罚

超时：
reset_buf = True
time_out_buf = True
结果 = False，不惩罚
```

真正是奖励还是惩罚，要看配置里的权重。通常 `termination` 权重是负数，所以它实际是摔倒惩罚。

---

## 6. 重置流程：`reset_idx()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
把已经摔倒、碰撞或超时的机器人重新放回训练场。
```

核心流程：

```text
check_termination()
判断哪些机器人该 reset
↓
compute_reward()
计算最后一步 reward，包括 termination penalty
↓
reset_idx(env_ids)
真正执行 reset
```

`reset_idx()` 主要做：

```text
1. 更新 curriculum，如果开启的话
2. 调用 _reset_dofs() 重置关节
3. 调用 _reset_root_states() 重置机器人身体
4. 重新生成 commands
5. 清空 last_actions、last_dof_vel、feet_air_time
6. 清空 episode_length_buf
7. 统计 episode reward 日志
8. 清空 episode_sums
9. 把 timeout 信息传给 PPO
```

---

## 7. 重置关节：`_reset_dofs()`

作用：

```text
reset 时重置机器人关节角度和关节速度。
```

核心逻辑：

```text
关节角度 = 默认关节角度 × 0.5~1.5 的随机数
关节速度 = 0
```

为什么要随机？

```text
让机器人每次 reset 后不是完全一样的姿态，
增强训练鲁棒性。
```

它只管：

```text
关节角度
关节速度
```

不管机器人整体位置和身体速度。

---

## 8. 重置身体：`_reset_root_states()`

作用：

```text
reset 时重置机器人整体身体状态。
```

它负责：

```text
身体位置
身体姿态
身体线速度
身体角速度
```

主要做：

```text
1. 把机器人恢复到 base_init_state
2. 放到对应环境的出生点 env_origins
3. 如果 custom_origins=True，在 x/y 方向随机偏移一点
4. 给身体线速度和角速度随机初始化到 -0.5 到 0.5
5. 写回 Isaac Gym
```

和 `_reset_dofs()` 的区别：

```text
_reset_dofs()
重置关节

_reset_root_states()
重置机器人整体身体
```

---

## 9. Reset 线完整理解

目前 reset 线可以这样记：

```text
机器人摔倒 / 超时
↓
check_termination()
标记 reset_buf
↓
_reward_termination()
判断是否属于失败惩罚
↓
reset_idx()
组织重置流程
↓
_reset_dofs()
重置关节
↓
_reset_root_states()
重置身体
↓
_resample_commands()
重新生成目标速度命令
↓
新 episode 开始
```

---

## 10. PPO 主循环：`on_policy_runner.py`

文件位置：

```text
rsl_rl/rsl_rl/runners/on_policy_runner.py
```

核心函数：

```python
learn()
```

作用：

```text
让策略和环境反复交互，收集数据，然后用 PPO 更新网络。
```

整体流程：

```text
获取 obs
↓
actor 根据 obs 输出 action
↓
env.step(action)
↓
环境返回 new obs、reward、done
↓
PPO 保存这一批交互数据
↓
重复 num_steps_per_env 步，完成 rollout
↓
compute_returns()
计算 return 和 advantage
↓
update()
更新 actor 和 critic
↓
记录日志、保存模型
```

一句话：

```text
learn() = 采样 + 学习
```

---

## 11. PPO 的 `act()`

文件位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

核心函数：

```python
act(obs, critic_obs)
```

作用：

```text
让 actor 根据 obs 输出 action，
让 critic 根据 critic_obs 估计 value，
并保存 PPO 更新时需要的数据。
```

它保存的数据包括：

```text
observations
critic_observations
actions
values
actions_log_prob
action_mean
action_sigma
```

其中：

```text
actions
是真正要传给 env.step() 的动作

values
是 critic 对当前状态的估值

actions_log_prob
是旧策略下这个 action 的概率，PPO 更新时要用

action_mean / action_sigma
表示动作分布的均值和标准差
```

---

## 12. ActorCritic 的 `act()`

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
先根据 observation 更新动作分布，
再从动作分布里随机采样 action。
```

这就是 action 层面的随机探索。

---

## 13. `update_distribution()`

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
策略认为比较合理的动作中心

std
动作探索的随机幅度

distribution.sample()
从动作分布里采样真正执行的 action
```

训练时：

```text
用 sample，有随机性，用于探索。
```

播放/测试时通常：

```text
直接用 mean，更稳定。
```

---

## 14. PPO action 生成完整链路

```text
learn()
调用 self.alg.act(obs, critic_obs)
↓
PPO.act()
调用 self.actor_critic.act(obs)
↓
ActorCritic.act()
调用 update_distribution(obs)
↓
update_distribution()
mean = actor(obs)
distribution = Normal(mean, std)
↓
distribution.sample()
采样 action
↓
env.step(action)
执行动作
```

---

## 15. 当前对随机性的完整理解

训练里有两种随机：

```text
1. command 随机
_resample_commands()
随机生成任务目标，比如前进、后退、转向、站立。

2. action 随机
ActorCritic 从动作分布中 sample action，
随机尝试不同关节动作。
```

二者关系：

```text
command 随机解决“学什么任务”
action 随机解决“怎么探索动作”
reward 评价“完成得好不好”
PPO 根据评价更新策略
```

---

## 16. 当前完整强化学习闭环

到现在，主线可以串成：

```text
commands 随机生成目标
↓
compute_observations()
把机器人状态和 commands 拼成 obs
↓
actor 根据 obs 输出动作分布
↓
sample action
↓
env.step(action)
↓
_compute_torques()
action 变 torque
↓
Isaac Gym simulate
机器人运动
↓
post_physics_step()
更新状态
↓
compute_reward()
根据 commands 和实际运动算 reward
↓
check_termination()
判断是否摔倒/超时
↓
reset_idx()
需要的话重置环境
↓
PPO 收集 rollout
↓
compute_returns()
计算优势
↓
update()
更新 actor 和 critic
```

---

## 17. 当前阶段最重要的一句话

```text
commands 是训练目标；
observation 是机器人看到的状态；
actor 根据 observation 输出动作分布；
sample 得到真正执行的 action；
action 通过 PD 控制变成 torque；
reward 判断机器人有没有完成 command；
PPO 用这些数据更新策略。
```

