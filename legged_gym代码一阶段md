# legged_gym 第一轮代码主线学习笔记

## 1. 总体主线

我现在学习的是 `legged_gym` 中一条完整的机器人强化学习训练流程：

```text
任务注册
↓
创建环境
↓
创建 PPO 训练器
↓
机器人获得 observation
↓
策略网络输出 action
↓
action 变成关节力矩 torque
↓
Isaac Gym 推进物理仿真
↓
计算 reward
↓
返回新的 observation、reward、done 给 PPO
```

---

## 2. 训练入口：`train.py`

文件位置：

```text
legged_gym/scripts/train.py
```

核心代码：

```python
env, env_cfg = task_registry.make_env(...)
ppo_runner, train_cfg = task_registry.make_alg_runner(...)
ppo_runner.learn(...)
```

作用：

```text
make_env()
创建机器人训练环境

make_alg_runner()
创建 PPO 训练器

learn()
开始训练
```

`train.py` 是训练入口，不负责具体奖励、动作和机器人细节。

---

## 3. 任务登记本：`task_registry.py`

文件位置：

```text
legged_gym/utils/task_registry.py
```

核心作用：

```text
根据任务名找到对应的环境代码、环境配置、PPO 配置。
```

内部有三张表：

```python
self.task_classes = {}
self.env_cfgs = {}
self.train_cfgs = {}
```

含义：

```text
task_classes：任务名 → 环境类
env_cfgs：任务名 → 环境配置
train_cfgs：任务名 → PPO 配置
```

---

## 4. 任务注册：`envs/__init__.py`

文件位置：

```text
legged_gym/envs/__init__.py
```

`anymal_c_flat` 在这里被注册：

```python
task_registry.register(
    "anymal_c_flat",
    LeggedRobot,
    AnymalCFlatCfg(),
    AnymalCFlatCfgPPO()
)
```

含义：

```text
--task=anymal_c_flat

对应：
环境代码：LeggedRobot
环境配置：AnymalCFlatCfg
PPO 配置：AnymalCFlatCfgPPO
```

---

## 5. 平地任务配置：`anymal_c_flat_config.py`

文件位置：

```text
legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py
```

这个文件定义 ANYmal C 平地任务怎么训练。

主要包括：

```python
class AnymalCFlatCfg
class AnymalCFlatCfgPPO
```

### 环境配置

```python
num_observations = 48
mesh_type = 'plane'
measure_heights = False
```

含义：

```text
机器人每一步看到 48 个数字。
地形是平地。
平地任务不需要测地形高度。
```

### 奖励权重

```python
orientation = -5.0
torques = -0.000025
feet_air_time = 2.
```

含义：

```text
orientation
身体歪了会被惩罚。

torques
关节用力太大会被惩罚。

feet_air_time
鼓励机器人迈步，而不是拖着脚滑。
```

注意：

```text
config.py 里写的是奖励权重；
legged_robot.py 里才是真正计算奖励。
```

---

## 6. 奖励准备：`_prepare_reward_function()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
把配置文件里的奖励名字，自动连接到真正的奖励函数。
```

规则：

```text
配置名 orientation
↓
函数名 _reward_orientation()

配置名 torques
↓
函数名 _reward_torques()

配置名 feet_air_time
↓
函数名 _reward_feet_air_time()
```

核心理解：

```text
配置文件决定算哪些 reward，以及每项权重是多少。
_prepare_reward_function() 负责把这些 reward 准备成函数列表。
```

---

## 7. 奖励计算：`compute_reward()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

核心逻辑：

```python
rew = self.reward_functions[i]() * self.reward_scales[name]
self.rew_buf += rew
```

含义：

```text
调用每个奖励函数
↓
乘以配置里的权重
↓
加到总 reward 里
```

最重要理解：

```text
_reward_xxx() 负责“怎么算”；
rewards.scales 负责“权重多大”；
compute_reward() 负责“统一加总”。
```

---

## 8. 已看的几个奖励函数

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

### `_reward_tracking_lin_vel()`

作用：

```text
鼓励机器人实际平移速度接近目标速度。
```

使用：

```text
commands[:, :2]：目标 x/y 速度
base_lin_vel[:, :2]：实际 x/y 速度
```

---

### `_reward_tracking_ang_vel()`

作用：

```text
鼓励机器人实际转向速度接近目标转向速度。
```

使用：

```text
commands[:, 2]：目标 yaw 角速度
base_ang_vel[:, 2]：实际 yaw 角速度
```

---

### `_reward_orientation()`

作用：

```text
惩罚身体歪斜，让机器人保持稳定。
```

---

### `_reward_torques()`

作用：

```text
惩罚关节用力太大，让动作更省力。
```

---

### `_reward_feet_air_time()`

作用：

```text
奖励脚抬起来再落地，让机器人学会迈步。
```

核心理解：

```text
tracking_lin_vel 和 tracking_ang_vel 让机器人按命令走；
orientation 让机器人别倒；
torques 让机器人省力；
feet_air_time 让机器人真正迈步。
```

---

## 9. Observation：`compute_observations()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
把机器人当前能看到的信息拼成 obs_buf，交给策略网络。
```

主要包含：

```text
base_lin_vel：身体线速度
base_ang_vel：身体角速度
projected_gravity：身体姿态/是否歪斜
commands：目标速度命令
dof_pos：关节位置
dof_vel：关节速度
actions：上一时刻动作
```

ANYmal C 平地任务是 48 维 observation：

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 = 48
```

---

## 10. 动作执行：`step(actions)`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
接收策略网络输出的 actions，
把动作执行到机器人身上，
然后返回新的 obs、reward、done。
```

核心流程：

```text
裁剪 action
↓
_compute_torques(actions)
↓
把 torque 发给 Isaac Gym
↓
simulate 推进物理仿真
↓
post_physics_step()
↓
返回 obs、reward、reset_buf
```

---

## 11. Action 变 Torque：`_compute_torques()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

核心理解：

```text
在 P 控制下，action 不是直接力矩。
action 表示默认站姿基础上的关节目标位置偏移。
```

核心公式逻辑：

```text
目标关节角度 = default_dof_pos + action * action_scale

torque =
P 增益 × 当前关节离目标还有多远
-
D 增益 × 当前关节速度
```

作用：

```text
把神经网络输出的 action 转换成真正施加到关节上的力矩。
```

---

## 12. 控制配置：`class control`

文件位置：

```text
legged_gym/envs/base/legged_robot_config.py
```

核心配置：

```python
control_type = 'P'
stiffness = {...}
damping = {...}
action_scale = 0.5
decimation = 4
```

含义：

```text
control_type = 'P'
使用位置控制。

stiffness
P 增益，关节追目标位置的力度。

damping
D 增益，防止关节乱甩和抖动。

action_scale
控制 action 对目标关节角度的影响大小。

decimation = 4
策略输出一次 action，仿真执行 4 个物理小步。
```

---

## 13. 物理后处理：`post_physics_step()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
机器人动完以后，更新状态、算 reward、算 observation、判断 reset。
```

核心流程：

```text
刷新机器人状态
刷新接触力
更新速度、姿态、重力方向
调用 _post_physics_step_callback()
检查是否摔倒/超时
计算 reward
重置需要 reset 的机器人
计算新的 observation
保存上一时刻动作和速度
```

---

## 14. Commands 更新：`_post_physics_step_callback()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
在计算 reward 和 observation 之前，更新 commands 等信息。
```

核心代码：

```python
env_ids = (...)
self._resample_commands(env_ids)
```

含义：

```text
每隔 resampling_time 秒，重新给机器人生成目标速度命令。
```

对于 `anymal_c_flat`：

```python
resampling_time = 4.
heading_command = False
```

含义：

```text
每 4 秒换一次目标速度。
不使用目标朝向模式，直接使用目标 yaw 角速度。
```

commands 会被两个地方使用：

```text
1. 放进 observation，让策略知道目标是什么。
2. 用于 reward，检查实际速度是否跟上目标速度。
```

---

## 15. 当前已经打通的环境主线

目前已经看懂的主线：

```text
任务名 anymal_c_flat
↓
注册到 LeggedRobot + AnymalCFlatCfg + AnymalCFlatCfgPPO
↓
make_env() 创建环境
↓
compute_observations() 生成 obs
↓
actor 根据 obs 输出 action
↓
step(actions) 执行动作
↓
_compute_torques() 把 action 转成 torque
↓
Isaac Gym 推进仿真
↓
post_physics_step() 更新状态
↓
compute_reward() 计算奖励
↓
返回 obs、reward、done 给 PPO
```

---

## 16. 当前阶段最重要的一句话

```text
observation 是机器人看到什么；
action 是策略想让机器人怎么动；
torque 是真正施加到关节上的力；
reward 是评价这一步做得好不好；
PPO 用 reward 去更新策略网络。
```

---

## 17. 后面还需要看的内容

接下来还剩三块：

```text
1. _resample_commands()
看目标速度 commands 是怎么随机生成的。

2. check_termination() 和 reset_idx()
看机器人什么时候摔倒、什么时候重置。

3. rsl_rl 中的 PPO 主线
看 obs 怎么进网络，action 怎么输出，reward 怎么用于更新策略。
```

