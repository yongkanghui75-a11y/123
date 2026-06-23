# legged_gym 第一轮主线笔记：环境部分

## 1. 当前学习目标

这一轮主要看懂 `legged_gym` 里环境部分的完整闭环：

```text
任务注册
↓
创建环境
↓
生成 observation
↓
策略输出 action
↓
action 变成 torque
↓
Isaac Gym 推进仿真
↓
更新状态
↓
计算 reward
↓
判断 reset
↓
重新开始下一轮 episode
```

目前重点不是完全精通每一行代码，而是先打通主线。

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

含义：

```text
make_env()
创建机器人训练环境

make_alg_runner()
创建 PPO 训练器

learn()
开始训练
```

`train.py` 是训练入口，不负责具体机器人动作、奖励和物理细节。

---

## 3. 任务登记本：`task_registry.py`

文件位置：

```text
legged_gym/utils/task_registry.py
```

`TaskRegistry` 可以理解为“任务登记本”。

里面有三张表：

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

例如：

```text
anymal_c_flat
→ LeggedRobot
→ AnymalCFlatCfg
→ AnymalCFlatCfgPPO
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

所以 `anymal_c_flat` 不是魔法名字，而是提前注册进任务登记本里的。

---

## 5. 平地任务配置：`anymal_c_flat_config.py`

文件位置：

```text
legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py
```

这个文件定义 ANYmal C 平地任务怎么训练。

主要有两个类：

```python
class AnymalCFlatCfg
class AnymalCFlatCfgPPO
```

其中环境部分主要看 `AnymalCFlatCfg`。

重要配置：

```python
num_observations = 48
mesh_type = 'plane'
measure_heights = False
```

含义：

```text
num_observations = 48
机器人每一步看到 48 个数字。

mesh_type = 'plane'
地形是平地。

measure_heights = False
平地任务不需要测周围地形高度。
```

奖励权重：

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

重要理解：

```text
config.py 里写的是奖励权重；
legged_robot.py 里才是真正计算奖励。
```

---

## 6. 创建环境：`make_env()`

文件位置：

```text
legged_gym/utils/task_registry.py
```

作用：

```text
根据任务名创建真正的机器人仿真环境。
```

主要流程：

```text
读取终端参数
↓
检查任务名是否注册
↓
找到环境类 LeggedRobot
↓
找到环境配置 AnymalCFlatCfg
↓
用命令行参数覆盖配置
↓
创建 Isaac Gym 环境
↓
返回 env 和 env_cfg
```

可以理解为：

```text
make_env() = 建训练场 + 把机器人放进去
```

---

## 7. 奖励准备：`_prepare_reward_function()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
把配置文件里的 reward 名字，自动连接到真正的 reward 函数。
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

核心逻辑：

```text
如果某个 reward 权重是 0，就不计算。
如果某个 reward 权重非 0，就把它加入 reward_functions 列表。
```

所以：

```text
rewards.scales 决定哪些奖励参与训练；
_prepare_reward_function() 把奖励名字变成真正要调用的函数。
```

---

## 8. 奖励计算：`compute_reward()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

核心代码：

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

## 9. 已看的几个 reward 函数

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

使用：

```text
projected_gravity
```

身体越歪，x/y 方向重力分量越大，惩罚越大。

---

### `_reward_torques()`

作用：

```text
惩罚关节用力太大，让动作更省力。
```

使用：

```text
self.torques
```

---

### `_reward_feet_air_time()`

作用：

```text
奖励脚抬起来再落地，让机器人学会迈步。
```

核心逻辑：

```text
脚离地一段时间
↓
再次接触地面
↓
给迈步奖励
```

如果机器人目标速度很小，也就是命令它站着，就不给这个迈步奖励。

---

## 10. Observation：`compute_observations()`

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

ANYmal C 平地任务 observation 是 48 维：

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 = 48
```

对应：

```text
身体线速度 3
身体角速度 3
重力方向 3
目标命令 3
关节位置 12
关节速度 12
上一时刻动作 12
```

---

## 11. Action 执行：`step(actions)`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
接收策略网络输出的 actions，
把动作执行到机器人身上，
然后返回新的 observation、reward、done。
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

一句话：

```text
step() = 把 action 真正执行到机器人身上，然后返回训练需要的数据。
```

---

## 12. Action 变 Torque：`_compute_torques()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

重要理解：

```text
在 P 控制下，action 不是直接力矩。
action 表示默认站姿基础上的关节目标位置偏移。
```

核心逻辑：

```text
目标关节角度 = default_dof_pos + action * action_scale

torque =
P 增益 × 当前关节离目标还有多远
-
D 增益 × 当前关节速度
```

所以：

```text
action = 策略网络给出的动作意图
torque = 真正施加到机器人关节上的力矩
```

---

## 13. Torque 是什么

`torque` 中文叫：

```text
力矩 / 扭矩
```

可以理解为：

```text
机器人关节电机真正输出的“拧动力量”。
```

区别：

```text
force
让物体直线移动的力。

torque
让关节或物体旋转的力。
```

在四足机器人里，髋关节、膝关节都是旋转关节，所以真正驱动机器人运动的是关节 torque。

---

## 14. 控制配置：`class control`

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

## 15. 物理后处理：`post_physics_step()`

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
更新身体速度、角速度、重力方向
调用 _post_physics_step_callback()
检查是否摔倒/超时
计算 reward
重置需要 reset 的机器人
计算新的 observation
保存上一时刻动作和速度
```

一句话：

```text
step() 负责让机器人动；
post_physics_step() 负责动完之后整理训练数据。
```

---

## 16. Commands 更新：`_post_physics_step_callback()`

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

---

## 17. Commands 随机生成：`_resample_commands()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

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

重要理解：

```text
command 随机 = 随机出题
action 探索 = 随机试答案
```

commands 会进入两条线：

```text
1. observation
告诉策略网络当前目标是什么。

2. reward
检查机器人实际运动有没有跟上目标。
```

完整关系：

```text
_resample_commands()
随机生成目标速度
↓
compute_observations()
把 commands 放进 observation
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

## 18. 为什么要随机生成 commands

如果 commands 永远固定，比如一直是：

```text
向前走 1.0 m/s
```

机器人最后可能只会这一种走法。

随机 commands 是为了让机器人学会：

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
commands 解决“学什么任务”；
actions 解决“怎么执行任务”；
reward 评价“做得好不好”。
```

---

## 19. 终止判断：`check_termination()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
判断哪些机器人需要 reset。
```

主要有两种 reset 原因：

```text
1. 不该接触地面的身体部位接触了地面
   通常代表摔倒或碰撞失败。

2. episode 时间太长，超过 max_episode_length
   这是正常超时结束，不算失败。
```

核心变量：

```text
reset_buf
表示这个环境是否需要 reset。

time_out_buf
表示是否因为超时而 reset。
```

---

## 20. 终止奖励：`_reward_termination()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

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
结果 = True，触发惩罚。

超时：
reset_buf = True
time_out_buf = True
结果 = False，不惩罚。
```

---

## 21. 重置流程：`reset_idx()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
把已经摔倒、碰撞或超时的机器人重新放回训练场。
```

主要做：

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

## 22. 重置关节：`_reset_dofs()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

作用：

```text
reset 时重置机器人关节角度和关节速度。
```

核心逻辑：

```text
关节角度 = 默认关节角度 × 0.5~1.5 的随机数
关节速度 = 0
```

为什么随机：

```text
让机器人每次 reset 后不是完全一样的姿态，
增强训练鲁棒性。
```

它只管：

```text
关节角度
关节速度
```

---

## 23. 重置身体：`_reset_root_states()`

文件位置：

```text
legged_gym/envs/base/legged_robot.py
```

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

## 24. Reset 线完整理解

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

## 25. 当前环境部分完整闭环

到这里，环境主线可以串成：

```text
任务名 anymal_c_flat
↓
注册到 LeggedRobot + AnymalCFlatCfg + AnymalCFlatCfgPPO
↓
make_env() 创建环境
↓
_resample_commands() 随机生成目标命令
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
check_termination() 判断是否摔倒/超时
↓
reset_idx() 重置需要重置的机器人
↓
返回 obs、reward、done 给 PPO
```

---

## 26. 当前阶段最重要的一句话

```text
commands 是训练目标；
observation 是机器人看到的状态；
action 是策略网络输出的动作意图；
torque 是真正施加到关节上的力矩；
reward 判断机器人有没有完成 command；
reset 负责让失败或超时的机器人重新开始训练。
```

---

## 27. 暂时不整理的内容

PPO 主线暂时不放入本笔记。

后面单独整理：

```text
on_policy_runner.py 的 learn()
ppo.py 的 act()
actor_critic.py 的 act()
update_distribution()
evaluate()
compute_returns()
update()
```

等 PPO 主线学完后，再整理成第二份笔记。

```
```

