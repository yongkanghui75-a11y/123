# legged_gym 学习笔记：从 `anymal_c_flat` 注册到奖励配置

## 1. 我现在学的主线

我现在要搞清楚的是：

```text
运行命令里的 --task=anymal_c_flat
到底是怎么一步步找到 ANYmal C 平地机器人任务的？
```

整体流程是：

```text
train.py
读取 --task=anymal_c_flat
↓
task_registry.py
根据任务名查登记表
↓
envs/__init__.py
注册 anymal_c_flat 任务
↓
anymal_c_flat_config.py
定义 ANYmal C 平地任务配置
↓
legged_robot.py
真正执行环境逻辑、observation、reward、action
```

---

## 2. 训练入口：`train.py`

文件位置：

```text
legged_gym/scripts/train.py
```

这个文件是训练入口，相当于“开始训练按钮”。

核心代码是：

```python
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```

对应含义：

```text
make_env()
创建机器人训练环境

make_alg_runner()
创建 PPO 训练器

learn()
正式开始训练
```

所以 `train.py` 本身不负责具体奖励，也不负责机器人动作细节，它主要负责把环境和 PPO 训练器连接起来。

---

## 3. 任务登记本：`task_registry.py`

文件位置：

```text
legged_gym/utils/task_registry.py
```

这里的 `TaskRegistry` 可以理解成“任务登记本”。

它里面有三张表：

```python
self.task_classes = {}
self.env_cfgs = {}
self.train_cfgs = {}
```

含义是：

```text
task_classes
任务名 → 环境代码

env_cfgs
任务名 → 环境配置

train_cfgs
任务名 → PPO训练配置
```

比如 `anymal_c_flat` 会对应：

```text
anymal_c_flat
→ LeggedRobot
→ AnymalCFlatCfg
→ AnymalCFlatCfgPPO
```

---

## 4. 任务注册：`register()`

文件位置：

```text
legged_gym/utils/task_registry.py
```

核心代码：

```python
def register(self, name, task_class, env_cfg, train_cfg):
    self.task_classes[name] = task_class
    self.env_cfgs[name] = env_cfg
    self.train_cfgs[name] = train_cfg
```

作用：

```text
把一个任务名登记进去。
以后用户输入这个任务名，程序就知道该用哪个环境、哪个配置、哪个训练参数。
```

比如：

```python
task_registry.register(
    "anymal_c_flat",
    LeggedRobot,
    AnymalCFlatCfg(),
    AnymalCFlatCfgPPO()
)
```

大白话：

```text
如果用户输入 --task=anymal_c_flat，
就用 LeggedRobot 作为环境代码，
用 AnymalCFlatCfg 作为环境配置，
用 AnymalCFlatCfgPPO 作为 PPO 训练配置。
```

---

## 5. `anymal_c_flat` 注册位置

文件位置：

```text
legged_gym/envs/__init__.py
```

这个文件负责把任务注册进 `task_registry`。

可以用命令查找：

```bash
grep -R "anymal_c_flat" -n legged_gym/envs
```

找到后会看到类似：

```python
task_registry.register(
    "anymal_c_flat",
    LeggedRobot,
    AnymalCFlatCfg(),
    AnymalCFlatCfgPPO()
)
```

这就是 `anymal_c_flat` 和 ANYmal C 平地任务绑定起来的位置。

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
检查 anymal_c_flat 是否已经注册
↓
找到对应环境类 LeggedRobot
↓
找到对应环境配置 AnymalCFlatCfg
↓
用命令行参数覆盖配置
↓
解析 Isaac Gym 仿真参数
↓
创建环境 env
```

关键代码：

```python
env = task_class(
    cfg=env_cfg,
    sim_params=sim_params,
    physics_engine=args.physics_engine,
    sim_device=args.sim_device,
    headless=args.headless
)
```

大白话：

```text
make_env() = 建训练场 + 把机器人放进去
```

这里的训练场包括：

```text
机器人模型
地形
物理仿真
奖励规则
观测数据
GPU设置
并行环境数量
```

---

## 7. 创建 PPO 训练器：`make_alg_runner()`

文件位置：

```text
legged_gym/utils/task_registry.py
```

作用：

```text
创建 PPO 训练器。
```

核心代码：

```python
runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
```

大白话：

```text
make_alg_runner() = 找 PPO 教练 + 制定训练计划
```

它负责准备：

```text
训练环境 env
PPO 参数 train_cfg_dict
日志保存目录 log_dir
训练设备 cuda/cpu
```

后面 `train.py` 调用：

```python
ppo_runner.learn(...)
```

才真正开始训练。

---

## 8. ANYmal C 平地任务配置：`anymal_c_flat_config.py`

文件位置：

```text
legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py
```

这个文件定义：

```text
ANYmal C 在平地任务中怎么训练。
```

它有两大部分：

```python
class AnymalCFlatCfg(AnymalCRoughCfg):
```

这是环境配置。

```python
class AnymalCFlatCfgPPO(AnymalCRoughCfgPPO):
```

这是 PPO 训练配置。

---

## 9. 环境配置：`AnymalCFlatCfg`

### 观测数量

```python
class env(AnymalCRoughCfg.env):
    num_observations = 48
```

含义：

```text
机器人每一步能看到 48 个数字。
```

这些数字之后会输入神经网络。

---

### 地形配置

```python
class terrain(AnymalCRoughCfg.terrain):
    mesh_type = 'plane'
    measure_heights = False
```

含义：

```text
地形是平地。
不需要测周围地形高度。
```

这就是任务名里 `flat` 的含义。

---

### 奖励配置

```python
class rewards(AnymalCRoughCfg.rewards):
    max_contact_force = 350.

    class scales(AnymalCRoughCfg.rewards.scales):
        orientation = -5.0
        torques = -0.000025
        feet_air_time = 2.
```

这里写的是“奖励权重”，不是完整奖励函数。

含义：

```text
orientation = -5.0
身体歪了会被重罚，鼓励机器人保持稳定。

torques = -0.000025
关节用力太大会被惩罚，鼓励机器人省力。

feet_air_time = 2.
脚离地时间合适会被奖励，鼓励机器人迈步。

max_contact_force = 350.
脚和地面的接触力不能太大。
```

关键理解：

```text
reward 不是单纯的分数。
reward 是用来塑造机器人行为的。
```

---

## 10. PPO 配置：`AnymalCFlatCfgPPO`

核心代码：

```python
actor_hidden_dims = [128, 64, 32]
critic_hidden_dims = [128, 64, 32]
activation = 'elu'
entropy_coef = 0.01
experiment_name = 'flat_anymal_c'
max_iterations = 300
```

含义：

```text
actor_hidden_dims
动作网络大小，actor 负责输出动作。

critic_hidden_dims
价值网络大小，critic 负责判断状态好不好。

entropy_coef
鼓励机器人探索，不要太早变得死板。

experiment_name = 'flat_anymal_c'
训练日志保存到 logs/flat_anymal_c。

max_iterations = 300
默认训练 300 轮。
```

---

## 11. 目前最重要的结论

```text
anymal_c_flat_config.py 里写的是奖励权重。
legged_robot.py 里的 compute_reward() 才是真正计算奖励的地方。
```

也就是说：

```text
config.py = 写评分标准
legged_robot.py = 真正打分
```

---

## 12. 下一步要看的文件

下一步看：

```text
legged_gym/envs/base/legged_robot.py
```

重点函数：

```python
compute_reward()
```

学习目标：

```text
理解奖励到底是怎么计算出来的。
```

之后再看：

```python
compute_observations()
```

学习目标：

```text
理解机器人每一步到底“看到”了什么。
```

再看：

```python
step()
_compute_torques()
```

学习目标：

```text
理解 action 是怎么变成机器人关节力矩的。
```

最后看：

```text
rsl_rl/rsl_rl/runners/on_policy_runner.py
rsl_rl/rsl_rl/algorithms/ppo.py
```

学习目标：

```text
理解 PPO 是怎么更新策略网络的。
```

