# Chemistry-experiment
I will apply machine learning and reinforcement learning to student experiment and discuss the optimization of experimental conditions
# 必要なライブラリのインストール
!pip install numpy torch

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
!pip install japanize-matplotlib
import japanize_matplotlib # noqa

# 仮想実験環境の定義
class SaponificationEnv:
    """
    安息香酸エチルの加水分解を模擬した仮想環境
    """
    def __init__(self):
        # 状態：[温度(℃), 反応時間(min)]
        self.state_space_low = np.array([50., 10.])   # 温度, 時間の下限
        self.state_space_high = np.array([150., 60.]) # 温度, 時間の上限

        # 行動：[温度変化(℃), 時間変化(min)]
        self.action_space_low = np.array([-10., -5.])
        self.action_space_high = np.array([10., 5.])

        # 最適条件
        self._optimal_temp = 115.
        self._optimal_time = 35.

    def reset(self):
        """初期状態をランダムに設定"""
        self.state = np.random.uniform(low=self.state_space_low, high=self.state_space_high)
        return self.state

    def step(self, action):
        """行動を取り、次の状態と報酬を返す"""
        # 行動をクリップして範囲内に収める
        action = np.clip(action, self.action_space_low, self.action_space_high)

        # 次の状態を計算
        self.state = np.clip(self.state + action, self.state_space_low, self.state_space_high)

        # 報酬（収率）を計算
        temp, time = self.state

        # 最適条件からの距離に基づいて収率（報酬）を決定
        # ガウス関数を用いて、最適点で収率が最大(約100%)になるように設定
        temp_diff = (temp - self._optimal_temp)**2
        time_diff = (time - self._optimal_time)**2

        # 収率を報酬として返す（マイナスの値なので、0に近いほど良い）
        reward = - (0.005 * temp_diff + 0.02 * time_diff) + np.random.normal(0, 0.01) # ノイズを追加

        # この簡易環境では常に False
        done = False

        return self.state, reward, done

# ダイナミクスモデル（ファイル改変）
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim, units=(64, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, units[0]),
            nn.ReLU(),
            nn.Linear(units[0], units[1]),
            nn.ReLU(),
            nn.Linear(units[1], output_dim)
        )
        self._loss_fn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, inputs):
        return self.model(inputs)

    def fit(self, inputs, labels):
        self.train()
        predicts = self.predict(inputs)
        loss = self._loss_fn(predicts, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

# ランダム方策（ファイル改変）
class RandomPolicy:
    def __init__(self, action_low, action_high, act_dim):
        self._action_low = action_low
        self._action_high = action_high
        self._act_dim = act_dim

    def get_actions(self, batch_size):
        return np.random.uniform(
            low=self._action_low,
            high=self._action_high,
            size=(batch_size, self._act_dim))

# グローバル変数としてモデルと方策を定義
env = SaponificationEnv()
obs_dim = len(env.state_space_low)
act_dim = len(env.action_space_low)

dynamics_model = DynamicsModel(input_dim=obs_dim + act_dim, output_dim=obs_dim)
policy = RandomPolicy(env.action_space_low, env.action_space_high, act_dim)

# 次の状態予測関数（ファイル改変）
def predict_next_state(obses, acts):
    inputs = np.concatenate([obses, acts], axis=1)
    inputs = torch.from_numpy(inputs).float()
    
    dynamics_model.eval()
    with torch.no_grad():
        obs_diffs = dynamics_model.predict(inputs).numpy()
        
    next_obses = obses + obs_diffs
    return next_obses

# 報酬予測関数（環境の報酬関数を利用）
def reward_fn(obses, acts):
    rewards = []
    for obs, act in zip(obses, acts):
        # 簡易的に、現在の状態で得られる報酬を計算
        temp, time = obs
        temp_diff = (temp - env._optimal_temp)**2
        time_diff = (time - env._optimal_time)**2
        reward = - (0.005 * temp_diff + 0.02 * time_diff)
        rewards.append(reward)
    return np.array(rewards)

# Random Shooting関数（改変）
def random_shooting(init_obs, n_mpc_episodes=128, horizon=15):
    init_actions = policy.get_actions(batch_size=n_mpc_episodes)
    returns = np.zeros(shape=(n_mpc_episodes,))
    obses = np.tile(init_obs, (n_mpc_episodes, 1))

    for i in range(horizon):
        acts = init_actions if i == 0 else policy.get_actions(batch_size=n_mpc_episodes)
        next_obses = predict_next_state(obses, acts)
        rewards = reward_fn(obses, acts)
        returns += rewards
        obses = next_obses

    best_episode_idx = np.argmax(returns)
    return init_actions[best_episode_idx]
# 収集したデータを保存するバッファ
data_buffer = []
buffer_size = 10000

# 1. ダイナミクスモデルの事前学習
print("ダイナミクスモデルの事前学習中...")
obs = env.reset()
for _ in range(2000):
    act = policy.get_actions(1)[0]
    next_obs, _, _ = env.step(act)
    data_buffer.append((obs, act, next_obs))
    if len(data_buffer) > buffer_size:
        data_buffer.pop(0)
    obs = next_obs

    if len(data_buffer) > 128 and _ % 10 == 0:
        indices = np.random.choice(len(data_buffer), 128)
        samples = [data_buffer[i] for i in indices]
        obs_b, act_b, next_obs_b = zip(*samples)
        inputs_torch = torch.from_numpy(np.concatenate([obs_b, act_b], axis=1)).float()
        labels_torch = torch.from_numpy(np.array(next_obs_b) - np.array(obs_b)).float()
        dynamics_model.fit(inputs_torch, labels_torch)

# 2. RSを用いた最適条件の探索
print("Random Shootingによる最適条件の探索開始...")
history = {'states': [], 'rewards': []}
obs = env.reset() # 実験の初期条件をリセット

for i in tqdm(range(200)): # 200回の実験（試行錯誤）をシミュレート

    # RSで次に試すべき最良の行動を決定
    best_action = random_shooting(obs)

    # 決定された行動を環境（実験）で実行
    next_obs, reward, _ = env.step(best_action)

    # 結果を記録
    data_buffer.append((obs, act, next_obs))
    if len(data_buffer) > buffer_size:
        data_buffer.pop(0)

    history['states'].append(next_obs.copy())
    history['rewards'].append(reward)

    obs = next_obs

    # 5回ごとにダイナミクスモデルを再学習して精度を上げる
    if i % 5 == 0 and len(data_buffer) > 128:
        indices = np.random.choice(len(data_buffer), 128)
        samples = [data_buffer[i] for i in indices]
        obs_b, act_b, next_obs_b = zip(*samples)
        inputs_torch = torch.from_numpy(np.concatenate([obs_b, act_b], axis=1)).float()
        labels_torch = torch.from_numpy(np.array(next_obs_b) - np.array(obs_b)).float()
        dynamics_model.fit(inputs_torch, labels_torch)

optimal_idx = np.argmax(history['rewards'])
optimal_state = history['states'][optimal_idx]
max_reward = history['rewards'][optimal_idx]

print("\\n探索終了！")
print(f"発見された最適条件: 温度 = {optimal_state[0]:.2f} ℃, 時間 = {optimal_state[1]:.2f} min")
print(f"その時の仮想収率（報酬）: {max_reward:.4f} (0に近いほど良い)")

# 結果の可視化

states_arr = np.array(history['states'])
plt.figure(figsize=(12, 6))

# パラメータ探索の軌跡
plt.subplot(1, 2, 1)
plt.plot(states_arr[:, 0], states_arr[:, 1], '-o', alpha=0.5, label='探索軌跡')
plt.plot(env._optimal_temp, env._optimal_time, 'ro', markersize=15, label='真の最適条件')
plt.plot(optimal_state[0], optimal_state[1], 'go', markersize=15, label='発見した最適条件')
plt.xlabel("温度 (℃)")
plt.ylabel("反応時間 (min)")
plt.title("実験パラメータの探索軌跡")
plt.legend()
plt.grid(True)

# 収率（報酬）の推移
plt.subplot(1, 2, 2)
plt.plot(history['rewards'])
plt.xlabel("実験回数 (ステップ)")
plt.ylabel("仮想収率（報酬）")
plt.title("収率の推移")
plt.grid(True)

plt.tight_layout()
plt.show()
