

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from env.cartpole_env import CartPoleEnv
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from utils.plotting import plot_scores
import numpy as np
import pickle

def train():
    env = CartPoleEnv(render_mode=None, is_training=True)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    env.agent = agent
    replay_buffer = ReplayBuffer(capacity=50000)
    batch_size = 128
    episodes = 400
    target_update_freq = 10
    scores = []
    writer = SummaryWriter("logs/tensorboard/dqn_cartpole")
    model_path = "models/dqn_cartpole.pt"
    buffer_path = "models/replay_buffer.pkl"
    state_path = "models/agent_state.pt"
    best_avg_reward = -float('inf')

    # Load model cũ nếu tồn tại
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    
    # Load trạng thái agent
    if os.path.exists(state_path):
        state_dict = torch.load(state_path, map_location=agent.device)
        agent.epsilon = state_dict.get('epsilon', agent.epsilon)
        agent.best_speed = state_dict.get('best_speed', agent.best_speed)
        agent.high_scores = state_dict.get('high_scores', agent.high_scores)
        # Giảm epsilon nếu model đã tốt
        if np.mean(agent.high_scores[-100:] or [0]) >= 300:
            agent.epsilon = 0.05
        print(f"Loaded agent state from {state_path}, epsilon set to {agent.epsilon}")

    # Load replay buffer (chỉ lấy 10000 samples mới nhất)
    # Load replay buffer
    if os.path.exists(buffer_path):
      with open(buffer_path, 'rb') as f:
        buffer_data = pickle.load(f)
        replay_buffer.buffer = list(buffer_data)[-10000:]  # Chuyển deque thành list và slice
      print(f"Loaded replay buffer from {buffer_path} with {len(replay_buffer)} samples")

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            step += 1

            if len(replay_buffer) >= 1000:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch, batch_size)
                if loss is not None:
                    writer.add_scalar("Loss/Step", loss, episode * 500 + step)
            
            if step % target_update_freq == 0:
                agent.soft_update_target(tau=0.01)

        agent.decay_epsilon()
        agent.update_speed(episode_reward)
        scores.append(episode_reward)
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        avg_reward = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        writer.add_scalar("Avg_Reward/100_Episodes", avg_reward, episode)
        writer.add_scalar("Speed/Episode", agent.best_speed, episode)

        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.1f}, "
              f"Avg Reward (100 ep): {avg_reward:.1f}, Epsilon: {agent.epsilon:.3f}")

        # Lưu model nếu avg reward cải thiện đáng kể
        if avg_reward > best_avg_reward + 10:  # Chỉ lưu nếu tăng ≥ 10
            best_avg_reward = avg_reward
            agent.save(model_path)
            state_dict = {
                'epsilon': agent.epsilon,
                'best_speed': agent.best_speed,
                'high_scores': agent.high_scores
            }
            torch.save(state_dict, state_path)
            with open(buffer_path, 'wb') as f:
                pickle.dump(replay_buffer.buffer, f)
            print(f"New best model, state, and buffer saved to models/ with avg reward {avg_reward:.1f}")

        if len(scores) >= 100 and avg_reward >= 475:
            print("CartPole solved! Stopping training.")
            break

    env.close()
    plot_scores(scores, "logs/tensorboard/dqn_cartpole")
    writer.close()
    return model_path

def test(model_path):
    env = CartPoleEnv(render_mode="human", is_training=False)  # Tắt penalty/delay
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    env.agent = agent
    agent.load(model_path)
    agent.epsilon = 0.0

    try:
        episode_rewards = []
        for ep in range(10):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)
                state, reward, done = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f"Test Episode {ep+1}/10, Reward: {total_reward:.1f}")

        avg_test_reward = np.mean(episode_rewards)
        print(f"Average Test Reward (10 episodes): {avg_test_reward:.1f}")

    except KeyboardInterrupt:
        print("Kết thúc test do người dùng yêu cầu.")
    finally:
        env.close()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    model_path = train()
    print("Training completed.")
    print("Starting test with GUI...")
    test(model_path)