# import gymnasium as gym
# import torch
# import time
# import numpy as np

# class CartPoleEnv:
#     """Quản lý môi trường CartPole-v1 từ Gymnasium."""
    
#     def __init__(self, env_name="CartPole-v1", render_mode=None, agent=None):
#         self.env = gym.make(env_name, render_mode=render_mode)
#         self.state_dim = self.env.observation_space.shape[0]  # 4D vector
#         self.action_dim = self.env.action_space.n  # 2 actions
#         self.render_mode = render_mode
#         self.agent = agent  # Tham chiếu đến agent để lấy best_speed
#         self.state = None

#     def reset(self):
#         state, _ = self.env.reset()
#         self.state = state
#         if self.render_mode == "human":
#             self.env.render()
#         return torch.FloatTensor(state)

#     def step(self, action):
#         angle = self.state[2]  # Góc của pole
#         # Tính tốc độ dựa trên góc: chậm khi cân bằng, nhanh khi nghiêng
#         speed = self.agent.best_speed * (0.5 + min(abs(angle) * 10, 1.0))
#         speed = float(speed)
        
#         next_state, reward, terminated, truncated, info = self.env.step(action)
#         self.state = next_state
#         done = terminated or truncated
        
#         # Giảm penalty để test reward cao hơn (gần training)
#         if done and terminated:
#             reward -= 1.0  # Thay vì -10, để tránh giảm nhiều
        
#         if self.render_mode == "human":
#             self.env.render()
#             time.sleep(speed)  # Áp dụng speed vào rendering
        
#         return torch.FloatTensor(next_state), reward, done

#     def render(self):
#         if self.render_mode is None:
#             self.env.render()
    
#     def close(self):
#         self.env.close()







import gymnasium as gym
import torch
import time
import numpy as np

class CartPoleEnv:
    """Quản lý môi trường CartPole-v1 từ Gymnasium."""
    
    def __init__(self, env_name="CartPole-v1", render_mode=None, agent=None, is_training=True):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.state_dim = self.env.observation_space.shape[0]  # 4D vector
        self.action_dim = self.env.action_space.n  # 2 actions
        self.render_mode = render_mode
        self.agent = agent
        self.state = None
        self.is_training = is_training  # Flag để điều chỉnh penalty và rendering

    def reset(self):
        state, _ = self.env.reset()
        self.state = state
        if self.render_mode == "human":
            self.env.render()
        return torch.FloatTensor(state)

    def step(self, action):
        angle = self.state[2]
        speed = self.agent.best_speed * (0.5 + min(abs(angle) * 5, 0.5))  # Giảm hệ số để delay nhỏ hơn
        speed = float(speed)
        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state = next_state
        done = terminated or truncated
        
        # Penalty chỉ áp dụng trong training
        if self.is_training and done and terminated:
            reward -= 1.0
        
        if self.render_mode == "human" and self.is_training:  # Delay chỉ trong training
            self.env.render()
            time.sleep(speed)
        elif self.render_mode == "human":  # Render không delay trong test
            self.env.render()
        
        return torch.FloatTensor(next_state), reward, done

    def render(self):
        if self.render_mode is None:
            self.env.render()
    
    def close(self):
        self.env.close()