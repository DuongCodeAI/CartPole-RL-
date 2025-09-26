# from collections import deque
# import random

# class ReplayBuffer:

    
#     def __init__(self, capacity):
    
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         """Thêm trải nghiệm vào buffer."""
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         """Lấy mẫu ngẫu nhiên từ buffer."""
#         return random.sample(self.buffer, batch_size)
    
#     def __len__(self):
#         return len(self.buffer)






from collections import deque
import random
import torch

class ReplayBuffer:
    """
    Replay buffer an toàn:
     - Lưu bản sao CPU của state/next_state (avoid mutable references)
     - sample() trả list of tuples như trước
     - sample_batch() trả tensors (states, actions, rewards, next_states, dones) đã to(device)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Lưu bản sao tách rời (CPU) để tránh trường hợp state tensor bị thay đổi ngoài ý muốn.
        Chấp nhận state/next_state là torch.Tensor.
        """
        # Nếu người dùng cung cấp numpy, convert sang tensor trước khi gọi push
        if isinstance(state, torch.Tensor):
            s = state.detach().cpu().clone()
        else:
            s = torch.tensor(state, dtype=torch.float32)
        if isinstance(next_state, torch.Tensor):
            ns = next_state.detach().cpu().clone()
        else:
            ns = torch.tensor(next_state, dtype=torch.float32)

        self.buffer.append((s, int(action), float(reward), ns, bool(done)))

    def sample(self, batch_size):
        """Trả về list of tuples (s,a,r,ns,done) giống cũ."""
        return random.sample(self.buffer, batch_size)

    def sample_batch(self, batch_size, device=None):
        """
        Trả về stacked tensors đã move sang device (nếu device không None).
        Trả về (states, actions, rewards, next_states, dones).
        """
        batch = self.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        if device is not None:
            states = states.to(device)
            next_states = next_states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
