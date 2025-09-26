from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

def plot_scores(scores, log_dir):
    """Vẽ biểu đồ điểm số và lưu log TensorBoard."""
    writer = SummaryWriter(log_dir)
    for episode, score in enumerate(scores):
        writer.add_scalar("Reward/Episode", score, episode)
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward over Episodes")
    plt.savefig(os.path.join(log_dir, "reward_plot.png"))
    plt.close()
    writer.close()