"""
Standalone SAC Training Script for Adaptive Honeytoken Deployment
This script can be run independently to train the SAC agent

Usage:
    python train_sac.py --episodes 1000 --save_path models/sac_best.pt
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import os
from datetime import datetime

# Import SAC agent and environment
from sac_agent import SACAgent
from honeytoken_env import HoneytokenEnvironment


class SACTrainer:
    """SAC Training Manager"""
    
    def __init__(self, save_dir='sac_models'):
        self.env = HoneytokenEnvironment()
        
        # Get dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create SAC agent
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_entropy=True,
            buffer_size=100000,
            batch_size=64
        )
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_detections = []
        self.episode_losses = []
        
        # Save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train(self, num_episodes=1000, eval_interval=50, save_interval=100):
        """Train SAC agent"""
        
        print("=" * 80)
        print("SAC TRAINING - ADAPTIVE HONEYTOKEN DEPLOYMENT ENGINE")
        print("=" * 80)
        print(f"State Dimension: {self.env.observation_space.shape[0]}")
        print(f"Action Dimension: {self.env.action_space.shape[0]}")
        print(f"Episodes: {num_episodes}")
        print(f"Save Directory: {self.save_dir}")
        print("=" * 80)
        
        best_accuracy = 0.0
        
        for episode in tqdm(range(num_episodes), desc="Training SAC"):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            step = 0
            
            while not done:
                # Select action (with exploration during training)
                action = self.agent.select_action(state, evaluate=False)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                step += 1
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Update agent
                losses = self.agent.update()
                if losses:
                    episode_loss.append(losses.get('policy_loss', 0))
                
                state = next_state
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_accuracies.append(info['accuracy'])
            self.episode_detections.append(info['detections'])
            if episode_loss:
                self.episode_losses.append(np.mean(episode_loss))
            
            # Periodic logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_accuracy = np.mean(self.episode_accuracies[-10:])
                avg_detections = np.mean(self.episode_detections[-10:])
                
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Reward: {avg_reward:.2f}")
                print(f"  Accuracy: {avg_accuracy:.2%}")
                print(f"  Detections: {avg_detections:.1f}")
                print(f"  Steps: {step}")
            
            # Save best model
            if info['accuracy'] > best_accuracy:
                best_accuracy = info['accuracy']
                self.agent.save(f"{self.save_dir}/best_model.pt")
                print(f"  ✓ New best accuracy: {best_accuracy:.2%}")
            
            # Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                eval_accuracy = self.evaluate(num_episodes=20)
                print(f"  Evaluation Accuracy: {eval_accuracy:.2%}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self.agent.save(f"{self.save_dir}/checkpoint_{episode + 1}.pt")
                self.save_metrics()
                self.plot_training()
        
        # Final save
        self.agent.save(f"{self.save_dir}/final_model.pt")
        self.save_metrics()
        self.plot_training()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print(f"Best Accuracy: {best_accuracy:.2%}")
        print(f"Models saved in: {self.save_dir}")
        print("=" * 80)
        
        return best_accuracy
    
    def evaluate(self, num_episodes=100):
        """Evaluate trained agent"""
        accuracies = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Deterministic action (evaluation mode)
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
            
            accuracies.append(info['accuracy'])
        
        return np.mean(accuracies)
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_accuracies': self.episode_accuracies,
            'episode_detections': self.episode_detections,
            'episode_losses': self.episode_losses,
            'final_accuracy': self.episode_accuracies[-1] if self.episode_accuracies else 0,
            'best_accuracy': max(self.episode_accuracies) if self.episode_accuracies else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{self.save_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {self.save_dir}/metrics.json")
    
    def plot_training(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Raw')
        if len(self.episode_rewards) > 50:
            smoothed = self._moving_average(self.episode_rewards, 50)
            axes[0, 0].plot(smoothed, linewidth=2, label='Smoothed (50 ep)')
        axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Detection Accuracy
        axes[0, 1].plot(self.episode_accuracies, alpha=0.6, label='Raw')
        if len(self.episode_accuracies) > 50:
            smoothed = self._moving_average(self.episode_accuracies, 50)
            axes[0, 1].plot(smoothed, linewidth=2, label='Smoothed (50 ep)')
        axes[0, 1].set_title('Detection Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='85% Target', alpha=0.5)
        axes[0, 1].axhline(y=0.96, color='g', linestyle='--', label='96% Target', alpha=0.5)
        
        # Successful Detections
        axes[1, 0].plot(self.episode_detections, alpha=0.6, label='Raw')
        if len(self.episode_detections) > 50:
            smoothed = self._moving_average(self.episode_detections, 50)
            axes[1, 0].plot(smoothed, linewidth=2, label='Smoothed (50 ep)')
        axes[1, 0].set_title('Successful Detections per Episode', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Detections')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Policy Loss
        if self.episode_losses:
            axes[1, 1].plot(self.episode_losses, alpha=0.6, label='Policy Loss')
            if len(self.episode_losses) > 50:
                smoothed = self._moving_average(self.episode_losses, 50)
                axes[1, 1].plot(smoothed, linewidth=2, label='Smoothed (50 ep)')
            axes[1, 1].set_title('Policy Loss', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.save_dir}/training_curves.png")
    
    @staticmethod
    def _moving_average(data, window=50):
        """Calculate moving average"""
        if len(data) < window:
            return data
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SAC Agent for Honeytoken Deployment')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--save_dir', type=str, default='sac_models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SACTrainer(save_dir=args.save_dir)
    
    # Train
    best_accuracy = trainer.train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_accuracy = trainer.evaluate(num_episodes=100)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Training Accuracy: {best_accuracy:.2%}")
    print(f"Final Evaluation Accuracy: {final_accuracy:.2%}")
    print(f"Target Range: 85% - 96%")
    print(f"Status: {'✓ PASSED' if 0.85 <= final_accuracy <= 0.96 else '✗ NEEDS MORE TRAINING'}")
    print("=" * 80)


if __name__ == "__main__":
    main()