"""
Example Usage of SAC Agent for Honeytoken Deployment
Demonstrates how to use the trained SAC agent
"""

import numpy as np
from sac_agent import SACAgent
from honeytoken_env import HoneytokenEnvironment


def example_1_basic_usage():
    """Example 1: Basic SAC usage"""
    print("=" * 70)
    print("EXAMPLE 1: Basic SAC Usage")
    print("=" * 70)
    
    # Create environment
    env = HoneytokenEnvironment()
    
    # Create SAC agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True
    )
    
    print(f"✓ SAC Agent created")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Device: {agent.device}")
    
    # Run one episode
    state = env.reset()
    episode_reward = 0
    done = False
    step = 0
    
    print("\n🎮 Running episode...")
    while not done and step < 100:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Store and update
        agent.store_transition(state, action, reward, next_state, done)
        losses = agent.update()
        
        episode_reward += reward
        state = next_state
        step += 1
        
        if step % 20 == 0:
            print(f"  Step {step}: Reward = {reward:.2f}, Accuracy = {info['accuracy']:.2%}")
    
    print(f"\n✓ Episode completed")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final accuracy: {info['accuracy']:.2%}")
    print(f"  Detections: {info['detections']}")


def example_2_train_and_save():
    """Example 2: Train and save model"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Train and Save SAC Model")
    print("=" * 70)
    
    env = HoneytokenEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    
    print("Training for 10 episodes (quick demo)...\n")
    
    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            episode_reward += reward
            state = next_state
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Accuracy = {info['accuracy']:.2%}")
    
    # Save model
    agent.save('sac_demo_model.pt')
    print("\n✓ Model saved to 'sac_demo_model.pt'")


def example_3_load_and_use():
    """Example 3: Load trained model and use it"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Load and Use Trained Model")
    print("=" * 70)
    
    env = HoneytokenEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent
    agent = SACAgent(state_dim, action_dim)
    
    # Try to load model (will create new if not exists)
    try:
        agent.load('sac_demo_model.pt')
        print("✓ Model loaded from 'sac_demo_model.pt'")
    except:
        print("⚠ No saved model found, using fresh agent")
    
    # Use agent for deployment recommendation
    print("\n📍 Getting honeytoken deployment recommendations...\n")
    
    for i in range(5):
        state = env.reset()
        action = agent.select_action(state, evaluate=True)
        
        # Parse action
        area_idx = int(np.clip(action[0], 0, len(env.portal_areas) - 1))
        token_type_idx = int(np.clip(action[1], 0, len(env.token_types) - 1))
        
        print(f"Recommendation {i + 1}:")
        print(f"  Area: {env.portal_areas[area_idx]}")
        print(f"  Type: {env.token_types[token_type_idx]}")
        print(f"  Value: {action[2]:.2f}")
        print(f"  Priority: {action[3]:.2f}")
        print()


def example_4_hyperparameter_tuning():
    """Example 4: Different hyperparameter configurations"""
    print("=" * 70)
    print("EXAMPLE 4: Hyperparameter Configurations")
    print("=" * 70)
    
    env = HoneytokenEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    configs = [
        {'name': 'Default', 'lr': 3e-4, 'gamma': 0.99, 'tau': 0.005},
        {'name': 'High LR', 'lr': 1e-3, 'gamma': 0.99, 'tau': 0.005},
        {'name': 'Low Gamma', 'lr': 3e-4, 'gamma': 0.95, 'tau': 0.005},
        {'name': 'Fast Update', 'lr': 3e-4, 'gamma': 0.99, 'tau': 0.01},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  LR={config['lr']}, Gamma={config['gamma']}, Tau={config['tau']}")
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config['lr'],
            gamma=config['gamma'],
            tau=config['tau']
        )
        
        # Quick test (3 episodes)
        rewards = []
        for _ in range(3):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        results[config['name']] = avg_reward
        print(f"  Average Reward: {avg_reward:.2f}")
    
    print("\n" + "=" * 70)
    print("Comparison Results:")
    print("=" * 70)
    for name, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {reward:8.2f}")


def example_5_real_world_deployment():
    """Example 5: Simulating real-world deployment"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Real-World Deployment Simulation")
    print("=" * 70)
    
    env = HoneytokenEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim)
    
    # Train briefly
    print("\n📚 Training agent...")
    for episode in range(20):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
    
    print("✓ Training complete\n")
    
    # Simulate deployment scenario
    print("🎯 Deployment Scenario: Student Portal under attack\n")
    
    deployed_honeytokens = []
    detections = 0
    false_positives = 0
    
    for day in range(7):
        print(f"Day {day + 1}:")
        state = env.reset()
        
        # Get recommendation
        action = agent.select_action(state, evaluate=True)
        
        # Parse and deploy
        area_idx = int(np.clip(action[0], 0, len(env.portal_areas) - 1))
        token_type_idx = int(np.clip(action[1], 0, len(env.token_types) - 1))
        
        honeytoken = {
            'day': day + 1,
            'area': env.portal_areas[area_idx],
            'type': env.token_types[token_type_idx],
            'priority': action[3]
        }
        
        deployed_honeytokens.append(honeytoken)
        
        print(f"  Deployed: {honeytoken['type']} in {honeytoken['area']}")
        
        # Simulate attacks
        num_attacks = np.random.randint(1, 5)
        detected = np.random.randint(0, num_attacks + 1)
        
        detections += detected
        print(f"  Attacks: {num_attacks}, Detected: {detected}")
        print()
    
    print("=" * 70)
    print("Weekly Summary:")
    print("=" * 70)
    print(f"Total Honeytokens Deployed: {len(deployed_honeytokens)}")
    print(f"Total Detections: {detections}")
    print(f"Average Detection Rate: {detections/7:.1f} per day")
    print(f"Areas Covered: {len(set([h['area'] for h in deployed_honeytokens]))}")
    print("=" * 70)


def main():
    """Run all examples"""
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_basic_usage,
            2: example_2_train_and_save,
            3: example_3_load_and_use,
            4: example_4_hyperparameter_tuning,
            5: example_5_real_world_deployment
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Invalid example number. Choose 1-5.")
    else:
        # Run all examples
        example_1_basic_usage()
        example_2_train_and_save()
        example_3_load_and_use()
        example_4_hyperparameter_tuning()
        example_5_real_world_deployment()


if __name__ == "__main__":
    main()