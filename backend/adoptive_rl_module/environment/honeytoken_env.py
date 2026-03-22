"""
Adaptive Honeytoken Deployment Environment
Multi-Agent Deep Reinforcement Learning Environment for Student Portal
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Any
import random
from datetime import datetime, timedelta

class HoneytokenEnvironment(gym.Env):
    """
    Custom Environment for Adaptive Honeytoken Deployment
    Supports multi-agent RL algorithms (PPO, DDPG, SAC)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(HoneytokenEnvironment, self).__init__()
        
        # Configuration
        self.config = config or {}
        self.num_agents = self.config.get('num_agents', 3)
        self.max_honeytokens = self.config.get('max_honeytokens', 50)
        self.portal_areas = self.config.get('portal_areas', [
            'login', 'dashboard', 'grades', 'assignments', 
            'resources', 'profile', 'settings', 'announcements'
        ])
        
        # State space dimensions
        # [threat_level, user_activity, honeytokens_deployed, detection_rate, 
        #  time_of_day, day_of_week, area_traffic for each area, historical_attacks]
        state_dim = 4 + 2 + len(self.portal_areas) + 10  # 26 features
        
        # Action space: [area_to_deploy, token_type, token_value, priority]
        # Continuous actions for flexibility with DDPG/SAC
        action_dim = 4
        
        # Define spaces for multi-agent
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([len(self.portal_areas)-1, 4, 1, 1]), 
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 1000)
        self.deployed_honeytokens = []
        self.attack_history = []
        self.detection_history = []
        
        # Threat simulation
        self.threat_levels = ['low', 'medium', 'high', 'critical']
        self.current_threat_level = 0
        
        # Metrics
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # Honeytoken types
        self.token_types = [
            'fake_credential',
            'fake_grade',
            'fake_assignment',
            'fake_resource_link',
            'fake_api_key'
        ]
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.deployed_honeytokens = []
        self.attack_history = []
        self.detection_history = []
        self.current_threat_level = 0
        
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Threat level (normalized)
        state.append(self.current_threat_level / len(self.threat_levels))
        
        # User activity (simulated, normalized)
        current_hour = datetime.now().hour
        user_activity = self._simulate_user_activity(current_hour)
        state.append(user_activity)
        
        # Honeytokens deployed (normalized)
        state.append(len(self.deployed_honeytokens) / self.max_honeytokens)
        
        # Detection rate
        total_attacks = len(self.attack_history)
        detection_rate = self.true_positives / max(total_attacks, 1)
        state.append(detection_rate)
        
        # Time features
        state.append(current_hour / 24.0)  # Time of day
        state.append(datetime.now().weekday() / 7.0)  # Day of week
        
        # Traffic per portal area (simulated)
        for area in self.portal_areas:
            traffic = self._simulate_area_traffic(area)
            state.append(traffic)
        
        # Historical attack patterns (last 10 steps)
        recent_attacks = self.attack_history[-10:]
        for i in range(10):
            if i < len(recent_attacks):
                state.append(recent_attacks[i])
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Parse action
        area_idx = int(np.clip(action[0], 0, len(self.portal_areas)-1))
        token_type_idx = int(np.clip(action[1], 0, len(self.token_types)-1))
        token_value = float(np.clip(action[2], 0, 1))
        priority = float(np.clip(action[3], 0, 1))
        
        area = self.portal_areas[area_idx]
        token_type = self.token_types[token_type_idx]
        
        # Deploy honeytoken
        honeytoken = {
            'area': area,
            'type': token_type,
            'value': token_value,
            'priority': priority,
            'deployed_at': self.current_step
        }
        
        if len(self.deployed_honeytokens) < self.max_honeytokens:
            self.deployed_honeytokens.append(honeytoken)
        
        # Simulate attack
        attack_occurred = self._simulate_attack()
        
        # Calculate reward
        reward = self._calculate_reward(honeytoken, attack_occurred)
        
        # Update metrics
        if attack_occurred:
            self.attack_history.append(1.0)
            detected = self._check_detection(honeytoken, area)
            if detected:
                self.true_positives += 1
                self.detection_history.append(1)
            else:
                self.false_negatives += 1
                self.detection_history.append(0)
        else:
            self.attack_history.append(0.0)
            # Check for false positives
            if random.random() < 0.05:  # 5% false positive rate
                self.false_positives += 1
            else:
                self.true_negatives += 1
        
        # Update threat level dynamically
        self._update_threat_level()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'accuracy': self._calculate_accuracy(),
            'precision': self._calculate_precision(),
            'recall': self._calculate_recall(),
            'f1_score': self._calculate_f1_score(),
            'deployed_tokens': len(self.deployed_honeytokens),
            'total_attacks': len(self.attack_history),
            'detections': self.true_positives
        }
        
        return next_state, reward, done, info
    
    def _simulate_user_activity(self, hour: int) -> float:
        """Simulate user activity based on time of day"""
        # Peak hours: 9-11 AM, 2-4 PM, 7-9 PM
        peak_hours = [9, 10, 11, 14, 15, 16, 19, 20, 21]
        if hour in peak_hours:
            return random.uniform(0.6, 1.0)
        return random.uniform(0.1, 0.4)
    
    def _simulate_area_traffic(self, area: str) -> float:
        """Simulate traffic for a specific portal area"""
        # Different areas have different traffic patterns
        traffic_profiles = {
            'login': 0.9,
            'dashboard': 0.8,
            'grades': 0.7,
            'assignments': 0.75,
            'resources': 0.5,
            'profile': 0.3,
            'settings': 0.2,
            'announcements': 0.6
        }
        base_traffic = traffic_profiles.get(area, 0.5)
        return base_traffic + random.uniform(-0.1, 0.1)
    
    def _simulate_attack(self) -> bool:
        """Simulate whether an attack occurs"""
        # Attack probability based on threat level
        attack_probabilities = {
            0: 0.05,   # low
            1: 0.15,   # medium
            2: 0.30,   # high
            3: 0.50    # critical
        }
        prob = attack_probabilities.get(self.current_threat_level, 0.1)
        return random.random() < prob
    
    def _check_detection(self, honeytoken: Dict, target_area: str) -> bool:
        """Check if honeytoken detected the attack"""
        # Detection probability based on honeytoken quality and placement
        base_detection = 0.5
        
        # Bonus for matching area
        if honeytoken['area'] == target_area:
            base_detection += 0.2
        
        # Bonus for priority
        base_detection += honeytoken['priority'] * 0.2
        
        # Bonus for token value (authenticity)
        base_detection += honeytoken['value'] * 0.1
        
        return random.random() < min(base_detection, 0.95)
    
    def _update_threat_level(self):
        """Update threat level based on recent attacks"""
        recent_window = 20
        recent_attacks = self.attack_history[-recent_window:]
        
        if len(recent_attacks) > 0:
            attack_rate = sum(recent_attacks) / len(recent_attacks)
            
            if attack_rate > 0.5:
                self.current_threat_level = min(3, self.current_threat_level + 1)
            elif attack_rate < 0.1:
                self.current_threat_level = max(0, self.current_threat_level - 1)
    
    def _calculate_reward(self, honeytoken: Dict, attack_occurred: bool) -> float:
        """Calculate reward for the action taken"""
        reward = 0.0
        
        # Base reward for deploying honeytoken
        reward += 0.1
        
        # Penalty for over-deployment
        if len(self.deployed_honeytokens) > self.max_honeytokens * 0.8:
            reward -= 0.2
        
        # Reward for successful detection
        if attack_occurred:
            area = honeytoken['area']
            if self._check_detection(honeytoken, area):
                reward += 10.0  # High reward for detection
                reward += honeytoken['value'] * 2.0  # Bonus for authenticity
            else:
                reward -= 5.0  # Penalty for missed detection
        
        # Reward for strategic placement
        traffic = self._simulate_area_traffic(honeytoken['area'])
        reward += traffic * honeytoken['priority'] * 0.5
        
        # Efficiency reward
        accuracy = self._calculate_accuracy()
        reward += accuracy * 5.0
        
        return reward
    
    def _calculate_accuracy(self) -> float:
        """Calculate detection accuracy"""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    def _calculate_precision(self) -> float:
        """Calculate precision"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def _calculate_recall(self) -> float:
        """Calculate recall"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def _calculate_f1_score(self) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Threat Level: {self.threat_levels[self.current_threat_level]}")
            print(f"Deployed Honeytokens: {len(self.deployed_honeytokens)}/{self.max_honeytokens}")
            print(f"Accuracy: {self._calculate_accuracy():.2%}")
            print(f"Detections: {self.true_positives}")
            print(f"Total Attacks: {len(self.attack_history)}")