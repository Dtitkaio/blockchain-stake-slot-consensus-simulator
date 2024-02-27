import numpy as np
from config import EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, REWARD_RATIO, STAKE_DISTRIBUTION

class PoSA_Simulator:
    def __init__(self, stake_distribution, epochs, authority_count, total_reward, reward_ratio, selection_strategy='stake'):
        self.stake_distribution = np.array(stake_distribution)
        self.epochs = epochs
        self.current_epoch = 0  # Initialize current epoch
        self.authority_count = authority_count
        self.ageing = np.zeros_like(self.stake_distribution)  # Initialize ageing for each validator
        self.total_reward = total_reward
        self.reward_ratio = reward_ratio
        if selection_strategy == 'stake':
            self.select_authorities = self.select_authorities_by_stake
        elif selection_strategy == 'random':
            self.select_authorities = self.select_authorities_randomly
        elif selection_strategy == 'multiplicative_ageing':
            self.select_authorities = self.select_authorities_by_multiplicative_ageing
        elif selection_strategy == 'exponential_ageing':
            self.select_authorities = self.select_authorities_by_exponential_ageing
        else:
            raise ValueError("Unknown selection strategy")
    
    def select_authorities_by_stake(self):
        # Strategy 1: Select top N validators based on stake
        return np.argsort(-self.stake_distribution)[:self.authority_count]
    
    def select_authorities_randomly(self):
        # Strategy 2: Random selection
        return np.random.choice(len(self.stake_distribution), self.authority_count, replace=False)
    
    def select_authorities_by_multiplicative_ageing(self):
        # Generate random hash for each validator (simulated with random numbers for simplicity)
        random_hashes = np.random.rand(len(self.stake_distribution))
        total_stake = np.sum(self.stake_distribution)
        # Calculate proofs based on the formula provided
        proofs = random_hashes * (self.stake_distribution / total_stake) * (1 + self.ageing)
        # Select top N validators based on proofs
        selected_indices = np.argsort(-proofs)[:self.authority_count]
        # Update ageing: increase by 1 for all, reset to 0 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 0
        return selected_indices
    
    def select_authorities_by_exponential_ageing(self):
        # Generate a random hash for each validator (simulated with random numbers for simplicity)
        random_hashes = np.random.rand(len(self.stake_distribution))
        total_stake = np.sum(self.stake_distribution)
        
        # Calculate proofs based on the modified formula with exponential ageing
        proofs = random_hashes * (self.stake_distribution / total_stake) * np.exp(self.ageing)
        
        # Select top N validators based on proofs
        selected_indices = np.argsort(-proofs)[:self.authority_count]
        
        # Update ageing: increase by 1 for all, reset to 0 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 0
        
        return selected_indices
    
    def distribute_rewards(self, authority_indices):
        total_stakes = np.sum(self.stake_distribution)
        authority_rewards = self.total_reward * self.reward_ratio[0]
        candidate_rewards = self.total_reward * self.reward_ratio[1]
        
        # Distribute rewards among authority validators
        authority_stakes = self.stake_distribution[authority_indices]
        for idx in authority_indices:
            self.stake_distribution[idx] += (authority_rewards * (self.stake_distribution[idx] / np.sum(authority_stakes)))
        
        # Distribute remaining rewards among candidate validators linearly
        candidate_indices = [i for i in range(len(self.stake_distribution)) if i not in authority_indices]
        for idx in candidate_indices:
            self.stake_distribution[idx] += (candidate_rewards * (self.stake_distribution[idx] / (total_stakes - np.sum(authority_stakes))))
    
    def simulate(self):
        for epoch in range(self.epochs):
            authority_indices = self.select_authorities()
            self.distribute_rewards(authority_indices)
            print(f"Epoch {epoch + 1}: Stake Distribution: {self.stake_distribution}")
    
# Run the simulation
simulator = PoSA_Simulator(STAKE_DISTRIBUTION, EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, REWARD_RATIO)
simulator.simulate()