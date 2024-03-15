import csv
import os
from datetime import datetime
import numpy as np
from scipy.stats import binom # type: ignore
from tqdm import tqdm # type: ignore
from config import EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, REWARD_RATIO, STAKE_DISTRIBUTION, SELECTION_STRATEGY, OUTPUT_MODE, SEED

class PoSA_Simulator:
    def __init__(self, stake_distribution, epochs, authority_count, total_reward, reward_ratio, selection_strategy='stake', output_mode='console', seed=None):
        self.stake_distribution = np.array(stake_distribution)
        self.selection_count = np.zeros_like(self.stake_distribution, dtype=int)
        self.epochs = epochs
        self.current_epoch = 0  # Initialize current epoch
        self.authority_count = authority_count
        self.ageing = np.zeros_like(self.stake_distribution)  # Initialize ageing for each validator
        self.total_reward = total_reward
        self.reward_ratio = reward_ratio
        self.selection_strategy = selection_strategy
        if self.selection_strategy == 'stake':
            self.select_authorities = self.select_authorities_by_stake
        elif self.selection_strategy == 'binomial_ageing':
            self.select_authorities = self.select_authorities_by_binomial_ageing
        elif self.selection_strategy == 'multiplicative_ageing':
            self.select_authorities = self.select_authorities_by_multiplicative_ageing
        elif self.selection_strategy == 'exponential_ageing':
            self.select_authorities = self.select_authorities_by_exponential_ageing
        elif self.selection_strategy == 'random':
            self.select_authorities = self.select_authorities_randomly
        else:
            raise ValueError("Unknown selection strategy")
        self.output_mode = output_mode
        if self.output_mode == 'file':
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_file_path = f'{output_dir}/simulation_results_{timestamp}.csv'
        else:
            self.output_file_path = None
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(len(self.stake_distribution))
    
    def select_authorities_by_stake(self):
        # Strategy 1: Select top N validators based on stake
        return np.argsort(-self.stake_distribution)[:self.authority_count]
    
    def select_authorities_randomly(self):
        # Strategy 2: Random selection
        return np.random.choice(len(self.stake_distribution), self.authority_count, replace=False)
    
    def select_authorities_by_multiplicative_ageing(self):
        # Strategy 3: Select top N validators based on proofs calculated using stake and multiplicative ageing
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
        # Strategy 4: Select top N validators based on proofs calculated using stake and exponential ageing
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
    
    def select_authorities_by_binomial_ageing(self):
        # Strategy 5: Select top N validators based on proofs calculated using stake and binomial ageing
        random_hashes = np.random.rand(len(self.stake_distribution))
        total_stake = np.sum(self.stake_distribution)
        proofs = np.zeros(len(self.stake_distribution))
        
        for i, stake in enumerate(self.stake_distribution):
            ciN = self.ageing[i] * AUTHORITY_COUNT  # c_i * N
            stake_ratio = stake / total_stake
            
            # Calculate the sum of binomial probabilities for the range [1, ciN]
            binom_sum = sum([binom.pmf(k, ciN, stake_ratio) for k in range(1, int(ciN) + 1)])
            
            proofs[i] = random_hashes[i] * binom_sum
        
        selected_indices = np.argsort(-proofs)[:self.authority_count]
        
        # Update ageing: increase by 1 for all, reset to 0 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 0
        
        return selected_indices
    
    def simple_distribute_rewards(self, authority_indices):
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
    
    def distribute_rewards(self, authority_indices):
        total_authority_stakes = np.sum(self.stake_distribution[authority_indices])
        
        # Calculating the rewards
        even_authority_rewards = self.total_reward * 0.5 / len(authority_indices)  # 50% evenly among authority validators
        proportional_authority_rewards = self.total_reward * 0.4  # 40% proportional to stakes among authority validators
        candidate_rewards = self.total_reward * 0.1  # 10% proportional to stakes among candidate validators
        
        # Distributing 50% of rewards evenly among authority validators
        for idx in authority_indices:
            self.stake_distribution[idx] += even_authority_rewards
        
        # Distributing 40% of rewards among authority validators based on their stakes
        for idx in authority_indices:
            self.stake_distribution[idx] += (proportional_authority_rewards * (self.stake_distribution[idx] / total_authority_stakes))
        
        # Distributing 10% of rewards among candidate validators based on their stakes
        candidate_indices = [i for i in range(len(self.stake_distribution)) if i not in authority_indices]
        total_candidate_stakes = np.sum(self.stake_distribution[candidate_indices])
        for idx in candidate_indices:
            self.stake_distribution[idx] += (candidate_rewards * (self.stake_distribution[idx] / total_candidate_stakes))

    
    def simulate(self):
        if self.output_mode == 'file':
            with open(self.output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write simulation settings header
                writer.writerow(['Simulation Settings'])
                writer.writerow(['Epochs', 'Authority Count', 'Total Reward', 'Reward Ratio', 'Selection Strategy'])
                writer.writerow([self.epochs, self.authority_count, self.total_reward, self.reward_ratio, self.selection_strategy])
                writer.writerow([])  # Blank row for separation
                # Write the data header
                validator_headers = [f"Validator{i+1}" for i in range(len(self.stake_distribution))]
                writer.writerow(['Epoch', *validator_headers])
                writer.writerow(['0', *self.stake_distribution])
                self._run_simulation(writer)
        else:
            print('Simulation Settings:')
            print(f'Epochs: {self.epochs}, Authority Count: {self.authority_count}, Total Reward: {self.total_reward}, Reward Ratio: {self.reward_ratio}, Selection Strategy: {self.selection_strategy}\n')
            print(f"Initial Stake Distribution: {self.stake_distribution}")
            self._run_simulation()

    def _run_simulation(self, writer=None):
        for epoch in tqdm(range(self.epochs), desc="Simulating Epochs"):
            authority_indices = self.select_authorities()
            self.distribute_rewards(authority_indices)
            self.selection_count[authority_indices] += 1
            if writer:
                writer.writerow([epoch + 1, *self.stake_distribution])
            else:
                print(f"Epoch {epoch + 1}: Stake Distribution: {self.stake_distribution}")
        
        if writer:
            writer.writerow([])  # Blank row for separation
            writer.writerow(['Final Selection Count', *self.selection_count])
        else:
            print("\n") # Blank line for separation
            print(f"Final Selection Count: {self.selection_count}")
    
# Run the simulation
simulator = PoSA_Simulator(STAKE_DISTRIBUTION, EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, REWARD_RATIO, selection_strategy=SELECTION_STRATEGY, output_mode=OUTPUT_MODE, seed=SEED)
simulator.simulate()