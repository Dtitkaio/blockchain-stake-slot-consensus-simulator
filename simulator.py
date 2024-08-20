import csv
import os
from datetime import datetime
import numpy as np
from scipy.stats import binom # type: ignore
from tqdm import tqdm # type: ignore
from config import EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, STAKE_DISTRIBUTION, SELECTION_STRATEGY, OUTPUT_MODE, SEED

class PoSA_Simulator:
    """
    A simulator for the Proof of Stake Authority (PoSA) consensus mechanism. This simulator
    models the process of selecting authority validators, distributing rewards, and tracking
    the evolution of stake distribution over multiple epochs.

    Attributes:
        stake_distribution (np.array): An array representing the initial stake of each validator.
        selection_count (np.array): An array tracking the number of times each validator has been selected as an authority validator.
        epochs (int): The total number of epochs to simulate.
        current_epoch (int): Tracks the current epoch in the simulation.
        authority_count (int): The number of validators to be selected as authorities in each epoch.
        ageing (np.array): An array tracking the number of epochs since each validator was last selected as an authority.
        total_reward (float): The total reward distributed to validators in each epoch.
        selection_strategy (str): The strategy used for selecting authority validators. Possible values
                                   include 'stake', 'binomial_ageing', 'multiplicative_ageing',
                                   'exponential_ageing', and 'random'.
        selection_count (np.array): An array counting how many times each validator has been selected as an authority.
        output_mode (str): Determines whether the simulation results are written to a file or printed to the console.
        output_file_path (str): The file path for saving the simulation results, applicable if output_mode is 'file'.

    Methods:
        simulate(): Runs the simulation for the specified number of epochs, tracking the stake distribution
                    and selection count, and outputs the results according to the specified mode.
        _run_simulation(writer=None): A helper method to conduct the simulation, updating stake distribution and
                                       selection counts, and writing or printing the results after each epoch.
    """

    def __init__(self, stake_distribution, epochs, authority_count, total_reward, selection_strategy='stake', output_mode='console', seed=None):
        self.stake_distribution = np.array(stake_distribution)
        self.selection_count = np.zeros_like(self.stake_distribution, dtype=int)
        self.epochs = epochs
        self.current_epoch = 0  # Initialize current epoch
        self.authority_count = authority_count
        self.ageing = np.ones_like(self.stake_distribution)  # Initialize ageing for each validator
        self.total_reward = total_reward
        self.selection_strategy = selection_strategy
        if self.selection_strategy == 'stake':
            self.select_authorities = self.select_authorities_by_stake
        elif self.selection_strategy == 'linear':
            self.select_authorities = self.select_authorities_by_stake_random
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
            self.output_file_path = f'{output_dir}/{self.selection_strategy}_simulation_results_{timestamp}.csv'
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
    
    def select_authorities_by_stake_random(self):
        # Strategy 2: Select top N validators randomly in proportion to their stake
        random_hashes = np.random.rand(len(self.stake_distribution))
        total_stake = np.sum(self.stake_distribution)
        # Calculate proofs based on stake and random hashes
        proofs = random_hashes * (self.stake_distribution / total_stake)
        # Select top N validators based on proofs
        selected_indices = np.argsort(-proofs)[:self.authority_count]
        return selected_indices
    
    def select_authorities_by_multiplicative_ageing(self):
        # Strategy 3: Select top N validators based on proofs calculated using stake and multiplicative ageing
        # Generate random hash for each validator (simulated with random numbers for simplicity)
        random_hashes = np.random.rand(len(self.stake_distribution))
        total_stake = np.sum(self.stake_distribution)
        # Calculate proofs based on the formula provided
        proofs = random_hashes * (self.stake_distribution / total_stake) * self.ageing
        # Select top N validators based on proofs
        selected_indices = np.argsort(-proofs)[:self.authority_count]
        # Update ageing: increase by 1 for all, reset to 1 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 1
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
        
        # Update ageing: increase by 1 for all, reset to 1 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 1
        
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
        
        # Update ageing: increase by 1 for all, reset to 1 for selected validators
        self.ageing += 1
        self.ageing[selected_indices] = 1
        
        return selected_indices
    
    def select_authorities_randomly(self):
        # Strategy 6: Random selection
        return np.random.choice(len(self.stake_distribution), self.authority_count, replace=False)
    
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
                writer.writerow(['Epochs', 'Authority Count', 'Total Reward', 'Reward Ratio', 'Selection Strategy', 'Seed'])
                writer.writerow([self.epochs, self.authority_count, self.total_reward, self.reward_ratio, self.selection_strategy, self.seed])
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
            writer.writerow([])  # Blank row for separation
            self.write_indices_to_output(writer)
        else:
            print("\n") # Blank line for separation
            print(f"Final Selection Count: {self.selection_count}")
            print("\n") # Blank line for separation
            self.write_indices_to_output()
    
    def calculate_percentiles(self, stakes):
        return np.percentile(stakes, [25, 50, 75, 90])

    def calculate_standard_deviation(self, stakes):
        return np.std(stakes)

    def calculate_gini_coefficient(self, stakes):
        # Sort stakes in ascending order
        sorted_stakes = np.sort(stakes)
        n = len(stakes)
        cumsum_stakes = np.cumsum(sorted_stakes)
        index = np.arange(1, n + 1)
        # Calculate Gini coefficient assuming sorted_stakes is populated in ascending order and values are positive.
        # Reference: https://kimberlyfessel.com/mathematics/applications/gini-use-cases/
        gini = (np.sum((2 * index - n - 1) * sorted_stakes)) / (n * np.sum(sorted_stakes))
        return gini

    def calculate_nakamoto_coefficient(self, stakes):
        sorted_stakes = np.sort(stakes)[::-1]
        cumulative_stakes = np.cumsum(sorted_stakes) / np.sum(stakes)
        return np.where(cumulative_stakes > 0.5)[0][0] + 1

    def calculate_entropy(self, stakes):
        total_stake = np.sum(stakes)
        stake_ratios = stakes / total_stake
        # Ensure nonzero values for log calculation
        stake_ratios = stake_ratios[stake_ratios > 0]
        entropy = -np.sum(stake_ratios * np.log(stake_ratios))
        return entropy

    def write_indices_to_output(self, writer=None):
        percentiles = self.calculate_percentiles(self.stake_distribution)
        std_dev = self.calculate_standard_deviation(self.stake_distribution)
        gini_coeff = self.calculate_gini_coefficient(self.stake_distribution)
        nakamoto_coeff = self.calculate_nakamoto_coefficient(self.stake_distribution)
        entropy_val = self.calculate_entropy(self.stake_distribution)

        selection_count_std_dev = self.calculate_standard_deviation(self.selection_count)
        selection_count_percentiles = self.calculate_percentiles(self.selection_count)

        if writer:
            writer.writerow(['Decentralization Indices', '25th', '50th', '75th', '90th', 'Std Dev', 'Gini', 'Nakamoto', 'Entropy', 'Selection Count Std Dev', 'Selection count 25th', '50th', '75th', '90th'])
            writer.writerow(['Values', *percentiles, std_dev, gini_coeff, nakamoto_coeff, entropy_val, selection_count_std_dev, *selection_count_percentiles])
        else:
            print(f"Decentralization Indices:\nPercentiles (25th, 50th, 75th, 90th): {percentiles}\nStandard Deviation: {std_dev}\nGini Coefficient: {gini_coeff}\nNakamoto Coefficient: {nakamoto_coeff}\nEntropy: {entropy_val}\nSelection Count Standard Deviation: {selection_count_std_dev}\nSelection Count Percentiles (25th, 50th, 75th, 90th): {selection_count_percentiles}")

    
# Run the simulation
simulator = PoSA_Simulator(STAKE_DISTRIBUTION, EPOCHS, AUTHORITY_COUNT, TOTAL_REWARD, selection_strategy=SELECTION_STRATEGY, output_mode=OUTPUT_MODE, seed=SEED)
simulator.simulate()