# config.py

# Initial stakes for each validator
STAKE_DISTRIBUTION = [100, 200, 300, 400, 500]

# Number of epochs to simulate
EPOCHS = 10

# Number of authority validators selected each epoch
AUTHORITY_COUNT = 3

# Reward distribution ratio between authority and candidate validators
REWARD_RATIO = [0.8, 0.2]

# Total reward distributed per epoch
TOTAL_REWARD = 10

# Selection strategy for authority validators
SELECTION_STRATEGY = 'binomial_ageing' # 'stake', 'multiplicative_ageing', 'binomial_ageing', 'exponential_ageing', 'random'

# Output mode for simulation results
OUTPUT_MODE = 'console'  # 'console', 'file'