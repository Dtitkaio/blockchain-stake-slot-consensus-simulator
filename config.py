# config.py

# Initial stakes for each validator: 50
STAKE_DISTRIBUTION = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# Number of epochs to simulate
EPOCHS = 1000

# Number of authority validators selected each epoch
AUTHORITY_COUNT = 20

# Total reward distributed per epoch
TOTAL_REWARD = 100

# Selection strategy for authority validators
SELECTION_STRATEGY = 'exponential_ageing' # 'stake', 'linear', 'multiplicative_ageing', 'binomial_ageing', 'exponential_ageing', 'random'

# Output mode for simulation results
OUTPUT_MODE = 'file'  # 'console', 'file'

# Seed for random number generation
SEED = 0