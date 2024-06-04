# config.py

# Initial stakes for each validator: 50
# 1600:25, 100: 25
# Olygopoly: 100: 20, 50: 20
# STAKE_DISTRIBUTION = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
#                       200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
STAKE_DISTRIBUTION = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                      400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# 4 times: 400:5, 100:35
# STAKE_DISTRIBUTION = [800, 800, 800, 800, 800, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# Uniform: 100: 40
# STAKE_DISTRIBUTION = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
#                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# Number of epochs to simulate
EPOCHS = 1000

# Number of authority validators selected each epoch
AUTHORITY_COUNT = 20

# Reward distribution ratio between authority and candidate validators
REWARD_RATIO = [0.8, 0.2]

# Total reward distributed per epoch
TOTAL_REWARD = 100

# Selection strategy for authority validators
SELECTION_STRATEGY = 'exponential_ageing' # 'stake', 'linear', 'multiplicative_ageing', 'binomial_ageing', 'exponential_ageing', 'random'

# Output mode for simulation results
OUTPUT_MODE = 'file'  # 'console', 'file'

# Seed for random number generation
SEEDS = [0, 10, 100]