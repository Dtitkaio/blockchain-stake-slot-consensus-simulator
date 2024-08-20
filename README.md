# Consensus Simulator for Proof of Stake and Authority (PoSA)

The PoSA simulator is a tool that simulates the changes in stake distribution when using different authority validator selection algorithms in a Proof of Stake and Authority (PoSA) consensus mechanism. This simulator is a tool to analyze how different validator selection strategies impact the network's decentralization.

## Purpose
The primary purpose of this simulator is to track the changes in validator stake over epochs and observe the impact that different validator selection strategies have on the stake distribution among validators.

## Configuration
The simulator's configuration is managed in the `config.py` file. The main configuration options are:

- `EPOCHS`: The number of epochs to simulate.
- `AUTHORITY_COUNT`: The number of authority validators per epoch.
- `TOTAL_REWARD`: The total reward amount per epoch.
- `STAKE_DISTRIBUTION`: The initial stake distribution.
- `SELECTION_STRATEGY`: The method to select authority validators. Available strategies include:
  - `stake`: Select the top validators based on their stake.
  - `linear`: Probabilistically select validators proportional to their stake.
  - `random`: Randomly select validators, regardless of their stake.
  - `multiplicative_ageing`: Select validators based on a combination of their stake and the duration they have not been selected.
  - `exponential_ageing`: Select validators based on a combination of their stake and the duration they have not been selected, with a higher emphasis on the ageing factor compared to `multiplicative_ageing`.
  - `binomial_ageing`: Select validators based on a binomial distribution of their stake and the duration they have not been selected.
- `OUTPUT_MODE`: The mode to output the simulation results. Available options are:
  - `console`: Display the results directly in the console.
  - `file`: Save the results to a CSV file in the `output/` directory, with a timestamp.
- `SEED`: The seed value for the random number generator used in the simulation.

## Usage
To run the simulation, follow these steps:

1. Install the required dependencies using the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```
2. Modify the `config.py` file to adjust the simulation settings as needed.
3. Run the simulator:
   ```
   python simulator.py
   ```
4. The output mode can be set to either file or console. If file output is selected, the results will be saved as a CSV file in the `output/` directory with a timestamp.

## Understanding the Simulation Results
The simulation results can be interpreted as follows:

- **File Output**: Open the CSV file saved in the `output/` directory. The file contains the simulation settings and the stake distribution at each epoch.
- **Console Output**: The console output displays the stake distribution at each epoch.
