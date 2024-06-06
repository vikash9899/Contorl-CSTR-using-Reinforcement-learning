
# Control CSTR using Reinforcement Learning

## Overview

This project implements a Reinforcement Learning (RL) based control system for a Continuous Stirred Tank Reactor (CSTR). The CSTR is a common type of chemical reactor used in industrial processes. The goal of this project is to design an RL agent that can efficiently control the CSTR to achieve desired performance metrics.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Reinforcement Learning Approach](#reinforcement-learning-approach)
6. [Environment](#environment)
7. [Simulation](#simulation)
8. [Utilities](#utilities)
9. [Results](#results)
10. [Documentation](#documentation)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

## Introduction

A Continuous Stirred Tank Reactor (CSTR) is widely used in chemical engineering for its simplicity and ease of operation. However, controlling the CSTR is challenging due to its nonlinear dynamics. This project explores the use of reinforcement learning to develop a control strategy for the CSTR, aiming to maintain the reactor at optimal operating conditions.

## Project Structure

The project is organized into the following directories:

- `Env/`: Contains the environment code for the CSTR.
- `Simulation/`: Includes the simulator code for running experiments.
- `Utils/`: Utility scripts for metrics, plotting, and other helper functions.
- `docs/`: Documentation files for the project.
- `.git/`: Git version control directory.

### Directory and File Descriptions

#### Env/
- `env.py`: Defines the environment for the CSTR where the RL agent interacts.

#### Simulation/
- `simulator.py`: Contains the simulation logic for the CSTR, integrating the environment and the RL agent.

#### Utils/
- `metrics.py`: Provides functions to calculate performance metrics.
- `plotting.py`: Scripts for plotting results and visualizations.
- `random_sa.py`: Random search algorithms for hyperparameter tuning.
- `utils.py`: General utility functions used across the project.

#### docs/
- Sphinx-generated documentation for the project.

## Installation

To run this project, you need to have Python installed along with several dependencies. The recommended way to install the dependencies is to use a virtual environment.

### Step-by-Step Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/vikash9899/Contorl-CSTR-using-Reinforcement-learning.git

    cd Control-CSTR-using-Reinforcement-Learning
    ```

2. **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

After installing the dependencies, you can run the simulations and train the RL agent.

### Running the Simulation

1. **Navigate to the Simulation Directory**:
    ```bash
    cd Simulation
    ```

2. **Run the Simulator**:
    ```bash
    python simulator.py
    ```

## Reinforcement Learning Approach

The RL approach used in this project involves training an agent to learn the optimal policy for controlling the CSTR. The agent interacts with the environment, receives rewards based on its actions, and updates its policy accordingly.

### Key Components

- **State**: Represents the current condition of the CSTR.
- **Action**: The control input provided by the RL agent.
- **Reward**: A scalar feedback signal used to guide the learning process.
- **Policy**: The strategy used by the agent to decide actions based on states.

## Environment

The environment (`env.py`) defines the interaction between the RL agent and the CSTR. It includes the state space, action space, reward function, and dynamics of the CSTR.

### Environment Details

- **State Space**: Variables representing the current status of the reactor (e.g., concentration, temperature).
- **Action Space**: Possible control actions (e.g., adjusting flow rates, temperature settings).
- **Reward Function**: Designed to encourage desired behaviors such as stability, efficiency, and safety.

## Simulation

The simulation (`simulator.py`) integrates the environment and the RL agent, allowing for training and evaluation. It handles the initialization, execution of episodes, and data collection for analysis.

### Key Features

- **Episode Management**: Running multiple episodes for training and testing.
- **Data Logging**: Collecting data on states, actions, rewards, and performance metrics.
- **Visualization**: Plotting the results for analysis and interpretation.

## Utilities

The `Utils/` directory contains helper functions and scripts to support the main codebase.

### Metrics

`metrics.py` provides functions to evaluate the performance of the RL agent, such as calculating cumulative rewards and stability measures.

### Plotting

`plotting.py` includes scripts to visualize the results, such as state trajectories, reward curves, and action distributions.

### Random Search

`random_sa.py` implements random search algorithms for hyperparameter tuning, helping to find the best settings for the RL agent.

### General Utilities

`utils.py` contains general-purpose functions used throughout the project, such as data normalization, logging, and configuration handling.

## Results

The results of the experiments, including trained models and performance metrics, are stored in the `results/` directory. Key findings and visualizations are documented to provide insights into the effectiveness of the RL-based control strategy.

## Documentation

Comprehensive documentation is provided in the `docs/` directory, generated using Sphinx. It includes detailed descriptions of the project components, installation instructions, usage guides, and API references.

### Building the Documentation

To build the documentation locally, navigate to the `docs/` directory and run:
```bash
make html
```
The generated HTML files will be available in `docs/_build/html/`.

## Contributing

Contributions to the project are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

### Contribution Guidelines

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

This project builds upon numerous open-source libraries and research contributions in the fields of reinforcement learning and chemical process control. We extend our gratitude to the contributors and maintainers of these projects.

---
<!-- 
This `README.md` provides an extensive overview of the project, covering all essential aspects from installation to usage, and from project structure to contributing guidelines. This should help users and developers understand and engage with the project effectively. -->