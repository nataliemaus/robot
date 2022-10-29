# Rank-Ordered Bayesian Optimization with Trust-regions (ROBOT)
Official implementation of ROBOT method from the paper Discovering Many Diverse Solutions with Bayesian Optimization https://arxiv.org/abs/2210.10953. This repository includes base code to run ROBOT on all tasks from the paper. 

## Weights and Biases (wandb) tracking
This repo is set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
By default, the code is run without wandb tracking. After creating an account, wandb tracking can be used for optimization runs by simply adding the following args `--track_with_wandb True --wandb_entity nmaus` (see example commands below). 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Environment Setup (Docker)
All optimization tasks can be run in the docker envioronment defined in docker/Dockerfile. Build the docker env using the following steps. 

1. If you do not have a docker account, create one here:
https://hub.docker.com/signup

2. If you would like to use wandb tracking, add your wandb API key to the docker file by adding the following line to docker/Dockerfile: 

```Bash
ENV WANDB_API_KEY=$YOUR_WANDB_API_KEY
```

3. Build the docker file: 

```Bash
cd docker 
docker build -t $YOUR_DOCKER_USER_NAME/robot .
```

The resultant docker environment will have all imports necessary to run ROBOT on all tasks from the paper.

## Running ROBOT

To start a ROBOT run on one of the three example continuous optimization tasks from the paper, run `scripts/continuous_space_optimization.py` with desired command line arguments. To start a structured molecule optimization run with ROBOT, run `scripts/molecule_optimization.py` with desired arguments. 

To get a list of command line args specifically for the continuous space optimization tasks, run the following: 

```Bash
cd scripts/
python3 continuous_space_optimization.py -- --help
```

To get a list of command line args specifically for the molecule optimization tasks with the SELFIES VAE, run the following: 

```Bash
cd scripts/
python3 molecule_optimization.py -- --help
```

For a list of all remaining possible args that are the more general ROBOT args (not specific to task) run the following:

```Bash
cd scripts/
python3 optimize.py -- --help
```

### Task IDs
#### Molecule Task IDs
This code base provides support for 13 Molecules Optimization Tasks, 12 from the GuacaMol Benchmark Suit and Penalized log P.

| task_id | Full Task Name     |
|---------|--------------------|
|  med1   | Median molecules 1 |
|  med2   | Median molecules 2 |
|  pdop   | Perindopril MPO    |
|  osmb   | Osimertinib MPO    |
|  adip   | Amlodipine MPO     |
|  siga   | Sitagliptin MPO    |
|  zale   | Zaleplon MPO       |
|  valt   | Valsartan SMARTS   |
|  dhop   | Deco Hop           |
|  shop   | Scaffold Hop       |
|  rano   | Ranolazine MPO     |
|  fexo   | Fexofenadine MPO   |
|  logP   | Penalized Log P    |

The original ROBOT paper features results on valt, siga, and logp. For descriptions of these and the other GuacaMol objectives listed, as well as a leaderboard for each of GuacaMol task, see https://www.benevolent.com/guacamol

#### Other Task IDs
The code provides support for three other continuous space optimization tasks. These are the three continuous optimization tasks the original ROBOT paper (see paper for more detialed descriptions of each). 

| task_id | Full Task Name     |
|---------|--------------------|
|  rover  | Rover Trajectory Optimization       |
|  lunar  | Lunar Lander Policy Optimization    |
|  stocks | S&P500 Stock Portfolio Optimization |

## Commands to Reproduce Paper Results
In this section we provide commands that can be used to start a ROBOT optimization and reproduce results from the paper. 

### Run ROBOT on Rover Optimization Task
##### (Replicates Result in Figure 2 in Paper)
```Bash 
cd scripts/
python3 continuous_space_optimization.py --task_id rover \
--max_n_oracle_calls 100000 --bsz 32 \
--M 3 --tau 0.15 - run_robot - done 
```

### Run ROBOT on Lunar Lander Task 
##### (Replicates Result in Figure 3 in Paper)
```Bash
python3 continuous_space_optimization.py --task_id lunar \
--max_n_oracle_calls 100000 --bsz 32 \
--M 20 --tau 0.6 - run_robot - done 
```

### Run ROBOT on S&P 500 Stock Portfolio Optimization Task
##### (Replicates Result in Figure 6 in Paper)
```Bash
python3 continuous_space_optimization.py --task_id stocks \
--max_n_oracle_calls 3500000 --bsz 50 \
--M 3 --tau 10 - run_robot - done 
```

### Run LOL-ROBOT on Molecule Optimization Tasks
To replicate the result in Figure 4 of the paper, simply run the below command with `--task_id valt`, and again with `--task_id siga`, once with each combination of M and tau shown in the paper (M=20,50,100 with tau=-0.53, M=5 with tau=-0.4). 

```Bash
python3 molecule_optimization.py --task_id valt \
--max_n_oracle_calls 350000 --bsz 10 \
--M 50 --tau -0.53 - run_robot - done 
```
