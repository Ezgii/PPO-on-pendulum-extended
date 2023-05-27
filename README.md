### Project description
An implementation of the PPO algorithm written in Python using Pytorch. Recurrence is added to the ActorCritic network to train in the environment with partial observability where obs = [cos(theta), sin(theta)]. An ensemble of 5 critics is used to increase stability.

### Pseudo code:

![pseudocode](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/pseudocode.png)

### Environment
[OpenAI's Gym](https://gym.openai.com/) is a framework for training reinforcement 
learning agents. It provides a set of environments and a
standardized interface for interacting with those.   
In this project, I used the Pendulum environment from gym.

### Installation

#### Using conda (recommended)    
1. [Install Anaconda](https://www.anaconda.com/products/individual)

2. Create the env    
`conda create a1 python=3.8` 

3. Activate the env     
`conda activate a1`    

4. install torch ([steps from pytorch installation guide](https://pytorch.org/)):    
- if you don't have an nvidia gpu or don't want to bother with cuda installation:    
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`    
  
- if you have an nvidia gpu and want to use it:    
[install cuda](https://docs.nvidia.com/cuda/index.html)   
install torch with cuda:   
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. other dependencies   
`conda install -c conda-forge matplotlib gym opencv pyglet`

#### Using pip
`python3 -m pip install -r requirements.txt`

### How to run the code
On terminal, write:

`python3 main.py`

### Results

#### Loss functions and learning curve:

![results](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/results/figure1.png)

#### Testing Angle vs Time:

![test](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/results/figure2.png)



