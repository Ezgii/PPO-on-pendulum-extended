# PPO-in-pendulum-extented

An implementation of the PPO algorithm written in Python using Pytorch. Recurrence is added to the ActorCritic network to train in the environment with partial observability where obs = [cos(theta), sin(theta)]. An ensemble of 5 critics is used to increase stability.

### Pseudo code:

![pseudocode](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/pseudocode.png)


### Loss functions and learning curve:

![results](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/results/figure1.png)


### Testing Angle vs Time:

![test](https://github.com/Ezgii/PPO-on-pendulum-extented/blob/main/results/figure2.png)



