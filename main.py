import gym
import a3_gym_env
import collections
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.distributions import MultivariateNormal


# sample hyperparameters
num_timesteps = 200 # T
num_trajectories = 10 # N
num_iterations = 250
epochs = 100

batch_size = 10
learning_rate = 1e-4
eps = 0.2 # clipping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
RNN reference design:
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
'''
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=2, activation=nn.functional.relu):
        super(Net, self).__init__()

        # Readout layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Initialize hidden state with zeros
        self.hidden = torch.zeros(num_layers, hidden_size)

        self.act = activation

    def forward(self, x):

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, self.hidden = self.rnn_layer(x, self.hidden.detach())
        
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        
        return out



class ReplayMemory():
    def __init__(self, batch_size=10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size

    def push(self, state, action, reward, reward_togo, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rewards_togo.append(reward_togo)
        self.advantages.append(advantage)
        self.values.append(value)  
        self.log_probs.append(log_prob)

    def sample(self):
        num_states = len(self.states)
        batch_start = torch.arange(0, num_states, self.batch_size)
        indices = torch.randperm(num_states)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (torch.tensor(self.states), 
                torch.tensor(self.actions), 
                torch.tensor(self.rewards),
                torch.tensor(self.rewards_togo),
                torch.tensor(self.advantages),
                torch.tensor(self.values),
                torch.tensor(self.log_probs), 
                batches)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []

       

# function to calculate the (discounted) reward-to-go from a sequence of rewards
def calc_reward_togo(rewards, gamma=0.99):
    n = len(rewards)
    reward_togo = np.zeros(n)
    reward_togo[-1] = rewards[-1]
    for i in reversed(range(n-1)):
        reward_togo[i] = rewards[i] + gamma * reward_togo[i+1]

    reward_togo = torch.tensor(reward_togo, dtype=torch.float)
    return reward_togo


# compute advantage estimates (as done in PPO paper)
def calc_advantages(rewards, values, gamma=0.99, lambda_=1):
    advantages = torch.zeros_like(torch.as_tensor(rewards))
    sum = 0
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        sum = delta + gamma * lambda_ * sum
        advantages[t] = sum
    
    return advantages


class PPO:
    def __init__(self, gamma=0.9):

        self.policy_net = Net(input_size=2)
        self.critic1_net = Net(input_size=2)
        self.critic2_net = Net(input_size=2)
        self.critic3_net = Net(input_size=2)
        self.critic4_net = Net(input_size=2)
        self.critic5_net = Net(input_size=2)

        self.env = gym.make('Pendulum-v1-custom')

        #self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        #self.critic_opt = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)

        self.optimizer = torch.optim.Adam([  # Update both models together
                {'params': self.policy_net.parameters(), 'lr': learning_rate},
                {'params': self.critic1_net.parameters(), 'lr': learning_rate},
                {'params': self.critic2_net.parameters(), 'lr': learning_rate},
                {'params': self.critic3_net.parameters(), 'lr': learning_rate},
                {'params': self.critic4_net.parameters(), 'lr': learning_rate},
                {'params': self.critic5_net.parameters(), 'lr': learning_rate}
                        ])
    
        self.memory = ReplayMemory(batch_size)

        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 1  # c1
        self.entropy_coef = 0.01  # c2

        # use fixed std
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5))



    def generate_trajectory(self):
        
        obs = self.env.reset()
        current_state = torch.tensor([obs[:2]])

        states = []
        actions = []
        rewards = []
        log_probs = []
        

        # Run the old policy in environment for num_timestep            
        for t in range(num_timesteps):
            
            # compute mu(s) for the current state
            mean = self.policy_net(current_state)

            # the gaussian distribution
            normal = MultivariateNormal(mean, self.std)

            # sample an action from the gaussian distribution
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            states.append(obs.flatten()[:2])

            # emulate taking that action
            obs, reward, done, info = self.env.step(action.tolist())
            next_state = torch.tensor([obs.flatten()[:2]])

            # store results in a list
            actions.append(action)
            rewards.append(torch.as_tensor(reward))
            log_probs.append(log_prob)
            
            #env.render()

            current_state = next_state
        
      
        # calculate reward to go
        rtg = calc_reward_togo(torch.as_tensor(rewards), self.gamma)


        # calculate values
        v1 = self.critic1_net(torch.as_tensor(states)).squeeze()
        v2 = self.critic2_net(torch.as_tensor(states)).squeeze()
        v3 = self.critic3_net(torch.as_tensor(states)).squeeze()
        v4 = self.critic4_net(torch.as_tensor(states)).squeeze()
        v5 = self.critic5_net(torch.as_tensor(states)).squeeze()
        values = (v1 + v2 + v3 + v4 + v5)/5
        

        # calculate advantages
        advantages = calc_advantages(rewards, values.detach(), self.gamma, self.lambda_)
        

        # save the transitions in replay memory
        for t in range(len(rtg)):
            self.memory.push(states[t], actions[t], rewards[t], rtg[t], advantages[t], values[t], log_probs[t])
  
        #env.close()


    def train(self):
        
        train_actor_loss = []
        train_critic_loss = []
        train_total_loss = []
        train_reward = []

        for it in range(num_iterations): # k

            # collect a number of trajectories and save the transitions in replay memory
            for _ in range(num_trajectories):
                self.generate_trajectory()

            # sample from replay memory
            states, actions, rewards, rewards_togo, advantages, values, log_probs, batches = self.memory.sample()

            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            reward_list = []
            for _ in range(epochs):

                # calculate the new log prob
                mean = self.policy_net(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)
                clipped_r = torch.clamp(r, 1 - eps, 1 + eps)


                new_v1 = self.critic1_net(states).squeeze()
                new_v2 = self.critic2_net(states).squeeze()
                new_v3 = self.critic3_net(states).squeeze()
                new_v4 = self.critic4_net(states).squeeze()
                new_v5 = self.critic5_net(states).squeeze()
                new_values = (new_v1 + new_v2 + new_v3 + new_v4 + new_v5)/5

                returns = (advantages + values).detach()

                actor_loss = (-torch.min(r * advantages, clipped_r * advantages)).mean()
                critic_loss = nn.MSELoss()(new_values.float(), returns.float())
           
                # Calcualte total loss
                total_loss = actor_loss + (self.vf_coef * critic_loss) - (self.entropy_coef * normal.entropy().mean())

                # update policy and critic network
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            # clear replay memory
            self.memory.clear()

            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            print("it = ", it)
            print('Actor loss = ', avg_actor_loss)
            print('Critic loss = ', avg_critic_loss)
            print('Total Loss = ', avg_total_loss)
            print('Reward = ', avg_reward)
            print("")

        # save the networks
        torch.save(self.policy_net.state_dict(), f'./results/policy_net.pt')
        torch.save(self.critic1_net.state_dict(), f'./results/critic1_net.pt')
        torch.save(self.critic2_net.state_dict(), f'./results/critic2_net.pt')
        torch.save(self.critic3_net.state_dict(), f'./results/critic3_net.pt')
        torch.save(self.critic4_net.state_dict(), f'./results/critic4_net.pt')
        torch.save(self.critic5_net.state_dict(), f'./results/critic5_net.pt')
        

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].plot(range(len(train_actor_loss)), train_actor_loss, 'r', label='Actor Loss')
        axes[0].set_title('Actor Loss', fontsize=18)

        axes[1].plot(range(len(train_critic_loss)), train_critic_loss, 'b', label='Critic Loss')
        axes[1].set_title('Critic Loss', fontsize=18)

        axes[2].plot(range(len(train_total_loss)), train_total_loss, 'm', label='Total Loss')
        axes[2].set_title('Total Loss', fontsize=18)

        axes[3].plot(range(len(train_reward)), train_reward, 'orange', label='Accumulated Reward')
        axes[3].set_title('Accumulated Reward', fontsize=18)

        
        fig.tight_layout()
        plt.savefig(f'./results/figure1.png')
        fig.show()

        

    def test(self):

        self.policy_net.load_state_dict(torch.load(f'./results/policy_net.pt'))

        obs = self.env.reset()
        current_state = obs[:2]

        angle_list = []
        
        for i in range(200):

            # compute mu(s) for the current state
            mean = self.policy_net(torch.as_tensor(current_state).squeeze().unsqueeze(0))

            # the gaussian distribution
            normal = MultivariateNormal(mean, self.std)

            # sample an action from the gaussian distribution
            action = normal.sample().detach().numpy()

            # save the state in a list
            angle_list.append(np.arccos(current_state[0].item()))

            # emulate taking that action
            obs, reward, done, info = self.env.step(action.tolist())
            next_state = obs[:2]

            self.env.render(mode="human")

            current_state = next_state

        self.env.close()


        fig = plt.figure(figsize=(5, 5))
        plt.plot(range(len(angle_list)), angle_list, 'r')
        plt.title('Angle VS Time', fontsize=18)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Angle', fontsize=18)
        
        plt.savefig(f'./results/figure2.png')
        fig.show()

            


 
if __name__ == '__main__':

    user_input = input("Press 0 to run test only.\nPress 1 to run training + test.\n")

    agent = PPO()

    if user_input == '1':
        agent.train()

        
    agent.test()
