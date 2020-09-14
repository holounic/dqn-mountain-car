import gym

import copy
import random
import numpy as np
import math
from collections import deque

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
%matplotlib inline

device = torch.device("cuda")
np.random.seed(22)

def calc_reward(prev_state, state, gamma=0.98):
  prev_position, prev_velocity = prev_state
  position, velocity = state
  reward = -1
  if position >= 0.5:
    reward = 0
  reward += 350 * (gamma * abs(velocity) - abs(prev_velocity))
  mul = reward / abs(reward)
  return mul * math.log(abs(reward) + 1)

def calc_rewards(prev_states, states):
  return [calc_reward(prev_states[i], states[i]) \
                      for i in range(len(prev_states))]

class DQN:
  def __init__(self, gamma=0.98, learning_rate=0.00001, 
               update_targer_every=200, device=torch.device("cpu")):
    self.device = device

    self.model = nn.Sequential(
        nn.Linear(2, 32), 
        nn.LayerNorm(32), 
        nn.ReLU(), 
        nn.Linear(32, 32), 
        nn.LayerNorm(32), 
        nn.ReLU(), 
        nn.Linear(32, 3)
    )

    self.target_model = copy.deepcopy(self.model)

    self.target_model.train()
    self.model.train()

    self.target_model.to(self.device)
    self.model.to(self.device)

    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.update_target_every = update_targer_every
    self.updates = 0
    self.gamma = gamma
    self.threshold = 0.1

  def act(self, state):
    state = torch.tensor(state).to(self.device).float()
    return self.target_model(state).max(0)[1].item()

  def update(self, batch):
    prev_states, actions, states, dones = batch
    rewards = calc_rewards(prev_states, states)
    prev_states = torch.tensor(prev_states).to(self.device).float()
    actions = torch.tensor(actions).to(self.device)
    states = torch.tensor(states).to(self.device).float()
    rewards = torch.tensor(rewards).to(self.device).float()
    
    with torch.no_grad():
      target_Q_s_a = self.target_model(states).max(dim=1)[0]
      target_Q_s_a[dones] = 0

    target_Q_s_a = target_Q_s_a * self.gamma + rewards
    pred_Q_s_a = self.model(prev_states) \
                     .gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(pred_Q_s_a, target_Q_s_a)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.updates += 1

    if self.updates % self.update_target_every == 0:
      self.target_model = copy.deepcopy(self.model)

class DQNUpdater:
  def __init__(self, dqn, buffer_size=5000, batch_size=128):
    self.memory_buffer = deque(maxlen=buffer_size)
    self.dqn = dqn
    self.batch_size = batch_size

  def update(self, trajectory):
    for transition in trajectory:
      self.memory_buffer.append(transition)
      if len(self.memory_buffer) > self.batch_size:
        batch = random.sample(self.memory_buffer, self.batch_size)
        batch = list(zip(*batch))
        self.dqn.update(batch)

class EpsilonGreedyAgent:
  def __init__(self, inner_agent, epsilon_max = 0.5, 
               epsilon_min = 0.2, decay_steps=20000):
    self.inner_agent = inner_agent
    self.epsilon_max = epsilon_max
    self.epsilon_min = epsilon_min
    self.decay_steps = decay_steps
    self.steps = 0

  def act(self, state):
    self.steps = min(self.decay_steps, self.steps + 1)
    epsilon = self.epsilon_max +\
              (self.epsilon_min - self.epsilon_max) * \
              self.steps / self.decay_steps
    if epsilon > np.random.random():
      return np.random.randint(0, 3)
    else:
      return self.inner_agent.act(state)

def play_episode(env, agent):
  state = env.reset()
  done = False
  trajectory = []
  steps = 0
  while not done:
    action = agent.act(state)
    new_state, _, done, _ = env.step(action)
    steps += 1
    trajectory.append((state, action, new_state, done))
    state = new_state
  return trajectory, steps

def train_agent(env, exploration_agent, exploitation_agent, \
                updater, env_steps=20000, exploit_every=20000):
  steps_count = 0
  exploits = 0
  steps = []
  solved_in = []
  while steps_count < env_steps:
    if exploits * exploit_every <= steps_count:
      exploits += 1
      _, steps_per_episode = play_episode(env, exploitation_agent)
      solved_in.append(steps_per_episode)
      steps.append(steps)
    else:
      trajectory, steps_per_episode = play_episode(env, exploration_agent)
      steps_count += len(trajectory)
      updater.update(trajectory)
  _, steps_per_episode = play_episode(env, exploitation_agent)
  solved_in.append(steps_per_episode)
  steps.append(steps_count)
  return rewards, steps

env = gym.make("MountainCar-v0")
max_steps = 280000
steps_per_episode = 200
dqn_agent = DQN(device=device)
updater = DQNUpdater(dqn_agent, buffer_size=max_steps)
epsilon_greedy = EpsilonGreedyAgent(dqn_agent, decay_steps=max_steps)
rewards, steps = train_agent(env, epsilon_greedy, dqn_agent, updater, \
                             env_steps=max_steps, \
                             exploit_every=max_steps//steps_per_episode)