# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:18:05 2020

User can pass an agent to this script, so that it is called every time.

@author: guido
"""

import gym
import math
import numpy as np
import gym.envs.classic_control as cc
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import time
from IPython import display
import os
from gym.utils import seeding
import random

class CMC_original(cc.MountainCarEnv):
    """ Derived class of Mountain Car, just modified the rendering function.
    """
    # Based on: https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/mountain_car.py

    n_steps_per_render = 5

    def __init__(self):
        self.render_step = 0
        self.figure_handle = []
        super(CMC_original, self).__init__()
    
    def reset(self):
        self.render_step = 0
        super(CMC_original, self).reset()
        if(self.figure_handle != []):
            plt.close('mountain_car')
            self.figure_handle = []
        return np.array(self.state)

    def render(self, mode='human', sleep_time=0.033):
        
        if os.name == 'nt':
            normal_display = True
        else:
            if 'DISPLAY' in os.environ.keys():
                normal_display = True
            else:
                normal_display = False
            
        if normal_display:
            super(CMC_original, self).render()
        else:
            self.render_step += 1
            if(self.render_step % self.n_steps_per_render == 0):
                # first plot the landscape:
                step = 0.01
                x_coords = np.arange(self.min_position, self.max_position, step)
                y_coords = self._height(x_coords)
                
                if(self.figure_handle == []):
                    self.figure_handle = plt.figure('mountain_car')
                    self.ax = self.figure_handle.add_subplot(111)
                    plt.ion()
                    #self.figure_handle.show()
                    self.figure_handle.canvas.draw()
                else:
                    plt.figure('mountain_car')
                
                self.ax.clear()
                self.ax.plot(x_coords, y_coords)
                self.ax.plot(self.state[0], self._height(self.state[0]), 'ro')
                self.ax.text(self.goal_position, self._height(self.goal_position)+0.02, 'Goal')        
                #        self.figure_handle.canvas.draw()
                #        self.figure_handle.show()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(sleep_time)

class Q_learning_agent(object):
    """Simple Q-learning agent for the MountainCarv0 task
       https://en.wikipedia.org/wiki/Q-learning
    """

    n_actions = 3

    def __init__(self, min_speed, max_speed, min_position, max_position, alpha = 0.1, gamma = 0.9, p_explore = 0.1):
        
        # number of grids per state variable
        self.n_grid = 10
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_step = (max_speed - min_speed) / self.n_grid
        self.min_position = min_position
        self.max_position = max_position
        self.position_step = (max_position - min_position) / self.n_grid
        # discretizing the 2-variable state results in this number of states:
        self.n_states = int(self.n_grid**2)
        # make an empty Q-matrix
        self.Q = np.zeros([self.n_states, self.n_actions])
        #self.Q = np.random.rand(self.n_states, self.n_actions)
        # initialize previous state and action
        self.previous_state = 0
        self.previous_action = 0
        # learning rate
        self.alpha = alpha
        # discount factor:
        self.gamma = gamma
        # e-greedy, p_explore results in a random action:
        self.p_explore = p_explore

    def act(self, observation, reward, done, verbose = False):
        
        # Determine the new state:
        pos = observation[0]
        if(pos > self.max_position):
            pos = self.max_position
        elif(pos < self.min_position):
            pos = self.min_position
        obs_pos = int((pos - self.min_position) // self.position_step)
                
        vel = observation[1]
        if(vel > self.max_speed):
            vel = self.max_speed
        elif(vel < self.min_speed):
            vel = self.min_speed
        obs_vel = int((vel - self.min_speed) // self.speed_step)
        new_state = obs_pos * self.n_grid + obs_vel
        
        if(verbose):
            print(f'Velocity {observation[1]}, position {observation[0]}, (grid {self.speed_step}, \
                          {self.position_step}), state = {new_state}')
        
        # Update the Q-matrix:
        self.Q[self.previous_state, self.previous_action] +=  self.alpha * \
            (reward + self.gamma * max(self.Q[new_state, :]) - self.Q[self.previous_state, self.previous_action])
        
        #print(self.Q)
        
        # determine the new action:
        if(random.random() < self.p_explore):
            action = random.randint(0, self.n_actions-1)
            #print(f'random action: {action:d}')
        else:
            action = np.argmax(self.Q[new_state, :])
            #print(f'action: {action:d}')
        
        # update previous state and action
        self.previous_state = new_state
        self.previous_action = action        
        
        # return the action
        return action

class random_agent(object):
    """Random agent"""

    def act(self, observation, reward, done):
        print('Observation ' + str(observation))
        print('Reward = ' + str(reward))
        return random.randint(0,2)
        

class CMC_adapted(cc.MountainCarEnv):
    """ Derived class of Continuous Mountain Car, so that we can change, e.g., the reward function.
    """
    # Based on: https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/continuous_mountain_car.py
    
    n_steps_per_render = 5
    
    def __init__(self):
        self.render_step = 0
        self.figure_handle = []
        super(CMC_adapted, self).__init__()
        self.max_distance = self.max_position - self.min_position
        self.min_distance = self.max_distance
    
    
    def reset(self):
        self.render_step = 0
        super(CMC_adapted, self).reset()
        self.max_distance = self.max_position - self.min_position
        self.min_distance = self.max_distance
        if(self.figure_handle != []):
            plt.close('mountain_car')
            self.figure_handle = []
        return np.array(self.state)
    
    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        
        # ***************
        # REWARD FUNCTION
        # ***************
        
        # Now you can change the reward function here:
        if(not done):
            reward = -1
        else:
            reward = 0.50
        
#        # determine the distance to the goal:
#        distance = abs(position - self.goal_position)
#        # keep track of the minimal distance during the run:
#        if(distance < self.min_distance):
#            self.min_distance = distance
#        
#        reward = 0
#        if done:
#            reward = 10.0
#        #reward -= math.pow(action,2)*0.1
#        reward += 1. - distance / (self.max_position - self.min_position)
#        reward += 20*abs(velocity)
        
        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}
        
    def render(self, mode='human', sleep_time=0.033):
        
        if os.name == 'nt':
            normal_display = True
        else:
            if 'DISPLAY' in os.environ.keys():
                normal_display = True
            else:
                normal_display = False
            
        if normal_display:
            super(CMC_adapted, self).render()
        else:
            self.render_step += 1
            if(self.render_step % self.n_steps_per_render == 0):
                # first plot the landscape:
                step = 0.01
                x_coords = np.arange(self.min_position, self.max_position, step)
                y_coords = self._height(x_coords)
                
                if(self.figure_handle == []):
                    self.figure_handle = plt.figure('mountain_car')
                    self.ax = self.figure_handle.add_subplot(111)
                    plt.ion()
                    #self.figure_handle.show()
                    self.figure_handle.canvas.draw()
                else:
                    plt.figure('mountain_car')
                
                self.ax.clear()
                self.ax.plot(x_coords, y_coords)
                self.ax.plot(self.state[0], self._height(self.state[0]), 'ro')
                self.ax.text(self.goal_position, self._height(self.goal_position)+0.02, 'Goal')        
                #        self.figure_handle.canvas.draw()
                #        self.figure_handle.show()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(sleep_time)

def run_cart_discrete(agent, simulation_seed=0, n_episodes=1, env=cc.MountainCarEnv(), \
                      max_steps = 200, graphics=False, verbose=False, \
                      random_initial_state=False, finished_at_goal=True):
    """ Runs the discrete cart problem, with the agent mapping observations to actions 
        - agent: should implement a method act(observation, reward, done)
        - simulation_seed: used to set the random seed for simulation
        - n_episodes: how many times the task is run for evaluation
        - env: the environment to be used. Standard is the standard discrete mountain car
        - graphics: If True, render() will be called.
    """
    
    env.seed(simulation_seed)
    random.seed(simulation_seed)
    cumulative_reward = 0
    rewards = []

    for i in range(n_episodes):
        # reset the environment
        ob = env.reset()
        if(verbose):
            print(f'Episode {i}: cumulative reward = {cumulative_reward}')
        step = 0        
        done = False # true if the car reached the goal
        finished = False # true if we want to stop this episode
        reward = 0
        
        if random_initial_state:    
            # select a random velocity and position:
            position = env.min_position + random.random() * (env.max_position - env.min_position)
            velocity = -env.max_speed + random.random() * 2 * env.max_speed
            env.state = (position, velocity)
            
        while True:
            
            # This is where we pass observations and rewards to the agent:            
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if(verbose):
                print(f'{reward}, ', end='')
            rewards.append(reward)
            cumulative_reward += reward
            step += 1
#            if(done):
#                # process the final successful action and high reward:
#                action = agent.act(ob, reward, done)
#                # agent succeeded in the task, restart:
#                ob = env.reset()
#                # make sure done = false
#                done = False
            if(finished_at_goal and done):
                finished = True
                if(verbose):
                    print(f'Episode {i}: Success!')
            if(step >= max_steps):
                finished = True
            if(graphics):
                env.render()
            if finished:
                break

    env.close()    
    
    return cumulative_reward, np.asarray(rewards);

if __name__ == '__main__':
    
    np.random.seed(0)
    
    env=CMC_original()

    # set up off-policy learning with p_explore = 1
    max_velocity = env.max_speed
    min_velocity = -max_velocity
    agent = Q_learning_agent(min_velocity, max_velocity, env.min_position, env.max_position, \
                             alpha = 0.20, gamma = 0.95, p_explore = 1.0)
    n_episodes = 1000
    reward, rewards = run_cart_discrete(agent, env=env, graphics=False, n_episodes=n_episodes)
    print('Reward per episode = ' + str(reward / n_episodes))
    
    # on-policy now with e-greedy
    agent.p_explore = 0.05
    reward, rewards = run_cart_discrete(agent, env=env, graphics=False, n_episodes=n_episodes)
    print('Reward per episode = ' + str(reward / n_episodes))
    
    n_episodes = 100
    agent.alpha = 0.05
    agent.p_explore = 0.02
    reward, rewards = run_cart_discrete(agent, env=env, graphics=False, n_episodes=n_episodes)
    print('Reward per episode = ' + str(reward / n_episodes))

    