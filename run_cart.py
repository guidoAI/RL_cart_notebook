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

# TODO: Create a new agent class that learns from the rewards it gets for its actions

class random_agent(object):
    """Random agent"""

    def act(self, observation, reward, done):
        return random.randint(0,2)
    
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
        
        # Now you can change the reward function here:
        distance = abs(position - self.goal_position)
        if(distance < self.min_distance):
            self.min_distance = distance
            
        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0],2)*0.1
        reward += 1. - self.min_distance / self.max_distance
        
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

def run_cart_discrete(agent, simulation_seed=0, n_episodes=1, env=cc.MountainCarEnv(), max_steps = 1000, graphics=False):
    """ Runs the discrete cart problem, with the agent mapping observations to actions 
        - agent: should implement a method act(observation, reward, done)
        - simulation_seed: used to set the random seed for simulation
        - n_episodes: how many times the task is run for evaluation
        - env: the environment to be used. Standard is the standard discrete mountain car
        - graphics: If True, render() will be called.
    """
    
    env.seed(simulation_seed)

    reward = 0
    cumulative_reward = 0
    done = False
    step = 0

    for i in range(n_episodes):
        ob = env.reset()
        while True:
            
            # This is where we pass observations and rewards to the agent:            
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            cumulative_reward += reward
            step += 1
            if(step >= max_steps):
                done = True
            if(graphics):
                env.render()
            if done:
                break

    env.close()    
    
    return cumulative_reward;

if __name__ == '__main__':
    agent = random_agent()
    reward = run_cart_discrete(agent, simulation_seed=0, env=CMC_original(), graphics=True)