import gym
import time
# This file is generally how you use GYM

# First we create an environment. we also call reset to get the initial state
# and to put it into a valid state to start an episode
env = gym.make('Breakout-v4', render_mode='human')
observation = env.reset()

# then you iterate for some number of steps, in this case 1000 steps
for _ in range(1000):
    # Usually you use an agent here to get an action. This is simply random.
    action = env.action_space.sample()

    # then we apply the action to the env, which returns a new state, reward, termination (bool), and more info
    observation, reward, terminated, info = env.step(action)

    # if the env has termianted, then we have to reset it to start a new episode0
    if terminated:
        observation = env.reset()

# this is not always needed, but sometimes useful to close up rendering windows
env.close()