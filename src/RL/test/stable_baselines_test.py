import gym


# Proximal policy optimization - works on discrete or continuous actions
# Outputs a distribtution of actions instead of a single action - this distribution is sample from
# Also uses KL divergence to add stability - esentially verifies that the output distribtution does not change very much, which
# can happen even with small learning updates
from stable_baselines3 import PPO

# create env to learn
env = gym.make("CartPole-v1")

# create alg and learn it for 10k steps
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Show the learned policy on the env
obs = env.reset()
for i in range(1000):
    # returns action from the agent
    action, _states = model.predict(obs, deterministic=True)

    # normal environment stuff
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()