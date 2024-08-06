

##########################
import gymnasium as gym
import numpy as np


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


env = gym.make("MountainCarContinuous-v0", render_mode="human")


# The noise object for TD3
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))


# Specified hyperparameters
n_timesteps = 300000
policy = 'MlpPolicy'
noise_type = 'ornstein-uhlenbeck'
noise_std = 0.5
gradient_steps = 1
train_freq = 1
learning_rate = 1e-3
batch_size = 256
policy_kwargs = dict(net_arch=[400, 300])


# Create the model with the specified hyperparameters
model = TD3(
   policy,
   env,
   action_noise=action_noise,
   gradient_steps=gradient_steps,
   train_freq=train_freq,
   learning_rate=learning_rate,
   batch_size=batch_size,
   policy_kwargs=policy_kwargs,
   verbose=1
)


# Train the model
model.learn(total_timesteps=n_timesteps, log_interval=10)
model.save("td3_mountaincar_continuous")
vec_env = model.get_env()


# Remove the model to demonstrate saving and loading
del model
     

# Load the model
model = TD3.load("td3_mountaincar_continuous")


obs = vec_env.reset()
while True:
   action, _states = model.predict(obs)
   obs, rewards, dones, info = vec_env.step(action)
   vec_env.render("human")
