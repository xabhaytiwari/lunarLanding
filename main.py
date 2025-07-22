import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Create environment.
env = make_vec_env('LunarLander-v3', n_envs=16)
# We stacked 16 environments, all independent, to get diverse experiences during the training

#Instatiate the agent, choose the model
#PPO stands for Proximal Policy is one of the SOTA Deep RL Algos
# MLpPolicy is multiplayer perceptron, as we have got vectorised input
model = PPO('MlpPolicy', env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=1)

#Train it for 10,000 steps (AMD Laptop!)
model.learn(total_timesteps=int(1000000))
model.save("ppo-LunarLander-v3")
del model 

# Evaluate the agent
eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
model = PPO.load("ppo-LunarLander-v3", env = eval_env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Here we go
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")