PPO for LunarLander-v3
This repository contains a Python script to train a Proximal Policy Optimization (PPO) agent to solve the LunarLander-v3 environment from Gymnasium. The agent is built and trained using the Stable Baselines3 library.

Motivation
This project was inspired by the participation in the Summer School held by Department of Computer Science and Automation (CSA) at the Indian Institute of Science (IISc), Bangalore.

Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
This script requires Python 3.8+ and the libraries listed in the requirements.txt file.

Installation
Clone the repository:

git clone https://github.com/xabhaytiwari/lunarLanding.git
cd lunarLanding

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
The requirements.txt file in the repository contains all the necessary dependencies.

pip install -r requirements.txt

Usage
To run the script, simply execute the main Python file:

python your_script_name.py

This will perform the following steps:

Create and vectorize the LunarLander-v3 environment.

Instantiate the PPO agent.

Train the agent for 10,000 timesteps.

Save the trained model to ppo-LunarLander-v3.zip.

Evaluate the trained model's performance.

Render the agent's performance in the environment for 1000 steps.

Code Explanation
The script is divided into several key parts:

Environment Setup:

env = make_vec_env('LunarLander-v3', n_envs=16)

We create 16 parallel environments to collect a diverse set of experiences during training, which speeds up the learning process.

Agent Instantiation:

model = PPO('MlpPolicy', env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=1)

We use the PPO algorithm with a Multi-Layer Perceptron (MlpPolicy) because the input from the environment is vectorized data (position, velocity, etc.). The hyperparameters have been chosen to provide a good starting point for training.

Training:

model.learn(total_timesteps=int(10000))
model.save("ppo-LunarLander-v3")

The agent is trained for a total of 10,000 timesteps. While this is a relatively short training duration, it's sufficient to see initial learning. The trained model is then saved.

Evaluation:

eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))
model = PPO.load("ppo-LunarLander-v3", env = eval_env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

The saved model is loaded and evaluated over 10 episodes to get a stable estimate of its performance. The Monitor wrapper is used to keep track of rewards and episode lengths.

Visualisation:

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

Finally, we watch the trained agent in action! The deterministic=True flag ensures that the agent chooses the best-known action at each step, rather than exploring.
