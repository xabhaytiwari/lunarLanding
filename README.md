# **PPO for LunarLander-v3**

This repository contains a Python script to train a Proximal Policy Optimization (PPO) agent to solve the LunarLander-v3 environment from [Gymnasium](https://gymnasium.farama.org/). The agent is built and trained using the [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.

## **Motivation**

This project was inspired by the participation in the Summer School held by the Department of Computer Science and Automation (CSA) at the Indian Institute of Science (IISc), Bangalore.

## **Getting Started**

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### **Prerequisites**

This script requires Python 3.8+ and the libraries listed in the requirements.txt file.

### **Installation**

1. **Clone the repository:**  
   git clone https://github.com/xabhaytiwari/lunarLanding.git
   
   cd lunarLanding

3. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows use \`venv\\Scripts\\activate\`

4. Install the required packages:  
   The requirements.txt file in the repository contains all the necessary dependencies.  
   pip install \-r requirements.txt

## **Usage**

To run the script, simply execute the main Python file:

python your\_script\_name.py

This will perform the following steps:

1. Create and vectorize the LunarLander-v3 environment.  
2. Instantiate the PPO agent.  
3. Train the agent for 10,000 timesteps.  
4. Save the trained model to ppo-LunarLander-v3.zip.  
5. Evaluate the trained model's performance.  
6. Render the agent's performance in the environment for 1000 steps.

## **Code Explanation**

The script is divided into several key parts:

1. **Environment Setup**:  
   env \= make\_vec\_env('LunarLander-v3', n\_envs=16)

   We create 16 parallel environments to collect a diverse set of experiences during training, which speeds up the learning process.  
2. **Agent Instantiation**:  
   model \= PPO('MlpPolicy', env, n\_steps=1024, batch\_size=64, n\_epochs=4, gamma=0.999, gae\_lambda=0.98, ent\_coef=0.01, verbose=1)

   We use the PPO algorithm with a Multi-Layer Perceptron (MlpPolicy) because the input from the environment is vectorized data (position, velocity, etc.). The hyperparameters have been chosen to provide a good starting point for training.  
3. **Training**:  
   model.learn(total\_timesteps=int(10000))  
   model.save("ppo-LunarLander-v3")

   The agent is trained for a total of 1,000,000 timesteps. While this is a relatively short training duration, it's sufficient to see initial learning. The trained model is then saved.  
4. **Evaluation**:  
   eval\_env \= Monitor(gym.make("LunarLander-v3", render\_mode="rgb\_array"))  
   model \= PPO.load("ppo-LunarLander-v3", env \= eval\_env)  
   mean\_reward, std\_reward \= evaluate\_policy(model, model.get\_env(), n\_eval\_episodes=10)

   The saved model is loaded and evaluated over 10 episodes to get a stable estimate of its performance. The Monitor wrapper is used to keep track of rewards and episode lengths.  
5. **Visualisation**:  
   vec\_env \= model.get\_env()  
   obs \= vec\_env.reset()  
   for i in range(1000):  
       action, \_states \= model.predict(obs, deterministic=True)  
       obs, rewards, dones, info \= vec\_env.step(action)  
       vec\_env.render("human")

   Finally, we watch the trained agent in action\! The deterministic=True flag ensures that the agent chooses the best-known action at each step, rather than exploring.
