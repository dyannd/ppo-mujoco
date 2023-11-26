import gymnasium as gym
from stable_baselines3.ppo import PPO
import os
import argparse
import datetime
import time

#constants
model_dir = "model"
log_dir = "logs"
os.makedirs(model_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
ENV = 'HalfCheetah-v4'

def train():
    model = PPO('MlpPolicy', ENV, verbose=1, device="cuda", tensorboard_log=log_dir, learning_rate=0.0002, gamma=0.97, n_steps=512) #MlpNN #change hyperParams
    TIMESTEPS = 1000000
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # Generate a timestamp in the "PPO_Jan18_1830PM" format
    current_time = datetime.datetime.now().strftime("%b%d_%I%M%p")
    
    # Append the formatted timestamp to the model filename
    model_filename = f"{model_dir}/PPO_{current_time}_{TIMESTEPS}timesteps"
    model.save(model_filename)

def test(path_to_model):
    env = gym.make(ENV, render_mode = 'human')
    model = PPO.load(path_to_model, env = env)
    obs, _ = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs) #passing state to get action
        obs, _, done, _, _ = env.step(action) #pass action to get next state
        time.sleep(0.001)
        if done:
            break
    
if __name__ == '__main__':
    #parse cmd inputs
    parser = argparse.ArgumentParser(description='Train/test model')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()
    
    if args.train:
        gymEnv = gym.make(ENV, render_mode = None)
        train()
        
    if (args.test):
        if os.path.isfile(args.test): #if path correct
            test(path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
            
    
    