import gymnasium as gym
from stable_baselines3.ppo import PPO
import os
import argparse
import datetime
import time
from torch import nn
#constants
model_dir = "model"
log_dir = "logs"
os.makedirs(model_dir, exist_ok = True)
os.makedirs(log_dir, exist_ok = True)
ENV = 'HalfCheetah-v4'

def train():
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.00001, n_steps=2048, 
                batch_size=64, n_epochs=50, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.1, tensorboard_log=log_dir) #MlpNN #change hyperParams
    TIMESTEPS = 500000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.00002, n_steps=512, 
                batch_size=64, n_epochs=25, gamma=0.98, gae_lambda=0.92, clip_range=0.1, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.0004, max_grad_norm = 0.8, vf_coef = 0.58, tensorboard_log=log_dir) #MlpNN #change hyperParams
    TIMESTEPS = 1000000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.000025, n_steps=1024, 
                batch_size=64, n_epochs=50, gamma=0.98, gae_lambda=0.92, clip_range=0.1, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.0004, max_grad_norm = 0.8, vf_coef = 0.58, tensorboard_log=log_dir) #MlpNN #change hyperParams
    TIMESTEPS = 1000000
    model = PPO('MlpPolicy', ENV, verbose=1, device="cuda", tensorboard_log=log_dir, learning_rate=0.0002, gamma=0.97, n_steps=512) #MlpNN #change hyperParams
    TIMESTEPS = 1000000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.00002, n_steps=512, 
                batch_size=64, n_epochs=100, gamma=0.98, gae_lambda=0.92, clip_range=0.1, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.0004, max_grad_norm = 0.8, vf_coef = 0.58, tensorboard_log=log_dir) #MlpNN #change hyperParams
    TIMESTEPS = 2000000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.0001, n_steps=512, 
                batch_size=64, n_epochs=70, gamma=0.98, gae_lambda=0.94, clip_range=0.15, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.00004, max_grad_norm = 0.8, vf_coef = 0.58, verbose=1, tensorboard_log=log_dir, policy_kwargs={'log_std_init':-2, 'ortho_init':False, 'activation_fn':nn.ReLU}) #MlpNN #change hyperParams
    TIMESTEPS = 1500000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.000015, n_steps=512, 
                batch_size=64, n_epochs=60, gamma=0.98, gae_lambda=0.94, clip_range=0.15, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.00004, max_grad_norm = 0.8, vf_coef = 0.58, verbose=1, tensorboard_log=log_dir, policy_kwargs={'log_std_init':-1, 'ortho_init':False, 'activation_fn':nn.LeakyReLU}) #MlpNN #change hyperParams
    TIMESTEPS = 1500000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.0000105, n_steps=512, 
                batch_size=64, n_epochs=80, gamma=0.98, gae_lambda=0.94, clip_range=0.15, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.00004, max_grad_norm = 0.8, vf_coef = 0.58, verbose=1, tensorboard_log=log_dir, policy_kwargs={'log_std_init':-1, 'ortho_init':False, 'activation_fn':nn.LeakyReLU}) #MlpNN #change hyperParams
    TIMESTEPS = 1500000
    model = PPO('MlpPolicy', ENV, device="cuda", learning_rate=0.000013, n_steps=512, 
                batch_size=64, n_epochs=80, gamma=0.98, gae_lambda=0.94, clip_range=0.15, clip_range_vf=None, 
                normalize_advantage=True, ent_coef=0.00004, max_grad_norm = 0.8, vf_coef = 0.58, verbose=1, tensorboard_log=log_dir, policy_kwargs={'log_std_init':-1, 'ortho_init':False, 'activation_fn':nn.LeakyReLU}) #MlpNN #change hyperParams
    TIMESTEPS = 1500000
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
            
    
    