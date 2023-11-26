import gymnasium as gym
from stable_baselines3.ppo import PPO
import os
import argparse

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
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/PPO_{TIMESTEPS}")

def test(path_to_model):
    env = gym.make(ENV, render_mode = 'human')
    model = PPO.load(path_to_model, env = env)
    obs, _ = env.reset()
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs) #passing state to get action
        obs, _, done, _, _ = env.step(action) #pass action to get next state
        
        if done:
            extra_steps -= 1
            if extra_steps < 0:
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
            
    
    