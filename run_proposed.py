"""
제안 기법을 테스트하는 코드입니다.

"""
import os, sys
from agents.rnd import RND
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla  
import gym
import numpy as np
from agents.sacrnd_model_revised import SACOfflineOnline
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
from env.wrapper import CarlaWrapperEnv 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime 

DATA_DIR = "offline_data_for_replaybuffer/dataset_town03"  
SIMULATION = "일반화실험_제안기법_모델저장"
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

def make_env(start_point):
    
    # carla 연결 
    client, world, carla_map = connect_to_carla()
    # 주행 시작 포인트 
    # town3 : (0,0)
    # town4 : (300,-100)
    
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=start_point,
        simulation="logs/"+SIMULATION+"/"+NOW,
        target_speed=22.0
    )
    env = Monitor(env)
   
    return env

def make_vec_env(start_point):
    return DummyVecEnv([lambda: make_env(start_point)])

def main(start_point):
    env = make_vec_env(start_point)
    trainer = SACOfflineOnline(env=env, buffer_size=4_000_000, batch_size=256, tau=0.005, verbose=1, tensorboard_log="logs/"+SIMULATION+"/"+NOW)
    obs_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    rnd = RND(obs_dim, lr=1e-3, device=str(trainer.device)) 
 
    trainer.attach_rnd(rnd) 
    trainer.replay_buffer.reset()
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    trainer.train_mcnet_from_buffer(epochs=30)
    trainer.replay_buffer.reset()
    trainer.save_mcnet_pth(f"mcnet/mcnet_pretrained_{NOW}.pth")
    trainer.save_mcnet_pickle(f"mcnet/mcnet_pretrained_{NOW}.pkl")
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    trainer.online_learn(log_interval=50, total_timesteps=1_500_000, tb_log_name="logs/"+SIMULATION+"/"+NOW)

    trainer.save(f"trained_1M_1M_{NOW}.zip")
    env.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("입력 형식: python run_sacrnd.py <start_x> <start_y>")
        sys.exit(1)

    try:
        start_x = float(sys.argv[1])
        start_y = float(sys.argv[2])
    except ValueError:
        print("start_x start_y 를 공백을 사이에 두고 숫자로 입력해 주세요 ")
        sys.exit(1)

    start_point = (start_x, start_y)
    main(start_point)
