"""
SAC 을 테스트하는 부분입니다. 
replay buffer에 데이터를 넣어 초기화하고, 학습을 진행하는 부분이 포함되어 있습니다.

"""
import os, sys
from agents.rnd import RND
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla  
import gym
import numpy as np
from agents.sac import SACOfflineOnline
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
from env.wrapper import CarlaWrapperEnv 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime 

DATA_DIR = "offline_data_for_replaybuffer/dataset_town03"  
SIMULATION = "일반화실험_SAC_모델저장"
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

def make_env(start_point, town):
    
    # carla 연결 
    client, world, carla_map = connect_to_carla(town)
   
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

def make_vec_env(start_point , town):
    return DummyVecEnv([lambda: make_env(start_point, town)])

def main(start_point, town):
    env = make_vec_env(start_point , town)
    trainer = SACOfflineOnline(env=env, buffer_size=4_000_000, batch_size=256, tau=0.005, verbose=1, tensorboard_log="logs/"+SIMULATION+"/"+NOW)

    trainer.replay_buffer.reset()
    trainer.prefill_from_npz_folder_mclearn(DATA_DIR)
    print("replay buffer 초기화 완료")
    print("=================== 학습 시작 ====================")
    trainer.online_learn(log_interval=50, total_timesteps=1_500_000, tb_log_name="logs/"+SIMULATION+"/"+NOW)

    trainer.save(f"trained_1M_1M_{NOW}.zip")
    env.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("입력 형식: python run_proposed.py <start_x> <start_y> <town이름>\n town 이름은 다음 중 하나입니다 : Town01 / Town02 / Town03 / Town04 / Town05")
        sys.exit(1)

    try:
        start_x = float(sys.argv[1])
        start_y = float(sys.argv[2])
        town = str(sys.argv[3])
    except ValueError:
        print("입력 형식: python run_proposed.py <start_x> <start_y> <town이름>")
        sys.exit(1)

    start_point = (start_x, start_y)
    main(start_point, town)
