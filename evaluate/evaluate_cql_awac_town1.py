import sys
import os
import numpy as np
import carla
import argparse
from datetime import datetime

# 1. CARLA 경로 설정 (사용자 환경에 맞춤)
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')

import d3rlpy
from d3rlpy.algos import AWAC
from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv

# GPU 사용 여부
USE_GPU = True 

def make_env(town, start_point):
    """
    테스트를 위한 환경 생성 함수
    로그 경로는 테스트용으로 임시 지정하거나 필요시 변경
    """
    log_root = "logs/test_results"
    os.makedirs(log_root, exist_ok=True)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    client, world, carla_map = connect_to_carla(town)
    
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=start_point,
        simulation=os.path.join(log_root, now),
    )
    return env

def test(model_path, start_point, town, n_episodes=5):
  
    print(f"[Info] 환경 생성 중... Town: {town}, Start: {start_point}")
    env = make_env(town, start_point)

    awac = AWAC(use_gpu=USE_GPU, batch_size=256)


    awac.build_with_env(env)
    
    print(f"[Info] 모델 로드 중: {model_path}")
    awac.load_model(model_path)

    # 5. 테스트 루프 실행
    total_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"--- Episode {ep+1} Start ---")
        
        while not done:
            # d3rlpy predict는 배치 입력을 기대하므로 [obs] 형태로 넣어줍니다.
            # 결과도 배치 형태이므로 [0]으로 첫 번째 액션을 가져옵니다.
            action = awac.predict([obs])[0]
            
            # 환경 step 진행
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
            
            # (선택사항) 너무 길어지면 강제 종료
            if step_count > 5000:
                print("Max steps reached.")
                break

        print(f"Episode {ep+1} Finished. Total Reward: {episode_reward:.2f}, Steps: {step_count}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"\n[Result] Average Reward over {n_episodes} episodes: {avg_reward:.2f}")

    # 종료 시 정리 (필요에 따라)
    # env.close() 

if __name__ == "__main__":
    # 입력 인자 처리
    if len(sys.argv) < 5:
        print("사용법: python test_model.py <start_x> <start_y> <town이름> <모델파일경로.d3>")
        print("예시: python test_model.py 10.5 20.0 Town03 awac_model.d3")
        sys.exit(1)

    try:
        start_x = float(sys.argv[1])
        start_y = float(sys.argv[2])
        town = str(sys.argv[3])
        model_path = str(sys.argv[4])
    except ValueError:
        print("좌표는 숫자여야 합니다.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"에러: 모델 파일을 찾을 수 없습니다 -> {model_path}")
        sys.exit(1)

    start_point = (start_x, start_y)
    
    # 테스트 실행
    test(model_path, start_point, town)