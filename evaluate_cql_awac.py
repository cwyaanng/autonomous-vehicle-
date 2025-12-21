import sys
import os
import numpy as np
import carla
from datetime import datetime

# 1. CARLA 경로 설정
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')

import d3rlpy
from d3rlpy.algos import AWAC, CQL
from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv

USE_GPU = True 

def make_env(town, start_point, run_name):
    """
    테스트를 위한 환경 생성 함수
    run_name: 결과가 저장될 폴더 이름에 붙을 태그 (예: 'AWAC_1차')
    """
    log_root = f"TEST_일반화실험_{town}/실험_{run_name}"
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

def test(model_path, start_point, town, algo_type, run_name, n_episodes=5):
    """
    Args:
        model_path: .d3 모델 파일 경로
        start_point: (x, y) 튜플
        town: 맵 이름
        algo_type: "AWAC" 또는 "CQL" (대소문자 구분 없음) - 알고리즘 클래스 결정용
        run_name: 로그 폴더 이름에 붙을 식별자 (예: "제안기법_1차")
    """
    print(f"\n === 테스트 시작: {run_name} ({algo_type}) ===")
    print(f"환경 생성 Town: {town}, Start: {start_point}")
    
    env = make_env(town, start_point, run_name)

    # 알고리즘 초기화 
    if algo_type.upper() == "AWAC":
        algo = AWAC(use_gpu=USE_GPU, batch_size=256)
        print("[Info] AWAC 알고리즘 객체 생성")
        
    elif algo_type.upper() == "CQL":
        algo = CQL(use_gpu=USE_GPU)
        print("[Info] CQL 알고리즘 객체 생성")
        
    algo.build_with_env(env)
    
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    algo.load_model(model_path)

    total_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"--- Episode {ep+1}/{n_episodes} Start ---")
        
        while not done:

            action = algo.predict([obs])[0]
            
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
            
            if step_count > 30000:
                print("Max steps reached. Forcing done.")
                break

        print(f"Episode {ep+1} Finished. Reward: {episode_reward:.2f}, Steps: {step_count}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"[Result] {run_name} Average Reward: {avg_reward:.2f}")
    print("="*50)


if __name__ == "__main__":
    
    # 공통 설정
    current_town = "Town01"
    start_pos = (0, 0)  

    """
    
    
    AWAC 모델 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251216_203430/model/awac_offline_online_20251216_203427.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_1차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_010718/model/awac_offline_online_20251217_010715.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_2차", 
        n_episodes=5
    )
  
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_053745/model/awac_offline_online_20251217_053742.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_3차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_091842/model/awac_offline_online_20251217_091840.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_4차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_165559/model/awac_offline_online_20251217_165556.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_5차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_203311/model/awac_offline_online_20251217_203309.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_6차", 
        n_episodes=5
    )

    """
    
    
    CQL 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251212_221919/model/cql_offline_online_20251212_221915.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",        
        run_name="CQL_베이스라인_1차",  
        n_episodes=5
    )
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_115140/model/cql_offline_online_20251214_115137.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",        
        run_name="CQL_베이스라인_2차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_193829/model/cql_offline_online_20251214_193826.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_3차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251215_233402/model/cql_offline_online_20251215_233359.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_4차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_071904/model/cql_offline_online_20251216_071902.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_5차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_150552/model/cql_offline_online_20251216_150550.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_6차",  
        n_episodes=5
    )
    
    current_town = "Town02"
    start_pos = (0, 0)  
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251216_203430/model/awac_offline_online_20251216_203427.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_1차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_010718/model/awac_offline_online_20251217_010715.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_2차", 
        n_episodes=5
    )
  
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_053745/model/awac_offline_online_20251217_053742.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_3차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_091842/model/awac_offline_online_20251217_091840.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_4차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_165559/model/awac_offline_online_20251217_165556.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_5차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_203311/model/awac_offline_online_20251217_203309.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_6차", 
        n_episodes=5
    )

    """
    
    
    CQL 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251212_221919/model/cql_offline_online_20251212_221915.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_1차",  
        n_episodes=5
    )
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_115140/model/cql_offline_online_20251214_115137.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_2차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_193829/model/cql_offline_online_20251214_193826.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_3차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251215_233402/model/cql_offline_online_20251215_233359.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_4차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_071904/model/cql_offline_online_20251216_071902.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_5차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_150552/model/cql_offline_online_20251216_150550.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_6차",  
        n_episodes=5
    )
  
    current_town = "Town03"
    start_pos = (0, 0)  
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251216_203430/model/awac_offline_online_20251216_203427.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_1차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_010718/model/awac_offline_online_20251217_010715.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_2차", 
        n_episodes=5
    )
  
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_053745/model/awac_offline_online_20251217_053742.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_3차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_091842/model/awac_offline_online_20251217_091840.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_4차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_165559/model/awac_offline_online_20251217_165556.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_5차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_203311/model/awac_offline_online_20251217_203309.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_6차", 
        n_episodes=5
    )

    """
    
    
    CQL 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251212_221919/model/cql_offline_online_20251212_221915.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_1차",  
        n_episodes=5
    )
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_115140/model/cql_offline_online_20251214_115137.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_2차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_193829/model/cql_offline_online_20251214_193826.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_3차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251215_233402/model/cql_offline_online_20251215_233359.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_4차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_071904/model/cql_offline_online_20251216_071902.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_5차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_150552/model/cql_offline_online_20251216_150550.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_6차",  
        n_episodes=5
    )
    
    
    current_town = "Town04"
    start_pos = (300,-100)  
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251216_203430/model/awac_offline_online_20251216_203427.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_1차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_010718/model/awac_offline_online_20251217_010715.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_2차", 
        n_episodes=5
    )
  
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_053745/model/awac_offline_online_20251217_053742.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_3차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_091842/model/awac_offline_online_20251217_091840.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_4차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_165559/model/awac_offline_online_20251217_165556.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_5차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_203311/model/awac_offline_online_20251217_203309.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_6차", 
        n_episodes=5
    )

    """
    
    
    CQL 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251212_221919/model/cql_offline_online_20251212_221915.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_1차",  
        n_episodes=5
    )
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_115140/model/cql_offline_online_20251214_115137.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_2차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_193829/model/cql_offline_online_20251214_193826.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_3차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251215_233402/model/cql_offline_online_20251215_233359.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_4차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_071904/model/cql_offline_online_20251216_071902.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_5차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_150552/model/cql_offline_online_20251216_150550.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_6차",  
        n_episodes=5
    )
    
    
    current_town = "Town05"
    start_pos = (0,0)  
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251216_203430/model/awac_offline_online_20251216_203427.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_1차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_010718/model/awac_offline_online_20251217_010715.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_2차", 
        n_episodes=5
    )
  
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_053745/model/awac_offline_online_20251217_053742.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_3차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_091842/model/awac_offline_online_20251217_091840.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_4차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_165559/model/awac_offline_online_20251217_165556.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_5차", 
        n_episodes=5
    )
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_AWAC_모델저장/20251217_203311/model/awac_offline_online_20251217_203309.d3", 
        start_point=start_pos,
        town=current_town,
        algo_type="AWAC",       
        run_name="AWAC_제안기법_6차", 
        n_episodes=5
    )

    """
    
    
    CQL 테스트 
    
    
    """
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251212_221919/model/cql_offline_online_20251212_221915.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_1차",  
        n_episodes=5
    )
    
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_115140/model/cql_offline_online_20251214_115137.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_2차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251214_193829/model/cql_offline_online_20251214_193826.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_3차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251215_233402/model/cql_offline_online_20251215_233359.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_4차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_071904/model/cql_offline_online_20251216_071902.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",         
        run_name="CQL_베이스라인_5차",  
        n_episodes=5
    )
    
    test(
        model_path="/home/wise/chaewon/logs/일반화실험_CQL_모델저장/20251216_150552/model/cql_offline_online_20251216_150550.d3",     
        start_point=start_pos,
        town=current_town,
        algo_type="CQL",        
        run_name="CQL_베이스라인_6차",  
        n_episodes=5
    )