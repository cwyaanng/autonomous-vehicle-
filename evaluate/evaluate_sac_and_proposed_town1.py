"""
학습한 모델을 Town1에서 50에피소드 동안 평가합니다. 
"""

import os
import sys
from typing import List, Dict, Any

sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')

import carla
import gym
import numpy as np
import torch as th
from datetime import datetime 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file

from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
SIMULATION = f"TEST_일반화실험_Town03_to_Town01_{NOW}"

def make_env(category , batch_size: int = 256):

    client, world, carla_map = connect_to_carla(town="Town01")
    points = (300, -100)
    NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
    SIMULATION = f"TEST_일반화실험_TOWN01/실험_{NOW}"

    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=points,
        simulation=SIMULATION+"_"+category,
        target_speed=22.0,
        test=True
    )
    env = Monitor(env)
    return env


def make_vec_env(category , batch_size: int = 256 ):
    return DummyVecEnv([lambda: make_env(category , batch_size)])


def load_policy_from_zip(model_path: str, env: gym.Env, device: str = "cpu") -> SAC:
    """
    SB3가 저장한 .zip 파일에서 전체 파라미터를 로드하려 하면
    SACOfflineOnline 구조 mismatch 때문에 에러가 발생하여 zip 안에서 'policy' 부분(state_dict)만 꺼내서 새 SAC 모델에 학습한 정책만 얹어주는 방식으로 테스트했습니다. 
    """
    data, params, pytorch_vars = load_from_zip_file(model_path, device=device)

    if "policy" not in params:
        raise KeyError(f"'policy' key not found in params of {model_path}")

    policy_state_dict = params["policy"]

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        buffer_size=4_000_000, 
        batch_size=256, 
        tau=0.005
    )

    model.policy.load_state_dict(policy_state_dict)
    print(f"[LOAD] Loaded policy weights from: {model_path}")

    return model


def evaluate(
    model_path: str,
    episodes: int = 10,
    max_steps_per_episode: int = 10_000,
    deterministic: bool = True,
    batch_size: int = 256,
    render: bool = False,
    category="proposed method"
):
    # VecEnv + 모델 로드
    env = make_vec_env( category , batch_size=batch_size)
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = load_policy_from_zip(model_path, env=env, device=device)

    # 통계 저장용
    ep_returns: List[float] = []
    ep_route_completions: List[float] = []
    ep_success_flags: List[int] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0

        route_completion = None
        success = 0  

        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            if dones[0]:
                info = infos[0]

                ep_info: Dict[str, Any] = info.get("episode", {})
                route_completion = ep_info.get("episode/route_completion", None)
                done_reached = ep_info.get("episode/done_reached", None)
                if done_reached is not None:
                    success = int(done_reached)

                print(
                    f"[Episode {ep}] steps={step+1}, "
                    f"return={ep_reward:.2f}, "
                    f"route_completion={route_completion}, "
                    f"success={success}"
                )
                break

        else:
            print(
                f"[Episode {ep}] MAX steps reached, return={ep_reward:.2f}, "
                f"route_completion={route_completion}, success={success}"
            )

        ep_returns.append(ep_reward)
        if route_completion is not None:
            ep_route_completions.append(float(route_completion))
        ep_success_flags.append(success)

    env.close()

    n = len(ep_returns)
    mean_return = np.mean(ep_returns) if n > 0 else float("nan")
    std_return = np.std(ep_returns) if n > 0 else float("nan")

    if ep_route_completions:
        mean_rc = np.mean(ep_route_completions)
        std_rc = np.std(ep_route_completions)
    else:
        mean_rc, std_rc = float("nan"), float("nan")

    success_rate = np.mean(ep_success_flags) if ep_success_flags else float("nan")

if __name__ == "__main__":
    
    """ 
    
    제안기법 테스트
    
    """

    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251210_115936/model/제안기법_1차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251210_201408/model/제안기법_2차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251211_114714/model/제안기법_3차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251212_002245/model/제안기법_4차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251212_103620/model/제안기법_5차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_제안기법_모델저장/20251212_162017/model/제안기법_6차.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="제안기법"
    )

    """ 
    
    SAC 테스트 
    
    """
    
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251218_160916/model/trained_1M_1M_20251218_160916.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251218_203607/model/trained_1M_1M_20251218_203607.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251219_130432/model/trained_1M_1M_20251219_130432.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251219_162702/model/trained_1M_1M_20251219_162702.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251219_194714/model/trained_1M_1M_20251219_194714.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
    evaluate(
        "/home/wise/chaewon/logs/일반화실험_SAC_모델저장/20251219_230718/model/trained_1M_1M_20251219_230718.zip",
        episodes=50,          
        max_steps_per_episode=30_000,
        deterministic=True,
        batch_size=256,
        render=False,
        category="SAC"
    )
   
    """ 
    
    SAC 테스트 
    
    """