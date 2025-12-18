#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect hardware environment and model architectures
for SAC (Stable-Baselines3) and AWAC/CQL (d3rlpy).
"""

import os
import platform
import textwrap

print("=" * 80)
print("1) HARDWARE / SYSTEM INFORMATION")
print("=" * 80)

# --- OS / CPU 정보 ---
print(f"OS          : {platform.platform()}")
print(f"Python      : {platform.python_version()}")
print(f"CPU cores   : {os.cpu_count()}")

# --- 메모리 정보 (psutil이 있으면) ---
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM total   : {mem.total / (1024 ** 3):.2f} GB")
except ImportError:
    print("psutil not installed. Install with `pip install psutil` for RAM info.")

# --- GPU 정보 (PyTorch 있으면) ---
try:
    import torch

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"CUDA        : available ({n_gpu} GPU(s))")
        for i in range(n_gpu):
            print(f"  GPU {i} name : {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA        : not available")
except ImportError:
    print("PyTorch not installed. Install with `pip install torch` for GPU info.")

print("\n")


# =====================================================================
# 2) STABLE-BASELINES3 SAC 모델 구조 확인
# =====================================================================
print("=" * 80)
print("2) STABLE-BASELINES3 SAC MODEL ARCHITECTURE")
print("=" * 80)

try:
    from stable_baselines3 import SAC
    import gym  # 너 환경에서 이미 gym 쓰고 있었으니까 gym 기준으로

    # === 여기서 env는 실제로 쓰는 CARLA env로 바꾸면 제일 좋음 ===
    # 예: from your_carla_wrapper import CarlaEnv
    #     env = CarlaEnv(...)
    # 여기서는 예시로 Pendulum-v1 사용 (관찰/액션 차원만 맞으면 구조는 동일)
    env_id = "Pendulum-v1"
    try:
        env = gym.make(env_id)
    except Exception as e:
        print(f"Failed to create env {env_id}: {e}")
        env = None

    if env is not None:
        model = SAC("MlpPolicy", env, verbose=0)

        print(f"SAC policy class : {model.policy.__class__.__name__}")
        print("\n[SAC policy network structure]")
        print(model.policy)

        # policy 내부 구성도 조금 더 보기 좋게 출력
        print("\n[SAC actor network (mu/head)]")
        try:
            print(model.policy.mu)
        except Exception as e:
            print(f"Could not access actor mu network: {e}")

        print("\n[SAC critic / Q-networks]")
        try:
            # 버전에 따라 q_net / q_net1 / q_net2 이름이 다를 수 있음
            attrs = ["q_net", "qf1", "qf2", "critic", "critic_target"]
            for a in attrs:
                if hasattr(model.policy, a):
                    print(f"--- model.policy.{a} ---")
                    print(getattr(model.policy, a))
        except Exception as e:
            print(f"Could not inspect critic networks: {e}")

        env.close()

except ImportError as e:
    print("Stable-Baselines3 is not installed or cannot be imported.")
    print("Install with: pip install stable-baselines3[extra]")
    print(f"Error detail: {e}")

print("\n")


# =====================================================================
# 3) D3RLPY AWAC / CQL 모델 구조 확인 템플릿
# =====================================================================
print("=" * 80)
print("3) D3RLPY AWAC / CQL MODEL ARCHITECTURE (TEMPLATE)")
print("=" * 80)

try:
    import d3rlpy
    import numpy as np
    from d3rlpy.dataset import MDPDataset

    print(f"d3rlpy version: {d3rlpy.__version__}")

    # === 여기를 네 CARLA 관찰/행동 차원에 맞게 수정해줘야 함 ===
    # 예: obs_dim = 관찰 벡터 길이, act_dim = action 벡터 길이
    OBS_DIM = 24   # TODO: 여기를 실제 obs dimension으로 바꿔
    ACT_DIM = 2    # TODO: 여기를 실제 action dimension으로 바꿔

    n_dummy = 32
    observations = np.zeros((n_dummy, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_dummy, ACT_DIM), dtype=np.float32)
    rewards = np.zeros((n_dummy,), dtype=np.float32)
    terminals = np.zeros((n_dummy,), dtype=bool)
    dataset = MDPDataset(observations, actions, rewards, terminals)

    print("\n[AWAC default model info]")
    try:
        awac = d3rlpy.algos.AWAC()
        # d3rlpy 0.90 에서는 impl 초기화를 위해 dataset 기반 build 필요
        awac.build_with_dataset(dataset)
        print(awac)
        print("\nAWAC implementation object:")
        print(awac.impl)

        # q_func / policy 내부 구조를 볼 수 있으면 출력
        for attr in ["q_func", "policy", "actor", "critic"]:
            if hasattr(awac.impl, attr):
                print(f"\n[awac.impl.{attr}]")
                print(getattr(awac.impl, attr))
    except Exception as e:
        print(f"Failed to inspect AWAC model: {e}")

    print("\n[CQL default model info]")
    try:
        cql = d3rlpy.algos.CQL()
        cql.build_with_dataset(dataset)
        print(cql)
        print("\nCQL implementation object:")
        print(cql.impl)

        for attr in ["q_func", "policy", "actor", "critic"]:
            if hasattr(cql.impl, attr):
                print(f"\n[cql.impl.{attr}]")
                print(getattr(cql.impl, attr))
    except Exception as e:
        print(f"Failed to inspect CQL model: {e}")

    print("\nNote:")
    print(textwrap.dedent(
        """
        - d3rlpy 0.90 의 내부 구현(impl)은 버전에 따라 속성 이름이 다를 수 있습니다.
        - 위 코드에서 OBS_DIM, ACT_DIM 을 실제 CARLA 환경의 관찰/행동 차원으로
          맞추면, d3rlpy가 사용하는 기본 MLP 구조를 그대로 볼 수 있습니다.
        - 논문에는 이 결과를 요약해서
          "two-layer MLP with 256 hidden units and ReLU activations"
          와 같은 형태로 정리해서 쓰면 됩니다.
        """
    ))

except ImportError as e:
    print("d3rlpy is not installed or cannot be imported.")
    print("Install with: pip install d3rlpy")
    print(f"Error detail: {e}")

print("\nDone.")

# =====================================================================
# 4) PYTHON LIBRARY VERSION CHECK
# =====================================================================
print("=" * 80)
print("4) PYTHON LIBRARY VERSIONS")
print("=" * 80)

import importlib

libraries = [
    "numpy",
    "pandas",
    "gym",
    "gymnasium",
    "torch",
    "torchvision",
    "stable_baselines3",
    "d3rlpy",
    "psutil",
    "matplotlib",
    "seaborn",
    "scipy",
    "sklearn",
    "yaml",
    "tqdm",
    "carla",
]

def get_version(lib_name):
    try:
        module = importlib.import_module(lib_name)
        return getattr(module, "__version__", "version attribute not found")
    except ImportError:
        return "NOT INSTALLED"

for lib in libraries:
    print(f"{lib:<20}: {get_version(lib)}")

print("\n")
