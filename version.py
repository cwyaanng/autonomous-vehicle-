
"""

하드웨어와 라이브러리 환경, 모델 구조를 출력하는 코드입니다. 


"""

import os
import platform
import textwrap

print("=" * 80)
print("1) HARDWARE / SYSTEM INFORMATION")
print("=" * 80)


print(f"OS          : {platform.platform()}")
print(f"Python      : {platform.python_version()}")
print(f"CPU cores   : {os.cpu_count()}")

try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"RAM total   : {mem.total / (1024 ** 3):.2f} GB")
except ImportError:
    print("psutil not installed. Install with `pip install psutil` for RAM info.")


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


print("=" * 80)
print("2) STABLE-BASELINES3 SAC MODEL ARCHITECTURE")
print("=" * 80)

try:
    from stable_baselines3 import SAC
    import gym 

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
        print("\n[SAC actor network (mu/head)]")
        try:
            print(model.policy.mu)
        except Exception as e:
            print(f"Could not access actor mu network: {e}")

        print("\n[SAC critic / Q-networks]")
        try:
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


print("=" * 80)
print("3) D3RLPY AWAC / CQL MODEL ARCHITECTURE (TEMPLATE)")
print("=" * 80)

try:
    import d3rlpy
    import numpy as np
    from d3rlpy.dataset import MDPDataset

    print(f"d3rlpy version: {d3rlpy.__version__}")

 
    OBS_DIM = 32 
    ACT_DIM = 3    

    n_dummy = 32
    observations = np.zeros((n_dummy, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_dummy, ACT_DIM), dtype=np.float32)
    rewards = np.zeros((n_dummy,), dtype=np.float32)
    terminals = np.zeros((n_dummy,), dtype=bool)
    dataset = MDPDataset(observations, actions, rewards, terminals)

    print("\n[AWAC default model info]")
    try:
        awac = d3rlpy.algos.AWAC()
      
        awac.build_with_dataset(dataset)
        print(awac)
        print("\nAWAC implementation object:")
        print(awac.impl)

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

except ImportError as e:
    print("d3rlpy is not installed or cannot be imported.")
    print("Install with: pip install d3rlpy")
    print(f"Error detail: {e}")

print("\nDone.")


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
