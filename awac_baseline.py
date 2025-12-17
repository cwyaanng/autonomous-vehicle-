# baseline.py  (d3rlpy 0.x 호환 버전)

import sys

from test import BATCH_SIZE
sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')  # CARLA egg
import os
import numpy as np
import carla  

import d3rlpy
from d3rlpy.algos import AWAC
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy import metrics

from env.env_set import connect_to_carla
from env.wrapper import CarlaWrapperEnv
from datetime import datetime


DATA_DIR = "offline_data_for_replaybuffer/dataset_town03"
LOG_ROOT = "logs/일반화실험_AWAC_모델저장"

DO_OFFLINE_PRETRAIN = False
PRETRAIN_STEPS = 50_000          
PRETRAIN_STEPS_PER_EPOCH = 5_000 

ONLINE_STEPS = 1_500_000
ONLINE_STEPS_PER_EPOCH = 10_000   

REPLAY_MAXLEN = 4_000_000
USE_GPU = True 

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

def build_mdpdataset_from_npz_dir(data_dir: str) -> MDPDataset:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    obs_chunks, act_chunks, rew_chunks, term_chunks = [], [], [], []
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files in {data_dir}")

    for fname in npz_files:
        path = os.path.join(data_dir, fname)
        d = np.load(path, allow_pickle=False)

        obs   = d["observations"].astype(np.float32)                # (N, obs_dim)
        acts  = d["actions"].astype(np.float32)                     # (N, act_dim)
        rews  = d["rewards"].astype(np.float32).reshape(-1)         # (N,1) -> (N,)
        terms = d["terminals"].astype(bool).reshape(-1)             # (N,1) -> (N,)

        N = min(obs.shape[0], acts.shape[0], rews.shape[0], terms.shape[0])
        obs, acts, rews, terms = obs[:N], acts[:N], rews[:N], terms[:N]
        
        if term_chunks:
            term_chunks[-1][-1] = True

        obs_chunks.append(obs)
        act_chunks.append(acts)
        rew_chunks.append(rews)
        term_chunks.append(terms)

    observations = np.concatenate(obs_chunks, axis=0)
    actions      = np.concatenate(act_chunks, axis=0)
    rewards      = np.concatenate(rew_chunks, axis=0)
    terminals    = np.concatenate(term_chunks, axis=0)

    print("[Dataset] shapes:",
          "obs", observations.shape,
          "act", actions.shape,
          "rew", rewards.shape,
          "term", terminals.shape)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,  # d3rlpy 0.x: timeouts 인자 없음
    )
    print("[Dataset] size (episodes):", len(dataset))
    return dataset


def make_env(log_root: str, start_point):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    client, world, carla_map = connect_to_carla()
    env = CarlaWrapperEnv(
        client=client,
        world=world,
        carla_map=carla_map,
        points=start_point,
        simulation=os.path.join(log_root, now),
    )
    return env


def main(start_point):
    dataset = build_mdpdataset_from_npz_dir(DATA_DIR)

    awac = AWAC(use_gpu=USE_GPU, batch_size = 256)

    def prefill_buffer_from_dataset(dataset: MDPDataset, buffer: ReplayBuffer):
        import numpy as np

        obs   = np.asarray(dataset.observations)
        acts  = np.asarray(dataset.actions)
        rews  = np.asarray(dataset.rewards).reshape(-1)

        if hasattr(dataset, "terminals"):
            terms = np.asarray(dataset.terminals)
        elif hasattr(dataset, "dones"):
            terms = np.asarray(dataset.dones)
        else:
            raise AttributeError("Dataset has neither 'terminals' nor 'dones'.")

        terms = terms.reshape(-1).astype(bool)

        N = len(acts)
        for i in range(N):
            o = np.asarray(obs[i]).squeeze()
            a = np.asarray(acts[i]).squeeze()
            r = float(np.asarray(rews[i]).squeeze())
            d = bool(np.asarray(terms[i]).squeeze())

            if d:
                buffer.append(o, a, r, True, True)   
            else:
                buffer.append(o, a, r, False)      


    print("Online Fine-Tuning 시작")
    env = make_env(LOG_ROOT, start_point)
    buffer = ReplayBuffer(maxlen=REPLAY_MAXLEN, env=env)
    prefill_buffer_from_dataset(dataset, buffer)

    try:
        awac.fit_online(
            env,
            buffer,
            n_steps=ONLINE_STEPS,
            n_steps_per_epoch=ONLINE_STEPS_PER_EPOCH
        )
    except TypeError:
        awac.fit_online(
            env,
            buffer,
            n_steps=ONLINE_STEPS,
        )

    out_path = f"awac_offline_online_{NOW}.d3"
    awac.save_model(out_path)
    print(f"Fine-Tuning 완료, 모델 저장 → {out_path}")

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("실행 명령어 형식: python awac_baseline.py <start_x> <start_y>")
        sys.exit(1)
    
    start_x = float(sys.argv[1])
    start_y = float(sys.argv[2])
    start_point = (start_x, start_y)
    
    main(start_point)

