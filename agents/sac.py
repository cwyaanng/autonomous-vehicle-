# agents/sac.py
"""
이 파일은 SB3 SAC 구현체를 오버라이드 하여 replay buffer을 offline data로 초기화하고
online training을 수행하기 위한 코드입니다. 

"""



import glob
import os
from typing import Optional
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update 
import torch.nn as nn
import pickle
from datetime import datetime
NOW = datetime.now().strftime("%Y%m%d_%H%M")


class RunningMeanStd:
    def __init__(self, shape=None, eps=1e-4, device="cpu"):
        
        """
        온라인으로 데이터의 평균과 분산을 업데이트하는 클래스입니다.
        """

        import torch as th
        self.device = th.device(device)
        if shape is None:
            self.mean  = th.tensor(0.0, device=self.device, dtype=th.float32)
            self.var   = th.tensor(1.0, device=self.device, dtype=th.float32)
        else:
            if isinstance(shape, int):
                shape = (shape,)
            self.mean  = th.zeros(shape, device=self.device, dtype=th.float32)
            self.var   = th.ones(shape,  device=self.device, dtype=th.float32)
        self.count = th.tensor(eps, device=self.device, dtype=th.float32)

    @th.no_grad()
    def update(self, x):
        """배치 데이터 x를 받아 mean, var, count를 갱신합니다."""
        import torch as th, numpy as np
        if not isinstance(x, th.Tensor):
            x = th.as_tensor(x, dtype=th.float32, device=self.device)

        if self.mean.ndim == 0:
            x = x.view(-1) 
            batch_mean = x.mean()
            batch_var  = x.var(unbiased=False)
            batch_count = x.shape[0]
        else:
            x = x.to(self.device, dtype=th.float32)  
            batch_mean = x.mean(dim=0)              
            batch_var  = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x, eps=1e-8):
        """현재까지의 통계치를 바탕으로 입력값 x를 정규화합니다."""
        import torch as th
        if not isinstance(x, th.Tensor):
            x = th.as_tensor(x, dtype=th.float32, device=self.device)
        return (x.to(self.device, dtype=th.float32) - self.mean) / th.sqrt(self.var + eps)


class SACOfflineOnline(SAC):
    """
    SB3 SAC를 상속받아 Offline data로 replay buffer을 초기화하여 사용하는 기능을 추가한 클래스입니다.
    """

    def __init__(
        self,
        env, 
        policy: str = "MlpPolicy", 
        learning_starts: int = 0, 
        target_entropy = -3,
        **sac_kwargs,
    ):
        if "learning_starts" not in sac_kwargs:
            sac_kwargs["learning_starts"] = learning_starts

        super().__init__(policy, env, target_entropy=target_entropy, **sac_kwargs) 
        print(f"target entropy : {target_entropy}")
        
        self.rnd = None 
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(self.device)
 
        self._mc_cached_size: int = -1  
        self._mc_cached_pos  = -1
        self._mc_cached_full = False
        self.ent_coef = 0.5
        
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        self._nov_rms = RunningMeanStd(shape=None, device=self.device)   # 스칼라 RMS (novelty)
        self.obs_rms  = RunningMeanStd(shape=obs_dim, device=self.device) # 차원별 RMS
        self.act_rms  = RunningMeanStd(shape=act_dim, device=self.device) # 차원별 RMS
           
        self.q_rms = RunningMeanStd(shape=None, device=self.device)  # Q 통계 (스칼라)
        self.g_rms = RunningMeanStd(shape=None, device=self.device)  # MCNet 통계 (스칼라)

        self.gamma = 0.99

        self.calibrated = 0
   

    def load_mcnet(self, path: str):
        ckpt = th.load(path, map_location=self.device)
        self.mcnet.load_state_dict(ckpt["mcnet"])
        if "mcnet_opt" in ckpt:
            self.mcnet.optimizer.load_state_dict(ckpt["mcnet_opt"])
        for name, rms in [("obs_rms", self.obs_rms), ("act_rms", self.act_rms)]:
            rms.mean  = ckpt[name]["mean"].to(self.device)
            rms.var   = ckpt[name]["var"].to(self.device)
            rms.count = ckpt[name]["count"].to(self.device)
        print(f"[MCNet] Loaded from {path}")


    def _alpha(self) -> th.Tensor: 
        if self.ent_coef_optimizer is not None:
            with th.no_grad():
                return self.log_ent_coef.exp().detach() 
        if isinstance(self.ent_coef, (int, float)):
            return th.tensor(float(self.ent_coef), device=self.device)
        return self.ent_coef_tensor  


    def prefill_from_npz_folder_mclearn(self, data_dir: str, clip_actions: bool = True) -> int: 
        """
        인자로 오프라인 데이터 (.npz 형식) 가 들어있는 경로를 넣으면, 해당 경로에 있는 데이터로 replay buffer을 초기화합니다. 
        
        Args: 
            data_dir : 데이터 경로 
            
        """
        self.mc_targets = [] 
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        act_low = getattr(self.action_space, "low", None)
        act_high = getattr(self.action_space, "high", None)

        n_added, n_files = 0, 0
        all_rews = []
        all_dones = []
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                obs = d["observations"].astype(np.float32)
                acts = d["actions"].astype(np.float32)
                rews = d["rewards"].astype(np.float32).reshape(-1, 1)
                nobs = d["next_observations"].astype(np.float32)
                dones = d["terminals"].astype(np.float32).reshape(-1, 1)

            N = min(len(obs), len(acts), len(rews), len(nobs), len(dones))
           
            if N == 0:
                continue
            obs, acts, rews, nobs, dones = obs[:N], acts[:N], rews[:N], nobs[:N], dones[:N]

            if clip_actions and act_low is not None and act_high is not None:
                acts = np.clip(acts, act_low, act_high)

            for o, no, a, r, d in zip(obs, nobs, acts, rews, dones): # 전이 하나씩 삽입이라는데 
                self.replay_buffer.add(
                    o[None, :],                           # (1, obs_dim)
                    no[None, :],                          # (1, obs_dim)
                    a[None, :],                           # (1, act_dim)
                    np.array([float(r)], np.float32),     # (1,)
                    np.array([bool(d)], np.float32),      # (1,)
                    [{"TimeLimit.truncated": False}],     # info list 길이 = n_envs(=1)
                )
                
                all_rews.append(float(r))
                all_dones.append(bool(d))
                
                self.obs_rms.update(th.tensor(o).unsqueeze(0))  # shape (1, obs_dim)
                self.act_rms.update(th.tensor(a).unsqueeze(0))  # shape (1, act_dim)
            n_added += N
            n_files += 1

        return n_added

    def tstats(self, tensor: th.Tensor, name: str = "") -> str:
        """
        텐서의 요약 통계를 문자열로 반환하는 메서드입니다. 
        
        Args:
            tensor : 통계를 출력할 텐서
            name : 변수 이름(옵션)
        Returns:
            str: 정리된 통계 문자열
        """
        if not isinstance(tensor, th.Tensor):
            return f"{name}: Not a tensor"

        tensor = tensor.detach().cpu()
        stats = {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "shape": list(tensor.shape)
        }

        return (
            f"{name} | shape: {stats['shape']} | "
            f"mean: {stats['mean']:.4f} | std: {stats['std']:.4f} | "
            f"min: {stats['min']:.4f} | max: {stats['max']:.4f}"
        )

    def _nan_report(self, step: int, phase: str = "train", **tensors):
        """
        텐서들 중 NaN이 있는지 탐지하고 있으면 관련 정보를 출력하는 메서드입니다.
        예전에 에러날 때 디버깅하려고 쓴 것이라 거의 쓸 일이 없으실 것 같습니다! 

        Args:
            step : 현재 스텝 수
            phase : 학습 단계 이름 (예: "train", "online", "pretrain" 등)
            tensors : 검사할 텐서들 (키=이름, 값=텐서)
        """
        for name, tensor in tensors.items():
            if not isinstance(tensor, th.Tensor):
                continue
            if th.isnan(tensor).any() or th.isinf(tensor).any():
                print(f"[NaN DETECTED] step {step} | phase: {phase} | tensor: '{name}'")
                print(f"→ Shape: {tuple(tensor.shape)}")
                print(f"→ Values (sample): {tensor.flatten()[:5].tolist()}")
                print(f"→ Stats: mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
   
                raise ValueError(f"NaN detected in tensor '{name}' during {phase} at step {step}.")

    
    def _tick_log(self, step: int, interval: int, message: str):
        """
        critic loss, actor loss 출력용 함수입니다.
        이것도 예전에 학습이 자꾸 붕괴될 때 디버깅을 위해 사용한 것이라 쓰실 일이 없을 것 같습니다!
        
        Args :
            step : 현재 step 또는 업데이트 수
            interval : 로그 출력 간격
            message : 출력할 메시지
        """
        if step % interval == 0:
            print(message)

    # @th.no_grad()
    # def _actor_log_prob(self, obs: th.Tensor):
    #     """
    #     현재 Actor 정책을 사용하여 행동과 로그 확률을 반환합니다. 
    #     단순히 값 확인용으로 필요한 경우 사용하면 될 것 같습니다! 
    #     """
    #     a, logp = self.policy.actor.action_log_prob(obs)
    #     if logp.dim() == 1:
    #         logp = logp.view(-1, 1)
    #     return a, logp


    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """ 학습 루프 부분입니다. """

        self.policy.set_training_mode(True)
        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        for gradient_step in range(gradient_steps):
            global_update = self._n_updates + gradient_step
            
            """ 1. 리플레이 버퍼에서 배치를 뽑는다. """
            rb = self.replay_buffer
            size = rb.buffer_size if rb.full else rb.pos
            batch_inds = np.random.randint(0, size, size=batch_size)  
            replay_data = rb._get_samples(batch_inds)  

            """ 2. Entropy 계수 자동 튜닝 """
            actions_pi, log_prob = self.policy.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.view(-1, 1)
            ent_coef_loss = None
            
            """ 엔트로피 계수 업데이트 """ 
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.detach().cpu().numpy()))
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                
            """ 3. Critic 업데이트 """
            with th.no_grad():
                """ Target Action & Q값 계산 """
                next_actions, next_log_prob = self.policy.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(
                    self.policy.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_sac = replay_data.rewards + (1.0 - replay_data.dones) * float(self.gamma) * (
                    next_q_values - ent_coef * next_log_prob.view(-1, 1)
                )                
            """ 현재 Q값과 loss 계산 """
            current_q1, current_q2 = self.policy.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q_sac) + F.mse_loss(current_q2, target_q_sac))
            critic_losses.append(float(critic_loss.detach().cpu().numpy()))

            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            """ 4. Actor 업데이트 """
            q1_pi, q2_pi = self.policy.critic(replay_data.observations, actions_pi) 
            min_q_pi = th.min(q1_pi, q2_pi)

            with th.no_grad():
                deterministic_action = self.policy._predict(replay_data.observations, deterministic=True)

            self.q_rms.update(min_q_pi.detach()) 

            actor_loss = (ent_coef * log_prob - min_q_pi ).mean()
            actor_losses.append(float(actor_loss.detach().cpu().numpy()))

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()
            
            """ 로깅 """
            if global_update % 5000 == 0:
                print(f"[Update {global_update}]")
                print("q_pi:", self.tstats(min_q_pi, "q_pi"))
                print("logp:", self.tstats(log_prob, "logp"))
                print("alpha:", float(ent_coef.detach().cpu().numpy()))
                print("cal_q:", self.tstats(min_q_pi , "calibrated_q"))
                print(f"cal_q_count:{self.calibrated}")
                print("-" * 50)
            
            """ 타깃 네트워크(Critic) 업데이트  """
            if global_update % self.target_update_interval == 0:
                polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

                if hasattr(self, "batch_norm_stats") and hasattr(self, "batch_norm_stats_target"):
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
             
                self._nan_report(global_update, "online",
                    target_q_sac=target_q_sac, current_q1=current_q1, current_q2=current_q2,
                    critic_loss=critic_loss, log_prob=log_prob, min_q_pi=min_q_pi,actor_loss=actor_loss)

                self._tick_log(global_update, 1000,
                    f"[onl] up {global_update} | crit:{critic_loss.item():.4f} | actor:{actor_loss.item():.4f}")

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))

    def online_learn(self, total_timesteps=300_000, tb_log_name: str = NOW, log_interval: int = 10, callback=None):
  
        return self.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            log_interval=log_interval,
            callback=callback,
        )
