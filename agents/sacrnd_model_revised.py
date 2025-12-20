# agents/sacrnd_model.py
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

class MCNet(nn.Module):
    """
    상태와 행동을 입력받아 에피소드의 몬테 카를로 리턴을 예측하는 모델입니다. 
    Offline 데이터로 지도 학습을 수행하여 학습됩니다. 
    """
    def __init__(self, input_dim: int, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        print(f"dims:{dims}")
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        
        layers.append(nn.Linear(dims[-1], 1))  
        self.model = nn.Sequential(*layers)
        self.optimizer = th.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        return self.model(x)


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
        """
        배치 데이터 x를 받아 mean, var, count를 갱신합니다.
        """
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
    SB3의 SAC를 상속받아 제안 기법을 추가한 코드입니다.
    아래와 같은 부분이 포함되어 있습니다. 
    
    - offline data로 replay buffer 초기화 
    - MCNet (Monte carlo return 근사 네트워크) 학습
    - Random Network Distillation 으로 novelty 도출 및 보정 가중치로 이용
    """

    def __init__(
        self,
        env, 
        policy: str = "MlpPolicy", 
        learning_starts: int = 0, 
        target_entropy = -3,
        **sac_kwargs,
    ):
        # SAC 기본 설정
        if "learning_starts" not in sac_kwargs:
            sac_kwargs["learning_starts"] = learning_starts

        super().__init__(policy, env, target_entropy=target_entropy, **sac_kwargs) 
        print(f"target entropy : {target_entropy}")
        
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(self.device)
        
        # RND , MCNet 관련 설정 
        self.rnd = None 
        self.rnd_update_every = 5
        self.rnd_update_steps = 1
        
        # MC 리턴 계산을 위한 캐싱 변수 
        self._mc_cached_size: int = -1  
        self._mc_cached_pos  = -1
        self._mc_cached_full = False
        
        # 초기값만 0.5로 설정, 이후는 auto로 튜닝 
        self.ent_coef = 0.5
        
        # MCNet 모델 초기화 
        self.mc_targets = []
        self.mcnet = MCNet(input_dim=self.observation_space.shape[0] + self.action_space.shape[0]).to(self.device)
        
        # 정규화 객체 초기화 
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        self._nov_rms = RunningMeanStd(shape=None, device=self.device)  
        self.obs_rms  = RunningMeanStd(shape=obs_dim, device=self.device) 
        self.act_rms  = RunningMeanStd(shape=act_dim, device=self.device) 
           
        self.q_rms = RunningMeanStd(shape=None, device=self.device) 
        self.g_rms = RunningMeanStd(shape=None, device=self.device)  

        self.gamma = 0.99

        # 얼마나 보정되었는지 셀 용도로 만든 카운터입니다 
        self.calibrated = 0

    
    def _alpha(self) -> th.Tensor: 
        """ 현재 엔트로피 계수 얻는 메서드 """
        if self.ent_coef_optimizer is not None:
            with th.no_grad():
                return self.log_ent_coef.exp().detach() 
        if isinstance(self.ent_coef, (int, float)):
            return th.tensor(float(self.ent_coef), device=self.device)
        return self.ent_coef_tensor  
   
    
    def compute_mc_returns(self, gamma: float, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """ 
            몬테 카를로 리턴을 계산하는 메서드 
            
            Args : 
                rewards : offline data 안에 있는 reward값이 저장된 array
                dones : 에피소드 종료 지점 여부  
        """
        mc = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for i in reversed(range(len(rewards))):
            if bool(dones[i]):
                G = float(rewards[i])
            else:
                G = float(rewards[i]) + gamma * G
            mc[i] = G
        return mc

    def attach_rnd(self, rnd) -> None:
        """
        RND 모듈 연결하는 메서드 
        online learn 시작하기 전에 꼭 호출해야 합니다. 
        """
        self.rnd = rnd
        self.rnd.device = str(self.device)
        self._mc_returns = None
        self._mc_cached_size = -1


    def prefill_from_npz_folder_mclearn(self, data_dir: str, clip_actions: bool = True) -> int: 
        """
        data_dir 경로에 있는 offline data (.npz 형식의 파일) 들을 로드하여 replay buffer을 초기화하는 메서드입니다. 
        
        Args : 
            data_dir : offline data가 저장된 경로 
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

            for o, no, a, r, d in zip(obs, nobs, acts, rews, dones): 
                self.replay_buffer.add(
                    o[None, :],                          
                    no[None, :],                         
                    a[None, :],                          
                    np.array([float(r)], np.float32),     
                    np.array([bool(d)], np.float32),     
                    [{"TimeLimit.truncated": False}],    
                )
                
                all_rews.append(float(r))
                all_dones.append(bool(d))
                
                self.obs_rms.update(th.tensor(o).unsqueeze(0)) 
                self.act_rms.update(th.tensor(a).unsqueeze(0))  
            n_added += N
            n_files += 1

        mc_returns = self.compute_mc_returns(
            gamma=self.gamma,
            rewards=np.array(all_rews),
            dones=np.array(all_dones)
        )
        self.mc_targets = mc_returns.tolist()
        return n_added

  
    def update_rnd_with_critic_batch(self, batch: ReplayBufferSamples) -> th.Tensor:
        """현재 배치 안에 있는 데이터를 사용하여 RND 네트워크를 업데이트하는 메서드입니다."""
        if self.rnd is None:
            raise RuntimeError("RND is not attached. Call attach_rnd(rnd) first.")

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        #[배치별 정규화]
        obs_n = self.obs_rms.normalize(obs) 
        act_n = self.act_rms.normalize(act)
        
        x = th.cat([obs_n, act_n], dim=1)
        loss = self.rnd.update(x)

        return loss.detach()


    def train_mcnet_from_buffer(self, epochs=5, batch_size=512):
        """
        Replay Buffer의 상태(obs), 행동(act)과 replay buffer을 offline data로 초기화할 때 (prefill_from_npz_folder_mclearn 메서드에서) 계산해둔 몬테 카를로 리턴을 사용하여 MCNet 학습을 수행합니다. 
        """
        self.mcnet.train()
        dataset_size = len(self.mc_targets)
        
        buffer_len = self.replay_buffer.pos if not self.replay_buffer.full else self.replay_buffer.buffer_size

        assert dataset_size == buffer_len, f"mc_targets size mismatch ({dataset_size} vs {buffer_len})"

        for epoch in range(epochs):
            perm = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, batch_size):
                idxs = perm[i:i + batch_size]

                target = th.tensor(np.array(self.mc_targets)[idxs], dtype=th.float32).unsqueeze(-1).to(self.device)
             
                obs = self.replay_buffer.observations[idxs].squeeze(1)  
                act = self.replay_buffer.actions[idxs].squeeze(1)

                obs_n = self.obs_rms.normalize(obs)
                act_n = self.act_rms.normalize(act)
                x = th.cat([obs_n, act_n], dim=1)

                # print("obs.shape:", obs.shape)
                # print("act.shape:", act.shape)
                # print("target.shape:", target.shape)

                pred = self.mcnet(x)
                loss = F.mse_loss(pred, target)
                if i % 100 == 0:
                    p = pred[:5].detach().squeeze(-1).cpu().numpy()   # 앞 5개만
                    t = target[:5].detach().squeeze(-1).cpu().numpy()

                self.mcnet.optimizer.zero_grad()
                loss.backward()
                self.mcnet.optimizer.step()

            print(f"[MCNet] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            

    def tstats(self, tensor: th.Tensor, name: str = "") -> str:
        """
        텐서의 요약 통계를 문자열로 반환하는 메서드입니다.
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
        텐서들 중 NaN이 있는지 탐지하고 있으면 관련 정보를 출력합니다.
        옛날에 디버깅할 때 사용한 코드라 사용하실 일은 없을 것 같습니다!! 
        """
        for name, tensor in tensors.items():
            if not isinstance(tensor, th.Tensor):
                continue
            if th.isnan(tensor).any() or th.isinf(tensor).any():
                print(f"[NaN DETECTED] step {step} | phase: {phase} | tensor: '{name}'")
                print(f"→ Shape: {tuple(tensor.shape)}")
                print(f"→ Values (sample): {tensor.flatten()[:5].tolist()}")
                print(f"→ Stats: mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
                # Optional: raise error to halt training
                raise ValueError(f"NaN detected in tensor '{name}' during {phase} at step {step}.")

    
    def _tick_log(self, step: int, interval: int, message: str):
        """
        critic loss, actor loss 출력용 함수입니다.
        예전에 학습이 자꾸 붕괴될 때 디버깅을 위해 사용한 것이라 쓰실 일이 없을 것 같습니다!
        
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



    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """ 제안 기법 학습 루프를 구현한 부분입니다. """
        
        # 학습 세팅 (옵티마이저 설정)
        self.policy.set_training_mode(True)
        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        
        # 로깅을 하기 위한 리스트 정의 
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        
        for gradient_step in range(gradient_steps):
            global_update = self._n_updates + gradient_step
            
            """ 1. replay buffer에서 데이터를 샘플링합니다. """
            rb = self.replay_buffer
            size = rb.buffer_size if rb.full else rb.pos
            batch_inds = np.random.randint(0, size, size=batch_size)  
            replay_data = rb._get_samples(batch_inds)  

            """ 2. entropy 계수를 업데이트 합니다. """
            # 현재 정책의 행동과 log propbability를 계산합니다. 
            actions_pi, log_prob = self.policy.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.view(-1, 1)
            
            ent_coef_loss = None
            
            # 설정된 Target Entropy를 유지하도록 자동으로 alpha 값을 조절합니다. 
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
                
                
            """ 3. RND 네트워크를 업데이트 합니다. """
            if self.rnd is None:
                raise RuntimeError("RND is not attached. Call attach_rnd(rnd) before learn().")  
            # Critic 업데이트에 사용된 배치로 RND 내부 신경망도 똑같이 학습합니다. 
            self.update_rnd_with_critic_batch(replay_data)
            
            """ 4. Critic 네트워크를 업데이트합니다. """
            with th.no_grad():
                
                # [Critic 업데이트 1] 다음 상태에서의 행동과 Q값을 예측합니다. 
                next_actions, next_log_prob = self.policy.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(
                    self.policy.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                # [Critic 업데이트 2] Q-learning target을 계산합니다. 
                target_q_sac = replay_data.rewards + (1.0 - replay_data.dones) * float(self.gamma) * (
                    next_q_values - ent_coef * next_log_prob.view(-1, 1)
                )                
                
                """ obs, action을 정규화하여 RND에 입력으로 넣어 novelty를 계산합니다. -> 이것이 얼마나 보정할 지 결정하는 가중치가 됩니다. """
                obs_n = self.obs_rms.normalize(replay_data.observations)  
                act_n = self.act_rms.normalize(replay_data.actions)       
                mc_in = th.cat([obs_n, act_n], dim=1)
                nov = self.rnd.novelty(mc_in) 
                self._nov_rms.update(nov.detach().cpu().numpy())
                mean_t = th.tensor(self._nov_rms.mean, device=self.device)
                std_t  = th.tensor(self._nov_rms.var ** 0.5, device=self.device)
                nov_z = (nov - mean_t) / (std_t + 1e-6)
                w = th.sigmoid(nov_z.abs() - 1.0) 
                w = w * 0.9                      
             
            # [Critic 업데이트 3] Critic loss를 계산합니다. 
            current_q1, current_q2 = self.policy.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q_sac) + F.mse_loss(current_q2, target_q_sac))
            critic_losses.append(float(critic_loss.detach().cpu().numpy()))

            # [Critic 업데이트 4] 계산한 loss를 이용해 업데이트 합니다. 
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            """ 5. Actor loss에 들어가는 Q 값을 보정합니다. """
            
            # [Calibration 1] 두개의 Q값중 최솟값을 구합니다. 
            q1_pi, q2_pi = self.policy.critic(replay_data.observations, actions_pi) 
            min_q_pi = th.min(q1_pi, q2_pi)

            with th.no_grad():
                deterministic_action = self.policy._predict(replay_data.observations, deterministic=True)

            # [Calibration 2] 정규화된 observation, action을 이용하여 몬테 카를로 근사 값을 구합니다. 
            obs_n = self.obs_rms.normalize(replay_data.observations)
            act_n = self.act_rms.normalize(deterministic_action)
            g_pi = self.mcnet(th.cat([obs_n, act_n], dim=1)).detach()
            
            # 몇개의 Q값이 보정된건지 카운트하여 보정 기법이 작동하는지, 필요한지 체크합니다. 
            cali_q = th.minimum(g_pi,min_q_pi)
            mask = (g_pi < min_q_pi)
            self.calibrated += int(mask.sum().item())
       
            # Q값 보정 부분 
            # 수식 : (1-w) * 원래 Q값 + w * 몬테 카를로 근사 값 
            # w (novelty) 가 클수록 몬테 카를로 근사 값으로 보정하는 비중을 높입니다 
            w = w.detach()
            if w.dim() == 1:
                w = w.view(-1, 1)

            self.q_rms.update(min_q_pi.detach()) 
            self.g_rms.update(g_pi.detach())

            calibrated_q = (1 - w) * min_q_pi + w * cali_q


            if global_update % 5000 == 0:
                print(f"[Update {global_update}]")
                print("w:", self.tstats(w, "w"))
                print("q_pi:", self.tstats(min_q_pi, "q_pi"))
                print("mc_q:", self.tstats(g_pi, "mc_q"))
                print("logp:", self.tstats(log_prob, "logp"))
                print("alpha:", float(ent_coef.detach().cpu().numpy()))
                print("cal_q:", self.tstats(calibrated_q, "calibrated_q"))
                print(f"cal_q_count:{self.calibrated}")
                print("-" * 50)

            """ 6. 보정된 Q 값을 이용하여 actor loss를 계산하고 actor 을 업데이트합니다. """
            actor_loss = (ent_coef * log_prob - calibrated_q).mean()
            actor_losses.append(float(actor_loss.detach().cpu().numpy()))

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()

            """ 7. 타겟 네트워크를 업데이트합니다. """
            if global_update % self.target_update_interval == 0:
                polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

                if hasattr(self, "batch_norm_stats") and hasattr(self, "batch_norm_stats_target"):
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
             
                self._nan_report(global_update, "online",
                    target_q_sac=target_q_sac, current_q1=current_q1, current_q2=current_q2,
                    critic_loss=critic_loss, log_prob=log_prob, min_q_pi=min_q_pi,
                    g_pi=g_pi, calibrated_q=calibrated_q, actor_loss=actor_loss)

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
