import gym
import numpy as np
from gym import spaces
from agents.pid_control import compute_reward, find_closest_waypoint_index, find_closest_waypoints_ahead, make_obs_from_waypoints,compute_errors
import carla
import os 
from datetime import datetime
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_forward_route, generate_route , generate_route_town3, generate_route_town2, generate_route_town1, generate_route_town4
from run import calculate_spawn_transform
import utils.visualize as viz
from torch.utils.tensorboard import SummaryWriter

class CarlaWrapperEnv(gym.Env):
    
    """
    CARLA 시뮬레이터를 Gym 인터페이스로 wrapping 한 클래스입니다. 
    강화학습 에이전트가 CARLA 환경과 상호작용할 수 있게 하기 위해 구현한 클래스입니다. 
    """
    
    
    def __init__(self, client, world, carla_map, points, simulation, target_speed=22.0, test=False):
        super(CarlaWrapperEnv, self).__init__()
        """ 1. CARLA 기본 객체 설정 """
        self.client = client
        self.world = world
        self.map = carla_map
        self.blueprint_library = world.get_blueprint_library()
        
        """ 2. 경로, 타겟 속도, 차량, 충돌 센서 등 변수 초기화 """
        self.route_waypoints = None
        self.target_speed = target_speed
        self.vehicle = None
        self.collision_sensor = None
        self.collision_event = {"collided": False}
        self.test = test
        
        """ 3. 로깅 및 정지 임계 속도, 정지 신호 설정 """
        self.writer = SummaryWriter(log_dir=simulation)
        # 정지라고 판단하는 임계 속도 
        self.speed_stop_threshold = 1
        # 정지한 step 수를 카운트하는 변수 
        self.stop_count = 0
        self.camera_save_root = simulation+"/driving_scene" 
        
        """
        
        4. Gym space 정의
        - observation space : 차량 앞의 15개의 waypoint의 좌표 (차량 기준 상대 좌표) + heading error, lateral error 
        
        - action space : steer, throttle, brake 
        
        """
        self.observation_space = spaces.Box(
            low=np.array([-100.0,-100.0]*15 + [-np.pi/2, -10.0]),
            high=np.array([100.0, 100.0]*15 + [np.pi/2, 10.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_idx = 0
        self.simulation_category = simulation
        self.points = points 
        
        self.route_waypoints = None 
        
        self.actual_path_x = []
        self.actual_path_y = []
        
    
        
        
        """ 5. 맵별로 경로를 생성하는 부분 """
        if self.map.name == "Town01":
            end_point = (400, 350)
            self.route_waypoints =  generate_route_town1(self.map, self.points , end_point) 
        elif self.map.name == "Town02":
            self.route_waypoints = generate_route_town2(self.map, self.points)
        elif self.map.name == "Town03":
            self.route_waypoints = generate_route_town3(self.map, self.points)
        elif self.map.name == "Town04":
            self.route_waypoints = generate_route_town4(self.map, self.points)
        else :
            self.route_waypoints = generate_forward_route(self.map, self.points)
            
        """ 위에서 생성한 경로를 시각화하여 이미지로 저장 """
        viz.generate_planned_path_plot(self.route_waypoints,  self.simulation_category)
        if not self.route_waypoints:
            print("경로 생성 실패")
            return
        self.max_index = len(self.route_waypoints) - 1
        
        """ 
        6. CARLA 시뮬레이션 동기화 설정 
        - 0.05초로 시간 간격을 고정하여 시뮬레이션을 진행했습니다. 
        
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        """ 
        로깅을 하기 위해 세는 카운트 변수 
        """
        self._global_step = 0 
        self._episode_return = 0.0 # 에피소드의 총 보상 
        self.episode_num = 0 # 에피소드 수 
        self.count_step = 0 # 진행된 step 수 카운트
       
    def reset(self):
        """
        새로운 에피소드를 시작할 때 환경을 초기화하는 함수입니다.
        
        [동작 순서]
        1. 이전 에피소드에서 사용한 차량, 센서를 제거합니다.(제거를 하지 않을 경우 충돌이 나거나 시뮬레이션이 죽는 경우가 많았습니다)
        
        2. 새로운 차량을 spawn 합니다.
        
        3. 초기 observation을 계산하고 리턴합니다. 
        
        """
        
        """ 1. 변수 초기화 """
        self._cleanup()
        self.stop_count = 0 
        self.count_step = 0
        self.actual_path_x = []
        self.actual_path_y = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_return = 0.0
        self.episode_speed = 0
        
        """ 2. 차량 spwan - 최대 5번 재시도 """
        for attempt in range(5):
            try: 
                
                """ 시작 지접 계산 및 차량 생성 """
                spawn_transform = calculate_spawn_transform(self.route_waypoints)
                spawn_transform = self.map.get_waypoint(spawn_transform.location, project_to_road=True).transform
                self.vehicle = spawn_vehicle(self.world, self.blueprint_library, spawn_transform)
                
                """ 충돌 센서 부착 """
                self.collision_sensor, self.collision_event = attach_collision_sensor(self.world, self.vehicle)
                
                """ 시뮬레이션 1틱 진행 """
                self.world.tick()
                
                """ 초기 상태 관측 및 observation 계산 """
                loc = self.vehicle.get_location()
                vel = self.vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y, vel.z])  
                
                """ 현재 차량 위치에서 가장 가까운 waypoint를 찾고 해당 waypoint를 타겟으로 지정 """
                waypoints_ahead = find_closest_waypoints_ahead(loc, self.route_waypoints)
                idx = find_closest_waypoint_index(loc, self.route_waypoints)
                target_idx = min(idx + 5, len(self.route_waypoints) - 1)
                target_wp = self.route_waypoints[target_idx]
                
                """ heading error, lateral error 계산 """
                e, theta_e = compute_errors(self.vehicle, target_wp)
                
                """ 차량이 이동한 경로를 리스트에 저장하여 기록 """
                self.actual_path_x.append(loc.x)
                self.actual_path_y.append(loc.y)
                
                """ 관측값 생성 -> 강화학습 모델 입력으로 들어갑니다. """
                obs = make_obs_from_waypoints(self.vehicle, waypoints_ahead, e, theta_e).astype(np.float32, copy=False)
                return obs

            except Exception as e:
                print(f"[RESET ERROR] 차량 스폰 실패 시도한 횟수{attempt+1} : {e}")
                import traceback
                traceback.print_exc()

        raise RuntimeError("[RESET ERROR] 차량 스폰 5회 모두 실패.")

    def step(self, action):
        
        """
        강화학습 모델에서 action(steer, throttle, brake)을 받아 1 step의 시뮬레이션을 진행하고 결과를 반환합니다. 
        
        returns :
            obs : 관측값
            reward : 보상
            done : 에피소드 종료 여부 
        """
        
        self.count_step += 1
        info = {}
        
        """ 1. carla 환경에 input으로 받은 action을 적용 """
        steer, throttle, brake = float(action[0]) , float(action[1]) , float(action[2]) 
        brake = 0

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)
        self.world.tick()
        
        """ 2. 속도를 확인하고 정지 상태인지 확인 """
        speed_vec = self.vehicle.get_velocity()  
        speed = np.linalg.norm([speed_vec.x, speed_vec.y, speed_vec.z])
        
        if speed <= self.speed_stop_threshold:
            self.stop_count += 1
        else :
            self.stop_count = 0
 
        """ 3. 현재 위치 기반으로 가장 가까운 waypoint를 찾아 다음 target waypoint로 삼기 """
        loc = self.vehicle.get_location() 
        waypoints_ahead = find_closest_waypoints_ahead(loc, self.route_waypoints)
        idx = find_closest_waypoint_index(loc, self.route_waypoints)
        target_idx = min(idx + 5, len(self.route_waypoints) - 1) 
        target_wp = self.route_waypoints[target_idx]
        
        """ 4. heading error, lateral error 계산 """
        e, theta_e = compute_errors(self.vehicle, target_wp)
        obs = make_obs_from_waypoints(self.vehicle, waypoints_ahead, e, theta_e).astype(np.float32, copy=False)

        """ 5. 차량이 지나간 경로 기록 """
        self.actual_path_x.append(loc.x)
        self.actual_path_y.append(loc.y)

        """ 6. 충돌 감지 / 종료인지 감지 / 보상 계산 """
        collided = self.collision_event['collided']
        
        reached = (target_idx >= len(self.route_waypoints)-1) 
        done = collided or reached
        
        reward = compute_reward(obs, self.vehicle, collided=collided, reached=reached)
        
        self._episode_return += float(reward)
        self.episode_speed += speed 
        self._global_step += 1

        """ 차량이 너무 오래 멈추어 있거나, 한 에피소드당 2만스텝을 넘어가면 종료 """
        if self.stop_count >= 2000:
            reward -= 5
            done = True 
        
        if self.count_step > 20000:
            done = True
        
        info["metrics"] = {
            "reward/step": float(reward),
            "vehicle/speed_mps": float(speed),
            "progress/target_idx": int(target_idx),
            "status/stop_count": int(self.stop_count),
            "status/collided": int(bool(collided)),
            "status/reached": int(bool(reached)),
            "control/steer": float(steer),
            "control/throttle": float(throttle),
            "control/brake": float(brake),
        }
        self.episode_speed = self.episode_speed / self.count_step 
        
        if "metrics" in info:
            for k, v in info["metrics"].items():
                self.writer.add_scalar(k, v, self._global_step)
        
        
        """
        
        에피소드 종료 시 주행 경로를 시각화 하고
        로깅을 하는 부분입니다 
        
        """
        if done:
            
            if self.count_step > 0:
                avg_reward = self._episode_return / self.count_step
                
            self.count_step = 0
            self.episode_num += 1
            try:
                viz.generate_actual_path_plot(
                    route_waypoints=self.route_waypoints,
                    actual_path_x=self.actual_path_x,
                    actual_path_y=self.actual_path_y,
                    simulation_category=self.simulation_category,
                    timestamp=self.timestamp,
                )
                info["plot_saved"] = True
            except Exception as ex:
                print(f"[PLOT ERROR] {ex}")
                info["plot_saved"] = False
                
            info["episode"] = {
                "reward/episode_return": self._episode_return,
                "episode/episode_speed" : self.episode_speed,
                "episode/episode_average_reward" : avg_reward,
                "episode/done_collided": int(bool(collided)),
                "episode/done_reached": int(bool(reached)),
                "episode/waypoint_ahead" : target_idx 
            }
            
            for k, v in info["episode"].items():
                self.writer.add_scalar(k, v, self.episode_num) 
                
            self._cleanup()
        return obs, reward, done , info

    def _cleanup(self):
        
        """
        에피소드 종료 후 환경을 리셋할 때 차량과 센서를 제거하는 함수입니다. 
        """
        
        
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass
        self.camera = None

        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
        except Exception:
            pass
        self.collision_sensor = None

        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
        except Exception:
            pass
        self.vehicle = None
        
    def close(self):
        self.writer.close()
        super().close()