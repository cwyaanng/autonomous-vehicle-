"""
  pid_control.py
  
  역할 
    - PID Controller을 구현해 주행을 하고 강화학습을 위한 offline data를 저장하는 용도의 코드입니다. 
    
"""
import math
import time
import numpy as np
import carla
import os

def find_closest_waypoint_index(loc , route_waypoints):
    """
      현재 차량 위치에서 가장 가까이에 있는 waypoint를 찾아 리턴  
      
      Args : 
        loc : 현재 차량의 위치
        route_waypoints : 현재 target 경로 안에 들어 있는 waypoints 
        
      return : 
       현재 차량 가장 근처의 waypoint
    """
    min_dist = float('inf')
    closest_idx = 0
    for i, wp in enumerate(route_waypoints):
        dist = wp.transform.location.distance(loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            
    return closest_idx

def find_closest_waypoints_ahead(loc, route_waypoints, num_ahead=15):
    """
    현재 차량 위치에서 이후의 num_ahead개 waypoint 반환
    만약 남은 waypoint 수가 부족하면, 부족한 만큼 zero-padding
    """
    closest_idx = find_closest_waypoint_index(loc, route_waypoints)
    waypoints = route_waypoints[closest_idx:closest_idx + num_ahead]

    # Zero-padding
    while len(waypoints) < num_ahead:
        waypoints.append(None)

    return waypoints

def make_obs_from_waypoints(vehicle, waypoints, e, theta_e):
    """
    Waypoint들을 차량 기준 좌표계로 변환합니다.
    그리고 waypoint에 lateral error, heading error을  포함하는 1D state 벡터를 생성합니다. 
    
    Args:
        vehicle : carla.Vehicle
        waypoints : 앞선 15개( num_ahead개 ) 의 Waypoint의 리스트 
        e : lateral error 
        theta_e : heading erro 
    """
    # 차량 pose
    tf = vehicle.get_transform()
    vehicle_loc = tf.location
    vehicle_yaw = math.radians(tf.rotation.yaw)

    cos_yaw = math.cos(-vehicle_yaw)
    sin_yaw = math.sin(-vehicle_yaw)
    
    obs = []

    for wp in waypoints:
        if wp is None or not hasattr(wp, 'transform'):
            obs.extend([0.0, 0.0])
        else:
            loc = wp.transform.location
            dx = loc.x - vehicle_loc.x
            dy = loc.y - vehicle_loc.y
            # 차량 기준 상대좌표
            rel_x = dx * cos_yaw - dy * sin_yaw
            rel_y = dx * sin_yaw + dy * cos_yaw
            obs.extend([rel_x, rel_y])

    obs.extend([e, theta_e])
    
    obs_array = np.array(obs, dtype=np.float32)
    return obs_array 

def compute_errors(vehicle, target_wp):
    """
    자율주행 차량 제어에서 필요한 에러 계산 
    lateral error (차량이 목표 경로에서 얼마나 옆으로 벗어났는지)
    heading error (차량의 진행 방향이 목표 경로와 얼마나 각도 차이가 있는지)
    
    Args : 
      vehicle : 차량 객체
      target_wp : 현재 차량과 가장 가까이에 있는 waypoint - 목표 waypoint 
    
    return : 
      lateral_error, heading_error 
    """
    vehicle_transform = vehicle.get_transform()
    vehicle_loc = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    forward_vec = np.array([np.cos(vehicle_yaw), np.sin(vehicle_yaw)])
    target_loc = target_wp.transform.location
    to_target = np.array([target_loc.x - vehicle_loc.x, target_loc.y - vehicle_loc.y])
    dist = np.linalg.norm(to_target)
    if dist < 1e-6:
        return 0.0, 0.0

    target_vec = to_target / dist
    dot = np.clip(np.dot(forward_vec, target_vec), -1.0, 1.0)
    heading_error = np.arccos(dot)
    cross = forward_vec[0] * target_vec[1] - forward_vec[1] * target_vec[0]
    if cross < 0:
        heading_error *= -1

    normal_vec = np.array([-target_vec[1], target_vec[0]])
    offset_vec = np.array([vehicle_loc.x - target_loc.x, vehicle_loc.y - target_loc.y])
    lateral_error = np.dot(offset_vec, normal_vec)

    return lateral_error, heading_error

def compute_reward(obs, vehicle, collided=False, reached=False):
    """
    주행 리워드 계산
    
    speed 가 높을수록 점수가 크고, lateral & heading error 가 클수록 점수가 깎입니다.
    
    충돌 시 -200의 페널티를 주고, 완주 시 200의 reward 을 주었습니다. 
    
    """
    *_, lateral_error, heading_error = obs
    speed_vec = vehicle.get_velocity()
    speed = np.linalg.norm([speed_vec.x, speed_vec.y, speed_vec.z])

    # 보상 함수 
    reward = abs(speed * np.cos(heading_error)) -  abs(speed * np.sin(heading_error)) -  abs(speed * lateral_error) 

    if collided:
        reward = -200.0  
        
    if reached:
        reward = 200.0   
        
    return reward

def save_episode_as_npz(path, data_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data_dict)

def pid_steering_control(e, theta_e, previous_error, integral, dt, kp, ki, kd, ke):
    """
    차량의 조향 각도를 PID 제어 방식으로 계산
    

    Args:
        e : lateral error, 차량이 목표 경로에서 얼마나 벗어났는지 (좌우 거리)
        theta_e : heading error, 차량의 진행 방향과 경로 방향 사이의 각도 오차 (rad)
        previous_error : 이전 루프의 lateral error 값
        integral : 이전까지 누적된 오차 값
        dt : 루프 간 시간 간격 (초)

    Returns:
        steer : [-1.0, 1.0] 범위의 조향 명령 값
        integral : 현재까지 누적된 오차 값 (다음 루프에 사용됨)
    """

    prop = kp * e
    integral += ki * e * dt
    integral = np.clip(integral, -0.5, 0.5)
    derivative = kd * (e - previous_error) / dt
    
    steer = prop + integral + derivative - ke * theta_e
    steer = -np.clip(steer, -1.0, 1.0)
    return steer, integral
  
def pid_throttle_control(target_speed, current_speed, prev_error, integral, dt , kp, ki, kd):
  
    """
    목표 속도 (target_speed)를 추종하도록 차량의 속도를 PID 제어 방식으로 계산했습니다.

    Args:
        target_speed : 목표 속도 (m/s)
        current_speed : 현재 속도 (m/s)
        prev_error : 이전 루프에서의 속도 오차
        integral : 이전까지 누적된 속도 오차
        dt : 루프 간 시간 간격 (s))

    Returns:
        throttle : 가속 명령 값 (0.0 ~ 1.0)
        brake : 브레이크 명령 값 (0.0 ~ 1.0)
        error : 현재 속도 오차
        integral : 현재까지 누적된 속도 오차 (다음 루프에 사용됨)
    """

    error = target_speed - current_speed
    integral += error * dt
    derivative = (error - prev_error) / dt

    output = kp * error + ki * integral + kd * derivative

    throttle = np.clip(output, 0.0, 1.0)  
    brake = 0.0
    if throttle < 0.05 and error < 0: 
        brake = np.clip(-output, 0.0, 1.0)
        throttle = 0.0

    return throttle, brake, error, integral

def run_pid_drive_with_log(world, vehicle, route_waypoints, actual_path_x, actual_path_y, collision_event, kp_s, ki_s, kd_s, ke, kp_t, ki_t, kd_t, num, filename, max_steps=3500 ,output_dir = "dataset_town1_real" ):
  
    """
      PID Controller을 이용해 주행을 하고, 
      그 과정에서 발생한 state, action, reward, next state 을 저장합니다. 
      
      마지막에는 각 에피소드별로 .npz 파일을 저장하여 추후 강화학습에서 Demonstration data로 사용하고자 했습니다. 
    """
    
    """ pid control을 위한 변수 초기화 """
    steer_integral = 0.0 # 오차 누적값 
    steer_prev_error = 0.0 # 바로 직전 스텝의 오차 
    steer_prev = 0.0 # 바로 직전의 핸들 값 

    throttle_integral = 0.0 # 속도 오차 누적값 
    throttle_prev_error = 0.0 # 바로 직전 속도의 오차 

    dt = 0.1
    target_speed = 10.0  # 목표 속도 
    
    # 하나의 에피소드 동안의 주행 기록을 저장할 리스트 입니다 
    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = [], [], [], [], []
    
    last_idx = len(route_waypoints) - 1
    
    # 시뮬레이션 동기화 모드 설정 
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()
    
    """ max_steps 횟수만큼 pid controller을 이용해 주행 명령을 내리고 실제 주행을 수행합니다 """
    for t in range(max_steps):
        
        """ 1. 현재 차량 위치, 속도 측정 """
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y, vel.z])  
        
        """ 2. 타겟 waypoint 계산 """
        waypoints_ahead = find_closest_waypoints_ahead(loc, route_waypoints)
        idx = find_closest_waypoint_index(loc, route_waypoints)
        target_idx = min(idx + 5, len(route_waypoints) - 1)
        target_wp = route_waypoints[target_idx]

        """ 3. 타겟 waypoint와 현재 지점 사이의 lateral error, heading error 계산  """
        e, theta_e = compute_errors(vehicle, target_wp)
        
        """ 4. 현재 차량 상태에서의 state 값 생성 """
        obs = make_obs_from_waypoints(vehicle, waypoints_ahead , e , theta_e)
        
        """ 5. 위에서 계산한 에러를 PID controller 에 넣어서 steer, throttle 명령 생성 """
        steer, steer_integral = pid_steering_control(e, theta_e, steer_prev_error, steer_integral, dt, ki_s, kp_s, kd_s, ke)
        steer_prev_error = e
        steer = 0.7 * steer + 0.3 * steer_prev 
        steer_prev = steer
      
        throttle, brake, throttle_prev_error, throttle_integral = pid_throttle_control(
            target_speed, speed, throttle_prev_error, throttle_integral, dt , kp_t, ki_t, kd_t
        )

        """ 6. 생성한 명령을 차량에 적용하고 시뮬레이션 수행 """
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.apply_control(control)
        world.tick()
        
        
        """ 7. next state 생성 및 reward 계산 """
        next_loc = vehicle.get_location()
        next_waypoints = find_closest_waypoints_ahead(next_loc, route_waypoints)
        next_idx = find_closest_waypoint_index(loc, route_waypoints)
        next_target_idx = min(next_idx + 5, len(route_waypoints) - 1)
        next_target_wp = route_waypoints[next_target_idx]
        ne, ntheta_e = compute_errors(vehicle, next_target_wp)
        next_obs = make_obs_from_waypoints(vehicle, next_waypoints, ne , ntheta_e)
        
        collided = collision_event['collided']
        reached = (next_target_idx >= last_idx)  
        done = collided or reached
        
        reward = compute_reward(next_obs, vehicle, collided=collided, reached=reached)

        """ 8. 리스트에 앞서 계산한 현재 상태(obs) , 제어 명령, reward, 다음 상태(next_obs) , 종료 여부 추가 """
        obs_buf.append(obs)
        act_buf.append([steer, throttle, brake])
        rew_buf.append(reward)
        next_obs_buf.append(next_obs)
        done_buf.append(done)
        
        actual_path_x.append(next_loc.x)
        actual_path_y.append(next_loc.y)

        
        if done:
            if collided:
                print(f"[{t}] collision occur - simulation stop")
            elif reached:
                print(f"[{t}] goal reached - simulation stop")
            break
            
    """ 9. 주행 하나가 끝나면 .npz 파일로 주행 데이터 저장 """    
    save_episode_as_npz(
        f"{output_dir}/{filename}_route_episode_{num:04d}.npz",
        {
            "observations": np.array(obs_buf),
            "actions": np.array(act_buf),
            "rewards": np.array(rew_buf),
            "next_observations": np.array(next_obs_buf),
            "terminals": np.array(done_buf)
        }
    )
    
    print(f"[SAVE] {filename}_route_episode_{num:04d}.npz 저장 완료!")
    print(f"  - observations: {len(obs_buf)}개")
    print(f"  - actions: {len(act_buf)}개")
    print(f"  - rewards: {len(rew_buf)}개")
    print(f"  - next_observations: {len(next_obs_buf)}개")
    print(f"  - terminals: {len(done_buf)}개")


          