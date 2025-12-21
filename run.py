"""
run.py 

비례, 적분, 미분 gain을 변경하면서 PID controller로 다양한 주행 데이터를 수집하기 위한 코드입니다. 

"""





import os
import sys

sys.path.append('/home/wise/chaewon/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla

from agents.pid_control import run_pid_drive_with_log
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_forward_route_for_PID, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
import itertools

def calculate_spawn_transform(route_waypoints):
    """
        경로의 시작점과, 바로 다음 지점을 이용하여 차량이 경로와 같은 방향으로 
        spawn 되도록 위치와 회전각을 계산합니다. 
    """
    if len(route_waypoints) < 2:
        raise ValueError("웨이포인트가 충분하지 않습니다.")

    wp1 = route_waypoints[0].transform.location
    wp2 = route_waypoints[1].transform.location

    dx = wp2.x - wp1.x
    dy = wp2.y - wp1.y
    yaw = math.degrees(math.atan2(dy, dx))

    # 차량이 바닥에 끼어서 spawn 시 에러가 나지 않게 z축을 0.5m 띄워서 spawn 합니다. 
    spawn_transform = carla.Transform(
        location=carla.Location(x=wp1.x, y=wp1.y, z=wp1.z + 0.5),
        rotation=carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )
  
    return spawn_transform

def pid_control(client, world, carla_map, start_coords, filename):
    """
    설정된 경로에 대해 다양한 PID gain 조합으로 주행 시뮬레이션을 반복 실행하는 부분입니다. 
    """
    import itertools
    param_combinations = itertools.product(
        [1.0, 3.0, 5.0],           # kp_s : steer 반응성 
        [0.0, 0.01, 0.02],         # ki_s : steer 누적 오차 보정
        [0.0, 0.3, 0.6],           # kd_s : steer 진동 억제 
        [0.3, 0.6],                # ke : heading error gain

                                    # 가속 부분 P,I,D gain
        [0.5, 1.0],                # kp_t 
        [0.0, 0.05],               # ki_t 
       # [0.0, 0.05]                # kd_t 
        [0.0, 0.05] 
    )
    
    blueprint_library = world.get_blueprint_library()
    a = 0

    """
     
    위에서 지정한 PID gain 조합에 대해 반복문을 실행하면서 
    PID controller을 통한 주행을 실행합니다. 
    
    """
    for i, (kp_s, ki_s, kd_s, ke, kp_t, ki_t, kd_t) in enumerate(param_combinations):
  
        """ 주행 경로 생성 """
        route_waypoints = generate_forward_route_for_PID(carla_map, start_coords)
        
        if not route_waypoints:
            print("경로 생성 실패")
            return

        """ 차량 spawn 위치 계산 """
        spawn_transform = calculate_spawn_transform(route_waypoints)
        spawn_transform = carla_map.get_waypoint(
            spawn_transform.location, project_to_road=True
        ).transform

        """ 차량 및 센서 생성 """
        vehicle = spawn_vehicle(world, blueprint_library, spawn_transform)
        collision_sensor, collision_event = attach_collision_sensor(world, vehicle)

        # 필요 시 주석 해제하여 카메라 센서도 부착 
        # camera_sensor = attach_camera_sensor(world, vehicle, save_path="logs/driving_scene")

        actual_path_x, actual_path_y = [], []

        try:
            """ 주행 수행 및 로깅 (pid_control.py 파일 참고 )"""
            run_pid_drive_with_log(
                world, vehicle, route_waypoints,
                actual_path_x, actual_path_y, collision_event,
                kp_s, ki_s, kd_s, ke, kp_t, ki_t, kd_t, a, filename
            )
        finally:
            """ 센서 정리 후 차량 정리 """
            try:
                collision_sensor.destroy()
            except Exception:
                pass
            try:
                vehicle.destroy()
            except Exception:
                pass

        a += 1

        # 로그/그림 생성 후 닫기
        if a%30 == 0:
            generate_actual_path_plot(route_waypoints, actual_path_x, actual_path_y, filename)
            try:
                plt.close('all')
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("wrong argument : python run.py [PID|MPC|BEHAVIOR|...]\nexample : python run.py PID")
        sys.exit(1)

    mode = sys.argv[1].upper()
    print(mode)
    
    client, world, carla_map = connect_to_carla("Town04")
    visualize_all_waypoints(carla_map)
    
    if mode == "PID":
        # start_coords = (0, 50)
        # end_coords = (50, 0)
        # pid_control(client, world, carla_map , start_coords, "route_1")
        
        start_coords = (0, 100)
        end_coords = (0, -200)
        
        pid_control(client, world, carla_map , start_coords, "route_2")
        start_coords = (-450, 350)
        end_coords = (0, 0) 
        pid_control(client, world, carla_map , start_coords,  "route_3")
        
        start_coords = (250, -100)
        end_coords = (350, -200) 
        pid_control(client, world, carla_map , start_coords, "route_4")
        
        start_coords = (-350, 400)
        end_coords = (0, 0) 
        pid_control(client, world, carla_map , start_coords,  "route_5")
        
        start_coords = (200,100)
        end_coords = (50,0)
        pid_control(client, world, carla_map , start_coords, "route_6")
        
        start_coords = (100,100)
        end_coords = (0,300)
        pid_control(client, world, carla_map , start_coords, "route_7")
        
               
    if mode == "COLLECTION":
        start_coords = (0,300)
        pid_control(client, world, carla_map,start_coords, "route_6")
        
        start_coords = (200,0)
        pid_control(client, world, carla_map,start_coords, "route_7")
    else:
        print(f"❌ wrong argument")
