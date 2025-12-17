# 파일 구조
- agents : 제어 모델 
  - pid_control.py : pid 기반 제어 로직 
  - sac.py : sac 모델을 정의한 부분  
  - sacrnd_model.py : 제안 기법 모델 정의 - SB3의 SAC를 override 한 뒤, RND 와 MCNet을 붙임. 
  - rnd.py : Random Netowrk Distillation 적용을 위한 모델 파일 

- env : carla 환경 세팅 및 경로 세팅
  - route.py : 차량이 주행할 경로 생성 
  - env_set.py : carla 환경과 연결 담당 , 차량 센서 설정 
  - wrapper.py : 모델 학습 환경 설정, 및 로깅 설정 

- utils : 시각화 등 도구 기능
  visualize.py : 지도 시각화, 경로 시각화 등 기능 담당 

- result : 실험 결과 csv 파일 들 
  final_result_town3 : town3 환경에서 얻은 실험 결과
  final_result_town4 : town4 환경에서 얻은 실험 결과 

- evaluate : 학습한 모델을 불러와 평가하는 코드 
  evaluate_town1.py : town1에서 학습한 모델을 불러와 평가
  evaluate_town4.py : town4에서 학습한 모델을 불러와 평가 

# 시뮬레이션 환경에서 강화학습 실행 or  오프라인 데이터 수집 실행 코드 
- run.py : PID controller로 데이터를 수집할 때 실행하는 코드
- run_proposed.py : 제안 기법으로 강화학습을 실행하는 코드
- awac_baseline.py : awac 학습을 실행하는 코드 
- baseline.py : cql 학습을 실행하는 코드 

# 주행 시작 포인트 
  town3 : (0,0)
  town4 : (300,-100)
    

# 시각화 or 결과 분석 코드 
- make_plot.py : waypoint 진행도를 시각화해서 나타내는 코드 (window_size = 100으로 smoothing)
- make_plot_collision_reached.py : 6번에 seed 동안 학습하는 training phase에서 집계된 충돌률, 도달률을 집계하는 코드
