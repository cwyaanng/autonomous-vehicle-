
## 파일 구조

### agents : 제어 모델
- pid_control.py : PID 기반 제어 로직
- sac.py : SAC 모델 정의
- sacrnd_model_revised.py : 제안 기법 모델 정의  
  (Stable-Baselines3 SAC를 override하여 RND와 MCNet을 결합)
- rnd.py : Random Network Distillation(RND) 모델 정의

---

### env : CARLA 환경 및 경로 설정
- route.py : 차량 주행 경로 생성
- env_set.py : CARLA 환경 연결 및 차량 센서 설정
- wrapper.py : 모델 학습 환경 구성 및 로깅 설정

---

### utils : 유틸리티
- visualize.py : 지도 시각화, 주행 경로 시각화

---

### result : 실험 결과 
- result/Town03_일반화실험_데이터 : 일반화 성능 실험 데이터 텐서보드 로그를 Town별로 나누어 저장한 폴더입니다. 
- result/Town03_학습_결과데이터 : Town03 환경 제안기법&베이스라인 모델 학습 결과
- result/Town04_학습_결과데이터 : Town04 환경 제안기법&베이스라인 모델 학습 결과

---

### logs : 일반화 성능 실험을 진행하면서 저장한 로그와 저장한 모델이 있는 폴더입니다.
- logs/TOWN03_학습모델_로그_저장된모델 : Town03에서 학습한 모델 및 학습 과정 텐서보드 로그가 저장되어 있습니다.
- logs/town별_설정한_경로 : 각 Town별로 모델 학습, 모델 테스트 때 설정한 경로를 시각화해 저장해놓은 폴더입니다. 


### 시뮬레이션 환경에서 강화학습 실행 또는 오프라인 데이터 수집 코드

- run.py : PID controller를 사용한 데이터 수집
- run_proposed.py : 제안 기법 강화학습 실행
- run_sac.py : SAC 학습 실행
- awac_baseline.py : AWAC 학습 실행
- baseline.py : CQL 학습 실행

---

## 시각화 및 결과 분석 코드

- make_plot.py : waypoint 진행도를 그래프로 시각화   
  (window_size = 100으로 smoothing 적용)

- make_plot_collision_reached.py :  
  6개 seed에 대해 training phase 동안 충돌로 끝난 에피소드 비율/ 도달로 끝난 에피소드 비율 bar chart로 시각화
  6개 seed에 대해 training phase 동안 에피소드 마지막에 도달한 waypoint의 평균을 bar chart로 시각화 

---

## 테스트 코드 

- evaluate_sac_and_proposed.py : 저장된 제안기법, SAC로 학습한 모델을 불러와 일반화성능 테스트를 하는 코드입니다.

- evaluate_cql_awac.py : 저장된 CQL , AWAC 모델을 불러와 일반화성능 테스트를 하는 코드입니다. 

- rnd_test.py : Random Network Distillation이 다른 주행 trajectory를 구별할 수 있는지 검증할 때 사용한 코드입니다. 

- make_plot_advantage.py : 각 기법끼리 얼마나 역전을 했는지 분석하고, plot으로 시각화하는 코드입니다. 

---


## 실행 절차

### 1. CARLA 컨테이너 실행

```bash
sudo docker run -it --rm \
  --name carla-server2 \
  --gpus all \
  --shm-size=4g \
  -p 2000-2002:2000-2002 \
  carlasim/carla:0.9.8 \
  ./CarlaUE4.sh -RenderOffScreen
```

---

### 2. 학습 진행 

사용할 기법(제안 기법, CQL, AWAC, SAC)에 따라 아래 명령어 중 하나를 선택하여 실행합니다.

#### 제안 기법

```bash
python run_proposed.py <start_x> <start_y> <town_name>
```

예시:

```bash
python run_proposed.py 0 0 Town03
nohup python -u run_proposed.py 0 0 Town03 > 파일이름.txt 2>&1 &
```

#### CQL

```bash
python cql_baseline.py <start_x> <start_y> <town_name>
```

예시:

```bash
python cql_baseline.py 0 0 Town03
nohup python -u cql_baseline.py 0 0 Town03 > 파일이름.txt 2>&1 &
```

#### AWAC

```bash
python awac_baseline.py <start_x> <start_y> <town_name>
```

예시:

```bash
python awac_baseline.py 0 0 Town03
nohup python -u awac_baseline.py 0 0 Town03 > 파일이름.txt 2>&1 &
```

#### SAC

```bash
python run_sac.py <start_x> <start_y> <town_name>
```

예시:

```bash
python run_sac.py 0 0 Town03
nohup python -u run_sac.py 0 0 Town03 > 파일이름.txt 2>&1 &
```

---

## 주행 시작 포인트
* Town01 : (0,0)
* Town02 : (250,250)
* Town03 : (0, 0)
* Town04 : (300, -100)
* Town04 : (0, 0)

---

### 3. 학습 완료 후

아래 명령어를 사용하여 CARLA 컨테이너를 종료합니다.

```bash
sudo docker ps
sudo docker stop -t 60 {container id}
```

또한 주기적으로 `sudo reboot`를 수행했습니다.
(하루에 한 번 reboot를 수행하였으며, 학습 중 segmentation fault 발생이나 서버 다운을 예방하기 위해 수행했습니다.)

---

## 지도 종류

Town01 / Town02 / Town03 / Town04 / Town05

---

## 모델 학습 

- 일반화 실험 때 모델 학습 후 저장된 모델과 로그, 주행한 경로의 이미지들은 logs 폴더에 있습니다. 

- result 폴더안에 일반화 실험 전에 학습을 실행하여 얻었던 결과들을 저장했습니다. training phase 동안 각 에피소드의 충돌여부(done_collided) / 도달여부(done_reached) / 에피소드가 나아간 waypoint 를 csv 파일 형태로 저장하였습니다. 
 

## 추가 문서

* docs/SETUP.md : 환경 설정 과정 및 라이브러리 버전 
* docs/EXPERIMENT.md : 앞으로 할 실험 절차 / 주의사항 / 현재 문제점 
