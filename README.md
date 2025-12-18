````markdown
## 파일 구조

### agents : 제어 모델
- pid_control.py : PID 기반 제어 로직
- sac.py : SAC 모델 정의
- sacrnd_model.py : 제안 기법 모델 정의  
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
- 실험 결과 CSV 파일 저장 디렉토리
- final_result_town3 : Town03 환경 실험 결과
- final_result_town4 : Town04 환경 실험 결과

---

### evaluate : 평가 코드
- evaluate_town1.py : Town01에서 학습된 모델 평가
- evaluate_town4.py : Town04에서 학습된 모델 평가

---

## 시뮬레이션 환경에서 강화학습 실행 또는 오프라인 데이터 수집 코드

- run.py : PID controller를 사용한 데이터 수집
- run_proposed.py : 제안 기법 강화학습 실행
- run_sac.py : SAC 학습 실행
- awac_baseline.py : AWAC 학습 실행
- baseline.py : CQL 학습 실행

---

## 시각화 및 결과 분석 코드

- make_plot.py : waypoint 진행도 시각화  
  (window_size = 100으로 smoothing 적용)
- make_plot_collision_reached.py :  
  6개 seed에 대해 training phase 동안의 충돌률 및 도달률 집계

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
````

---

### 2. 학습 코드 실행

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

* Town03 : (0, 0)
* Town04 : (300, -100)

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

## 추가 문서

* docs/SETUP.md

```
```
