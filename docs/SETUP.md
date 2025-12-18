# 환경 설정 (Environment Setup)
이 문서에서는 하드웨어와 시스템 정보, 라이브러리 버전 정보, 그리고 연구실 wise4 서버에서 어떤 과정으로 환경을 세팅했는지 적었습니다.

---

## 1. 하드웨어 및 시스템 정보

| 항목     | 사양                                      |
| ------ | --------------------------------------- |
| OS     | Linux 5.15 (x86_64, Debian Bullseye 기반) |
| Python | 3.7.16                                  |
| CPU    | 28 cores                                |
| RAM    | 62.6 GB                                 |
| GPU    | NVIDIA GeForce RTX 4090 (1 GPU)         |
| CUDA   | 사용 가능 (PyTorch CUDA 11.7)               |

---

## 2. CARLA 시뮬레이터 설정

* **CARLA 버전**: 0.9.8
* CARLA는 Docker로 설치 
* Python API는 CARLA 설치 디렉토리 내 `PythonAPI`를 사용

## 3. Python 라이브러리 버전

| 라이브러리             | 버전                 |
| ----------------- | ------------------ |
| Python            | 3.7.16             |
| NumPy             | 1.21.6             |
| Pandas            | 1.3.5              |
| SciPy             | 1.7.3              |
| scikit-learn      | 1.0.2              |
| PyYAML            | 6.0.1              |
| tqdm              | 4.67.1             |
| gym               | 0.26.2             |
| gymnasium         | 0.28.1             |
| stable-baselines3 | 1.8.0              |
| d3rlpy            | 0.90               |
| PyTorch           | 1.13.1 (CUDA 11.7) |
| matplotlib        | 3.5.3              |
| seaborn           | 0.12.2             |
| psutil            | 7.0.0              |


## 4. 버전 관련 이슈

* 다른 라이브러리와의 호환성 때문에 CARLA 0.9.8 을 사용하였습니다.(Gym , numpy, pygame 들 이 충돌이 난 것으로 기억하여 CARLA 버전을 0.9.8 로 다운그레이드 한 것으로 기억하고 있습니다..)

## 5. 환경 세팅 과정 

### 1. Python 3.7 설치
### 2. Docker 다운로드

```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl status docker
```
### 3. Docker로 Carla 이미지 다운로드

```bash
sudo docker pull carlasim/carla:0.9.8
``` 

### 4. Docker이 GPU를 사용할 수 있도록 설정
NVIDIA Docker 리포지토리 등록을 하고 설치를 진행했습니다.

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list && \
sudo apt update && \
sudo apt install -y nvidia-container-toolkit && \
sudo systemctl restart docker

# 패키지 목록 업데이트
sudo apt-get update

# nvidia-container-runtime 설치
sudo apt-get install -y nvidia-container-runtime

# daemon.json 생성 

sudo nano /etc/docker/daemon.json

{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# 도커 재시작 
sudo systemctl restart docker

``` 

- 마지막으로 아래와 같은 명령어로 CARLA 컨테이너를 실행하여 잘 설치되었는지 확인합니다.

```bash
sudo docker run -it --rm \
  --name carla-server2 \
  --gpus all \
  --shm-size=4g \
  -p 2000-2002:2000-2002 \
  carlasim/carla:0.9.8 \
  ./CarlaUE4.sh -RenderOffScreen
```
