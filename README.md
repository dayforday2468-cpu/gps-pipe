# GPS Pipeline

GPS 데이터는 측위 오차, 불규칙한 샘플링, 순간적인 위치 튐 등 다양한 노이즈를 포함합니다.
따라서 GPS 데이터를 분석에 활용하기 위해서는 적절한 전처리와 오차 보정 과정이 필요합니다.

본 프로젝트는 Google Maps Timeline 데이터를 입력으로 받아 원시 위치 데이터를 분석 가능한 형태로 변환하는 GPS 데이터 처리 파이프라인입니다.

파이프라인은 다음 과정을 포함합니다.

- Sudden Position Jump 제거
- ST-DBSCAN 기반 이동 및 체류 데이터 분류
- Map Matching을 통한 도로 네트워크 기반 위치 보정
- Mobility Feature Extraction


## Pipeline Overview

전체 데이터 처리 과정은 다음과 같습니다.

```text
Google Timeline JSON
        ↓
데이터 추출 및 표준화
        ↓
시간 범위 필터링
        ↓
Sudden Position Jump 제거
        ↓
ST-DBSCAN 기반 이동 / 체류 클러스터링
        ↓
Map Matching
        ↓
Mobility Feature Extraction
```

Google Timeline JSON에서 필요한 위치 데이터를 추출한 뒤 내부에서 사용할 수 있는 형태로 표준화합니다.

이후 분석하고자 하는 시간 범위를 선택하고, 순간적으로 비정상적인 위치가 기록되는 Sudden Position Jump를 제거합니다.

정제된 위치 데이터에는 공간과 시간을 함께 고려하는 ST-DBSCAN을 적용하여 밀집된 위치 데이터를 클러스터링하고 이동 및 체류 패턴을 분석합니다.

이후 Map Matching을 통해 이동 경로를 실제 도로 네트워크에 대응시키고, 최종적으로 이동 거리, 체류 특성 등 후속 분석에 사용할 Mobility Feature를 추출합니다.


## Tech Stack

### Core

- Python 3.12
- Polars
- Pydantic
- ijson

### Analysis & Visualization

- Matplotlib

### Development

- Git


## Project Structure

```text
gps-pipe/
├── main.py
├── modules/
│   ├── __init__.py
│   ├── haversine.py
│   ├── sudden_position_jump.py
│   ├── dbscan.py
│   ├── dbscan_tuning.py
│   └── primitives/
│       ├── __init__.py
│       ├── config.py
│       ├── dataload.py
│       ├── datastore.py
│       ├── datafilter.py
│       ├── decorators.py
│       ├── logger.py
│       ├── pipeline.py
│       ├── schema.py
│       ├── timeutils.py
│       └── visualization.py
│
├── explorations/
│   ├── 01_raw_vs_timeline.py
│   ├── 02_distance_distribution.py
│   ├── 03_jump_segmentation_visualization.py
│   ├── 04_sudden_position_jump.py
│   └── dbscan/
│       ├── 01_spatial_k_dist_graph.py
│       ├── 02_temporal_k_dist_graph.py
│       └── 03_st_dbscan.py
│
├── data/
│   ├── timeline.json
│   ├── raw_positions.csv
│   ├── timeline_paths.csv
│   ├── visits.csv
│   ├── activities.csv
│   └── processed/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### `main.py`

전체 GPS 데이터 처리 파이프라인의 실행 순서를 관리합니다.

세부 알고리즘을 직접 구현하기보다 각 모듈을 호출하여 데이터가 파이프라인의 각 단계를 순서대로 통과하도록 구성합니다.


### `modules/`

GPS 데이터 처리에 사용되는 핵심 알고리즘을 포함합니다.

#### `haversine.py`

두 GPS 좌표 사이의 구면 거리를 계산합니다.

Haversine 거리 계산을 Polars Expression 형태로 제공하여 다른 알고리즘에서도 재사용할 수 있도록 구성합니다.


#### `sudden_position_jump.py`

GPS의 Sudden Position Jump를 제거합니다.

연속된 위치 사이의 거리를 이용하여 trajectory를 segment로 나누고, 이전 및 다음 segment와의 공간적 관계를 이용해 짧은 비정상 위치 segment를 탐지합니다.


#### `dbscan.py`

공간과 시간을 함께 고려하는 ST-DBSCAN 알고리즘을 구현합니다.

공간 거리와 시간 거리로 구성된 neighborhood를 이용하여 core point를 찾고, density-reachable한 point를 확장하여 cluster를 생성합니다.


#### `dbscan_tuning.py`

ST-DBSCAN 파라미터 분석을 위한 기능을 제공합니다.

공간 및 시간 k-distance를 계산하고 k-distance graph의 knee를 탐지하여 Spatial Eps와 Temporal Eps 후보를 결정하는 데 사용합니다.


### `modules/primitives/`

여러 알고리즘에서 공통으로 사용하는 기반 기능을 포함합니다.

#### `config.py`

데이터 경로, batch size 등 프로젝트의 공통 설정값을 관리합니다.


#### `dataload.py`

Google Timeline JSON을 streaming 방식으로 읽어 필요한 데이터를 추출합니다.

대용량 JSON을 한 번에 메모리에 올리지 않도록 `ijson`과 batch 처리를 사용합니다.


#### `datastore.py`

CSV 데이터의 저장과 로드를 담당합니다.

외부 데이터와 내부 표준 데이터 사이의 경계에서 schema validation을 수행합니다.


#### `datafilter.py`

GPS 및 Timeline 데이터를 특정 시간 범위로 필터링합니다.


#### `decorators.py`

실행 시간 측정 등 여러 모듈에서 공통으로 사용할 decorator를 제공합니다.


#### `logger.py`

프로젝트 전반에서 사용하는 logger를 설정합니다.


#### `pipeline.py`

데이터 디렉터리 초기화, Google Timeline 데이터 추출 및 CSV 생성 등 공통적인 파이프라인 초기화 과정을 관리합니다.


#### `schema.py`

CSV로 저장하거나 불러오는 데이터의 schema를 정의합니다.


#### `timeutils.py`

KST와 UTC 변환 등 시간 처리에 필요한 공통 기능을 제공합니다.


#### `visualization.py`

GPS point, trajectory 및 clustering 결과를 시각화하기 위한 기능을 제공합니다.


### `explorations/`

알고리즘을 실제 데이터에 적용하고 결과를 분석하기 위한 실험 코드를 포함합니다.

핵심 파이프라인 코드와 분리하여 알고리즘의 동작 과정과 파라미터 선택 과정을 확인할 수 있도록 구성합니다.


#### `01_raw_vs_timeline.py`

Google Timeline의 raw position과 semantic timeline 데이터를 비교합니다.


#### `02_distance_distribution.py`

연속된 GPS point 사이의 거리 분포를 분석합니다.


#### `03_jump_segmentation_visualization.py`

Sudden Position Jump 제거 과정에서 사용되는 trajectory segmentation을 시각화합니다.


#### `04_sudden_position_jump.py`

Sudden Position Jump 제거 전후의 GPS 데이터를 비교합니다.


### `explorations/dbscan/`

ST-DBSCAN의 파라미터 선택과 clustering 결과를 분석하기 위한 실험 코드를 포함합니다.


#### `01_spatial_k_dist_graph.py`

Spatial k-distance graph를 생성하고 knee를 탐지하여 Spatial Eps 후보를 분석합니다.


#### `02_temporal_k_dist_graph.py`

Temporal k-distance graph를 생성하고 knee를 탐지하여 Temporal Eps 후보를 분석합니다.


#### `03_st_dbscan.py`

선정한 Spatial Eps, Temporal Eps, MinPts를 이용하여 ST-DBSCAN을 실행하고 clustering 결과를 시각화합니다.


### `data/`

Google Timeline 원본 데이터와 파이프라인에서 생성한 데이터를 저장합니다.

```text
data/
├── timeline.json
├── raw_positions.csv
├── timeline_paths.csv
├── visits.csv
├── activities.csv
└── processed/
```

`timeline.json`은 사용자가 준비하는 원본 Google Timeline 데이터입니다.

다른 CSV 파일은 Timeline JSON에서 추출하고 표준화한 데이터이며 파이프라인 실행 과정에서 다시 생성됩니다.

`processed/`에는 시간 필터링, 전처리, clustering 등 실제 분석 파이프라인을 통과한 결과 데이터가 저장됩니다.


## Getting Started

### 1. Repository 준비

프로젝트 디렉터리로 이동합니다.

```bash
cd gps-pipe
```


### 2. Python 가상환경 생성

Python 3.12를 기준으로 가상환경을 생성합니다.

```bash
python3.12 -m venv .venv
```

가상환경을 활성화합니다.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```


### 3. Dependencies 설치

```bash
python -m pip install -r requirements.txt
```

프로젝트의 `modules` package를 현재 가상환경에서 사용할 수 있도록 editable mode로 설치합니다.

```bash
python -m pip install -e .
```


### 4. Google Timeline 데이터 준비

Google Timeline에서 내보낸 JSON 파일을 다음 위치에 배치합니다.

```text
data/
└── timeline.json
```


### 5. Pipeline 실행

```bash
python main.py
```

파이프라인을 실행하면 Google Timeline JSON에서 필요한 데이터를 추출하고 표준화된 CSV 데이터를 생성한 뒤 전처리 및 후속 분석 단계를 수행합니다.


### 6. Exploration 실행

각 알고리즘의 동작 및 분석 결과는 `explorations/`의 예제를 통해 확인할 수 있습니다.

예를 들어 Spatial k-distance graph를 확인하려면 다음과 같이 실행합니다.

```bash
python explorations/dbscan/01_spatial_k_dist_graph.py
```

ST-DBSCAN 결과를 확인하려면 다음과 같이 실행합니다.

```bash
python explorations/dbscan/03_st_dbscan.py
```


## Google Timeline 데이터 준비

이 프로젝트는 Google Maps 타임라인 데이터를 입력으로 사용합니다.

Google 공식 안내 문서는 아래 링크에서 확인할 수 있습니다.

- [Google Maps 타임라인 관리 및 데이터 내보내기 공식 문서](https://support.google.com/maps/answer/6258979?hl=ko&co=GENIE.Platform%3DAndroid)


### 먼저 알아두기

Google Maps의 타임라인 데이터 저장 방식은 이전과 달라졌습니다.

과거에는 Google 계정 및 서버를 중심으로 위치 기록을 관리하고 Google Takeout을 통해 관련 데이터를 내려받는 방식이 익숙했지만, Google은 Timeline 데이터를 **사용자의 기기에 저장하는 방식으로 전환**했습니다.

현재 타임라인 데이터는 각 기기에 저장되며, 사용자가 별도로 백업을 활성화한 경우 암호화된 백업 사본을 Google 서버에 저장할 수 있습니다.

따라서 현재 Android에서 타임라인 데이터를 내보낼 때는 Google Takeout이나 Google Maps 웹사이트가 아니라 **타임라인 데이터가 저장되어 있는 Android 기기에서 직접 내보내기**를 수행합니다.

> **Tip**
>
> 이 부분이 특히 헷갈릴 수 있습니다.
>
> `타임라인 데이터 내보내기`는 Google Maps 앱의 설정 메뉴가 아니라 Android의  
> **설정 앱 → 위치 → 위치 서비스 → 타임라인**에 있습니다.


### 타임라인 데이터 내보내기

1. Android 휴대전화 또는 태블릿에서 **설정 앱**을 엽니다.
2. **위치 → 위치 서비스 → 타임라인**으로 이동합니다.
3. **타임라인 데이터 내보내기**를 탭합니다.
4. **계속**을 탭합니다.
5. 데이터를 저장할 위치를 선택합니다.
6. **저장**을 탭합니다.

내보내기가 완료되면 **'내보내기 완료'** 팝업이 표시됩니다.

이때 데이터가 이메일로 전송되거나 별도의 Google 서비스에 업로드되는 것이 아니라,
**5번에서 직접 선택한 휴대전화의 저장 위치에 파일이 생성됩니다.**

따라서 내보내기가 완료된 후 Android의 **파일 앱**을 열어 저장할 때 선택했던 폴더로 이동하면 내보낸 Timeline JSON 파일을 확인할 수 있습니다.


### 프로젝트에 데이터 배치

내보낸 타임라인 JSON 파일을 PC로 옮긴 뒤 프로젝트의 `data` 디렉터리에 다음과 같이 배치합니다.

```text
data/
└── timeline.json
```

이후 `main.py`를 실행하면 Timeline JSON에서 필요한 데이터를 추출하여 파이프라인에서 사용할 CSV 데이터를 생성합니다.