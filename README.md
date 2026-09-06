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
Trajectory Segmentation
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

이후 분석하고자 하는 시간 범위를 선택하고, 연속된 GPS point 사이의 거리를 기준으로 trajectory를 segment 단위로 분할합니다.

각 segment의 평균 위치와 point 수, 시작 및 끝 position, 이전 및 다음 segment 사이의 거리 등을 계산합니다. 이 정보를 이용하여 순간적으로 비정상적인 위치가 기록되는 Sudden Position Jump를 탐지하고 제거합니다.

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
│   ├── dbscan.py
│   ├── haversine.py
│   ├── map_matching.py
│   ├── parameter_tuning.py
│   ├── projection.py
│   ├── road_network.py
│   ├── segmentation.py
│   ├── sudden_position_jump.py
│   └── primitives/
│       ├── __init__.py
│       ├── config.py
│       ├── datafilter.py
│       ├── dataload.py
│       ├── datastore.py
│       ├── decorators.py
│       ├── logger.py
│       ├── pipeline.py
│       ├── schema.py
│       ├── timeutils.py
│       └── visualization.py
├── explorations/
│   ├── 01_raw_vs_timeline.py
│   ├── sudden_position_jump/
│   │   ├── 01_jump_threshold.py
│   │   ├── 02_jump_segmentation_visualization.py
│   │   ├── 03_same_place_threshold.py
│   │   └── 04_jump_removal_visualization.py
│   ├── dbscan/
│   │   ├── 01_spatial_k_dist_graph.py
│   │   ├── 02_temporal_k_dist_graph.py
│   │   └── 03_st_dbscan.py
│   └── map_matching/
│       ├── 01_road_network.py
│       ├── 02_projection.py
│       ├── 03_road_k_distance.py
│       ├── 04_candidate_projection.py
│       └── 05_map_matching.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### `main.py`

전체 GPS 데이터 처리 파이프라인의 실행 순서를 관리합니다.  
세부 알고리즘은 각 모듈에 위임하고 파이프라인 단계 간 데이터 흐름을 연결합니다.

### `modules/`

GPS 데이터 처리와 Map Matching에 사용하는 핵심 알고리즘을 포함합니다.

#### `dbscan.py`

공간과 시간을 함께 고려하는 ST-DBSCAN을 수행합니다.  
클러스터링 결과와 함께 연속된 이동 구간 정보를 생성합니다.

#### `haversine.py`

위도와 경도로 표현된 두 GPS 좌표 사이의 구면 거리를 계산합니다.  
거리 기반 전처리와 파라미터 추정에서 공통으로 사용합니다.

#### `map_matching.py`

도로 후보점을 생성하고 emission/transition probability와 Viterbi 알고리즘을 이용해 Map Matching을 수행합니다.  
GPS 이동 경로를 실제 도로 네트워크상의 후보 경로로 보정합니다.

#### `parameter_tuning.py`

Sudden Position Jump, ST-DBSCAN, Map Matching에 필요한 거리 기반 파라미터를 추정합니다.  
k-distance와 knee, quantile 등을 이용해 데이터 기반 후보값을 계산합니다.

#### `projection.py`

GPS 좌표와 도로 geometry를 동일한 평면 좌표계에서 다룰 수 있도록 변환합니다.  
GPS point를 도로 edge에 투영하는 기하 연산도 제공합니다.

#### `road_network.py`

GPS 위치 범위에 맞는 OSM 도로 네트워크를 로드하고 캐시합니다.  
분석 시에는 필요한 범위로 자른 뒤 무방향 `MultiGraph` 형태로 제공합니다.

#### `segmentation.py`

연속 GPS point 사이의 거리 변화를 기준으로 trajectory를 segment 단위로 분할합니다.  
각 segment의 평균 위치, 시작·끝 position, point 수 등의 요약 정보를 생성합니다.

#### `sudden_position_jump.py`

segment 정보를 이용해 순간적으로 비정상적인 위치가 기록된 Sudden Position Jump를 탐지합니다.  
탐지 결과는 `position_id`와 jump 여부의 관계로 반환합니다.

### `modules/primitives/`

여러 알고리즘에서 공통으로 사용하는 데이터 입출력, 설정, schema, 시각화 등의 기반 기능을 포함합니다.

#### `config.py`

데이터 경로, cache 경로, 후보 수 등 프로젝트 전반의 공통 설정값을 관리합니다.

#### `datafilter.py`

GPS 및 Timeline 데이터를 지정한 시간 범위로 필터링합니다.

#### `dataload.py`

Google Timeline JSON을 streaming 방식으로 읽어 필요한 위치 데이터를 추출합니다.  
대용량 JSON을 한 번에 메모리에 올리지 않도록 `ijson`과 batch 처리를 사용합니다.

#### `datastore.py`

파이프라인에서 사용하는 CSV 데이터의 저장과 로드를 담당합니다.  
외부 데이터와 내부 표준 데이터의 경계에서 schema validation도 수행합니다.

#### `decorators.py`

실행 시간 측정 등 여러 모듈에서 공통으로 사용할 decorator를 제공합니다.

#### `logger.py`

프로젝트 전반에서 사용하는 logger 설정을 제공합니다.

#### `pipeline.py`

데이터 디렉터리 초기화와 Google Timeline 데이터 추출 등 공통 초기화 과정을 관리합니다.

#### `schema.py`

파이프라인에서 사용하는 Pydantic schema를 정의합니다.  
원본 위치, segment, cluster, movement, projection, candidate 등 단계별 데이터 구조를 관리합니다.

#### `timeutils.py`

KST와 UTC 변환 등 시간 처리에 필요한 공통 기능을 제공합니다.

#### `visualization.py`

GPS point, trajectory, clustering, 도로 네트워크 등의 결과를 시각화하는 공통 기능을 제공합니다.

### `explorations/`

각 알고리즘을 실제 데이터에 적용하고 파라미터와 결과를 단계별로 확인하기 위한 실험 코드를 포함합니다.

#### `01_raw_vs_timeline.py`

Google Timeline의 raw position과 semantic timeline 데이터를 비교합니다.

### `explorations/sudden_position_jump/`

Sudden Position Jump 탐지에 필요한 파라미터 선택과 제거 결과를 단계별로 검증합니다.

#### `01_jump_threshold.py`

연속 GPS point 사이의 거리 분포와 knee를 분석하여 `jump_thres`를 추정합니다.

#### `02_jump_segmentation_visualization.py`

추정한 `jump_thres`로 trajectory를 분할하고 segment별 공간 관계를 시각화합니다.

#### `03_same_place_threshold.py`

segment의 `prev_next_distance` 분포를 분석해 `same_place_thres`를 추정합니다.

#### `04_jump_removal_visualization.py`

Sudden Position Jump 제거 전후의 trajectory를 비교하고 제거된 위치를 시각화합니다.

### `explorations/dbscan/`

ST-DBSCAN의 공간·시간 파라미터와 clustering 결과를 단계별로 검증합니다.

#### `01_spatial_k_dist_graph.py`

Spatial k-distance graph와 knee를 이용해 Spatial Eps 후보를 분석합니다.

#### `02_temporal_k_dist_graph.py`

Temporal k-distance graph와 knee를 이용해 Temporal Eps 후보를 분석합니다.

#### `03_st_dbscan.py`

선정한 파라미터로 ST-DBSCAN을 실행하고 이동·체류 clustering 결과를 시각화합니다.

### `explorations/map_matching/`

도로 네트워크 준비부터 Viterbi Map Matching까지의 과정을 단계별로 검증합니다.

#### `01_road_network.py`

GPS 범위에 맞는 도로 네트워크를 로드하고 시각적으로 확인합니다.

#### `02_projection.py`

GPS와 도로망을 동일한 평면 좌표계로 투영한 결과를 확인합니다.

#### `03_road_k_distance.py`

GPS point와 인접 도로 사이의 k-distance 분포를 분석해 후보 탐색 반경을 추정합니다.

#### `04_candidate_projection.py`

각 GPS point 주변의 도로 후보점을 생성하고 투영 결과를 시각화합니다.

#### `05_map_matching.py`

후보점에 emission/transition probability와 Viterbi 알고리즘을 적용합니다.  
원본 projected GPS와 최종 Map Matched 위치를 비교해 동작을 검증합니다.

### `pyproject.toml`

프로젝트 package 설정을 정의하며 `modules`를 editable install로 사용할 수 있도록 구성합니다.

### `requirements.txt`

프로젝트 실행에 필요한 Python dependency 목록을 관리합니다.

## Data Structure

Google Timeline에서 추출한 원본 데이터와 파이프라인에서 생성한 처리 결과를 분리하여 저장합니다.

```text
data/
├── timeline.json
├── raw_positions.csv
├── timeline_paths.csv
├── visits.csv
├── activities.csv
└── processed/
    ├── position_segments.csv
    ├── segments.csv
    ├── cleaned_positions.csv
    └── position_clusters.csv
```

`timeline.json`은 사용자가 준비하는 원본 Google Timeline 데이터입니다.

`raw_positions.csv`, `timeline_paths.csv`, `visits.csv`, `activities.csv`는 Timeline JSON에서 추출하고 표준화한 데이터이며 파이프라인 초기화 과정에서 다시 생성됩니다.

### `raw_positions.csv`

원본 위치 정보를 저장합니다.

```text
position_id
latitude
longitude
timestamp
```

각 position에 고유한 `position_id`를 부여하여 이후 처리 결과와 연결합니다.

### `position_segments.csv`

각 position과 segment 사이의 관계를 저장합니다.

```text
position_id
segment_id
```

위도, 경도, timestamp를 다시 저장하지 않고 `position_id`를 이용해 `raw_positions.csv`와 연결할 수 있도록 구성합니다.

### `segments.csv`

trajectory segmentation 과정에서 생성된 segment 단위 정보를 저장합니다.

```text
segment_id
mean_latitude
mean_longitude
head_position_id
tail_position_id
point_count
prev_next_distance
```

`head_position_id`와 `tail_position_id`를 이용하면 원본 위치 데이터에서 해당 segment의 시작 및 끝 위치와 timestamp를 조회할 수 있습니다.

### `cleaned_positions.csv`

Sudden Position Jump가 제거된 위치 데이터를 저장합니다.

후속 ST-DBSCAN과 Map Matching 등의 입력 데이터로 사용합니다.

### `position_clusters.csv`

ST-DBSCAN 결과를 position과 cluster의 관계로 저장합니다.

```text
position_id
cluster_id
```

원본 위치 정보를 반복해서 저장하지 않고 필요한 경우 `position_id`를 기준으로 위치 데이터와 결합합니다.

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

파이프라인을 실행하면 Google Timeline JSON에서 필요한 데이터를 추출하고 표준화된 CSV 데이터를 생성합니다.

이후 선택한 시간 범위의 위치 데이터에 대해 trajectory segmentation, Sudden Position Jump 제거, ST-DBSCAN clustering 등의 처리 과정을 순서대로 수행합니다.

### 6. Exploration 실행

각 알고리즘의 동작 및 분석 결과는 `explorations/`의 예제를 통해 확인할 수 있습니다.

예를 들어 Sudden Position Jump의 segmentation 결과를 확인하려면 다음과 같이 실행합니다.

```bash
python explorations/sudden_position_jump/02_jump_segmentation_visualization.py
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

이때 데이터가 이메일로 전송되거나 별도의 Google 서비스에 업로드되는 것이 아니라, **5번에서 직접 선택한 휴대전화의 저장 위치에 파일이 생성됩니다.**

따라서 내보내기가 완료된 후 Android의 **파일 앱**을 열어 저장할 때 선택했던 폴더로 이동하면 내보낸 Timeline JSON 파일을 확인할 수 있습니다.

### 프로젝트에 데이터 배치

내보낸 타임라인 JSON 파일을 PC로 옮긴 뒤 프로젝트의 `data` 디렉터리에 다음과 같이 배치합니다.

```text
data/
└── timeline.json
```

이후 `main.py`를 실행하면 Timeline JSON에서 필요한 데이터를 추출하여 파이프라인에서 사용할 CSV 데이터를 생성합니다.
