# Zenodo 23921320 파일 형식 메모

- **데이터셋**: Long-term electrocardiogram and wrist-based photoplethysmogram recordings with annotated atrial fibrillation episodes
- **레코드**: https://zenodo.org/records/23921320 (DOI 10.5281/zenodo.23921320, concept DOI 10.5281/zenodo.5815073)
- **공개일**: 2026-11-03, 공개(open) 접근, 접근 요청 필요 없음
- **라이선스**: CC BY-NC-SA 4.0 → 데이터는 git에 올리지 않는다 (`/data/`는 `.gitignore`에 포함)
- **확인일**: 2026-11-24, 환자 006 한 쌍과 `subject_info.xlsx`를 받아 확인 (MD5 일치)

## 데이터셋 개요

| 항목 | 값 |
| --- | --- |
| 환자 | 39명, 5–8일 모니터링, 총 261일 |
| PPG가 있는 환자 | 35명 (024, 025, 034, 037은 ECG만 있음) |
| 파일 | 환자마다 `XXX_ECG.mat`, `XXX_PPG.mat`(있는 경우), 그리고 `subject_info.xlsx`, `LICENSE.txt` |
| 전체 용량 | 62.7 GB (PPG 34.7 GB, ECG 28.0 GB), 파일당 0.05–1.7 GB |

이번 버전에서는 AF 주석을 다시 검토했고, PPG를 ECG 시계에 다시 동기화했고, **환자 ID를 새로 붙였다** (이전 버전의 ID와 다를 수 있다).

## 공통 형식

- MATLAB v7.3(HDF5) 형식이다. `scipy.io.loadmat`으로는 읽을 수 없고 `h5py`로 읽는다.
- 문자열은 char(uint16) 배열이고, 셀 배열은 HDF5 object reference 배열이다.
- `signalHeader` 그룹에 `signal_labels`, `physical_dimension`, `physical_min`, `physical_max`, `samples_in_record`(샘플링 레이트)가 들어 있다.

## `XXX_ECG.mat` (Bittium Faros 180)

| 변수 | 형태 | 내용 |
| --- | --- | --- |
| `ECG` | (N, 1) double | 단일 채널, 500 Hz, µV |
| `Accelerometer_X/Y/Z` | (N/20, 1) double | 흉부 가속도, 25 Hz, mg |
| `QRSindex` | (n_beats, 1) double | R-peak 샘플 위치 (첫 값 131이므로 MATLAB 1-based로 추정) |
| `rr` | (n_beats − 1, 1) double | RR(초), `diff(QRSindex) / 500`과 정확히 같음 |
| `AF_annotation` | (n_beats − 1, 1) double | RR 간격마다 0/1 (1 = AF), `rr`과 길이가 같음 |
| `recording_startday` | char | 첫 모니터링일 기준 날짜 번호 (예: `"01"`) |
| `recording_starttime` | char | ECG 시계 기준 시작 시각 (예: `"13:54:25"`) |

- 기록이 두 번에 나눠 찍힌 경우 한 파일로 합쳐져 있고, 그 사이는 NaN으로 채워져 있다 (데이터셋 설명 기준, 006에는 NaN 없음).
- 잡음이나 전극 접촉 불량 구간에서는 R-peak이 검출되지 않아 긴 RR이 생긴다.

## `XXX_PPG.mat` (손목 기기)

모든 변수가 **PPG segment 하나당 셀 하나**인 셀 배열이고, 셀 순서는 변수끼리 같다.

| 변수 | 셀 내용 |
| --- | --- |
| `PPG_GREEN` | 녹색 LED PPG, 100 Hz, ADC 단위 |
| `PPG_AMBIENT` | 주변광 채널, 100 Hz, ADC 단위 |
| `Accelerometer_X/Y/Z` | 손목 가속도, 50 Hz, g |
| `recording_startday`, `recording_starttime` | segment 시작 날짜와 시각, **ECG 시계 기준** (예: `"02"`, `"08:14:50.610"`) |
| `sync_method` | `beat` (박동 단위로 확인, 약 ±0.05초) 또는 `coarse` (움직임으로만 확인, 약 ±1초) |

- 가속도 단위가 ECG 파일(mg)과 PPG 파일(g)에서 다르다.
- 데이터셋 설명에 따르면 R-peak에서 PPG까지의 지연은 동기화할 때 0.3초로 가정한 값이라 독립적인 측정이 아니다.

## PPG 구간에 AF 라벨 붙이기

1. ECG와 PPG segment의 시작 시점을 초로 바꾼다: `(startday − 1) × 86400 + starttime`.
2. PPG segment의 ECG 기준 오프셋 = PPG 시작 − ECG 시작.
3. ECG 박동 시각(`QRSindex / 500`) 가운데 `[오프셋, 오프셋 + segment 길이)`에 드는 RR을 골라 `AF_annotation`을 가져온다.

## 환자 006 확인 결과

- ECG: 183.3시간, 박동 864,623개, AF 25.3%(시간 기준), AF 에피소드 2개
- `rr`이 0.25–2.5초 범위를 벗어나는 경우: 4,825개 (0.56%), 최소 0.002초, 최대 245초
- PPG segment 3개, 총 6.1시간 (ECG에 비해 매우 짧음)

| segment | ECG 시작 기준 | 길이 | sync | 겹치는 박동 | AF |
| --- | --- | --- | --- | --- | --- |
| 0 | +18.34 h | 2.39 h | beat | 9,974 | 0% |
| 1 | +22.08 h | 0.80 h | coarse | 3,274 | 0% |
| 2 | +45.37 h | 2.95 h | beat | 17,495 | 100% |

## `subject_info.xlsx`

40행 × 32열이다. 환자 ID, 성별, 나이, 키, 몸무게, 음주·흡연, 신체 활동, 동반 질환(고혈압 단계 포함), 복용 약물이 들어 있다. 일부 열 이름이 비어 있다(`Unnamed: 23`–`27`).

## 2주차에 결정해야 할 것

- **용량**: 전체 62.7 GB인데 로컬 여유 공간은 약 16 GB다 → 사용할 환자 선택과 저장 위치를 정해야 한다.
- **`coarse` segment**: 시간 오차가 약 ±1초라, 박동 단위 라벨에 쓸지 `beat` segment만 쓸지 정해야 한다.
- **PPG 극성**: `PPG_GREEN` 값이 음수다 → 맥파 peak 검출 전에 신호가 뒤집혀 있는지 확인한다.
- **RR 필터**: AFDB와 같은 0.25–2.5초 기준을 적용할지 정한다.
- **의존성**: Zenodo 로더를 만들 때 `h5py`를 `requirements.txt`에 추가한다.
