# 설계와 구현

ECG → 손목 PPG 심방세동(AF) 전이 연구의 데이터 준비와 전처리 파이프라인 설계, 구현 세부 사항, 결정 과정을 기록한다. 환경 설정 방법은 [README.md](README.md)를 본다.

- **연구 질문**: 병원 ECG(MIT-BIH AFDB)로 학습한 AF 판별 능력을 손목 PPG로 옮길 수 있는가?
- **공통 표현**: 박동 간격 수열 (ECG의 RR, PPG의 맥파 간격 PPI)
- **역할**: A = 특징 (DFA, 엔트로피, 위상), B = 데이터와 전이. 이 문서는 현재 B 담당 부분(1주차)을 다룬다.

## 1. 전체 흐름

```
[AFDB]
download_afdb.py  →  data/mitbih_afdb/                 PhysioNet 원본 25개 기록
extract_rr.py     →  data/processed/afdb_rr.npz        RR, 리듬 라벨, 유효성 플래그
                     results/afdb_record_summary.csv   기록별 요약
build_afdb.py     →  data/processed/T_afdb.npz         60박동 구간 + 라벨
                     results/folds_afdb.json           환자 단위 5-fold
compare_detection.py → results/afdb_detection_compare.csv  자동 검출 vs 기준 박동

[Zenodo]
download_zenodo.py → data/zenodo_ppg_af/               선택한 환자의 ECG·PPG .mat
```

| 경로 | 내용 |
| --- | --- |
| `src/data/afdb.py` | 박동 로드, RR 계산, 리듬 라벨, 저장·요약 |
| `src/data/segment.py` | 구간 분할과 라벨 |
| `src/data/folds.py` | 환자 단위 fold |
| `src/data/detection.py` | neurokit2 검출과 기준 박동 비교 |
| `scripts/` | 위 모듈을 실행하는 스크립트 |
| `tests/` | 단위 테스트 (pytest) |
| `results/` | 커밋되는 작은 결과물 (표, fold, 그림) |
| `data/` | 원본·중간 데이터 (git 제외) |
| `docs/zenodo_format.md` | Zenodo 파일 형식 메모 |

## 2. MIT-BIH AFDB

### 2.1 다운로드 (`scripts/download_afdb.py`)

- PhysioNet의 `physionet.org/content/...` 엔드포인트가 502 오류를 내서 `wfdb.dl_database`가 실패했다 (2026-11-23).
- 대신 `physionet.org/files/...`가 리다이렉트하는 공식 S3 미러 `physionet-open.s3.amazonaws.com/afdb/1.0.0/`에서 받는다.
- 데이터베이스의 `SHA256SUMS.txt`를 파일 목록으로 쓰고, 파일마다 SHA256을 검증한다. 이미 받았고 체크섬이 맞는 파일은 건너뛴다.

**데이터 구성** (25개 기록, 각 약 10시간, 250 Hz, 2-lead)

| 파일 | 내용 |
| --- | --- |
| `.atr` | 리듬 주석: `(N`, `(AFIB`, `(AFL`, `(J` |
| `.qrs` | 자동 검출 박동 (사람이 검수하지 않음), 25개 기록 모두 |
| `.qrsc` | 수동 보정 박동, **05091, 07859 두 기록에만** 있음 |
| `.dat`/`.hea` | 신호, **00735, 03665는 신호 없음** (주석만 있음) |

### 2.2 RR 추출과 리듬 라벨 (`src/data/afdb.py`)

- **박동 출처**: `.qrsc`가 있으면 그것을, 없으면 `.qrs`를 쓴다. 출처는 `beat_source`로 기록한다.
- **범위**: 리듬 주석은 처음 10시간(9,000,000 샘플)만 덮으므로 그 뒤의 박동은 버린다. 신호가 더 짧으면(06453) 신호 끝까지만 쓴다.
- **RR**: `rr[i] = (beats[i+1] − beats[i]) / fs` (초).
- **리듬 라벨**: 각 RR 간격은 그 간격을 **끝내는 박동**(`beats[i+1]`) 시점의 리듬을 받는다. 정수 코드는 `UNKNOWN=-1, N=0, AFIB=1, AFL=2, J=3`이다. 정의되지 않은 리듬 문자열이 나오면 오류를 낸다.
- **유효성**: `0.25 s ≤ rr ≤ 2.5 s`이면 `valid`다. 여기서는 표시만 하고 구간 단계에서 처리한다.
- **저장**: `afdb_rr.npz`는 모든 기록을 이어 붙인 1차원 배열과 기록별 `offsets`로 저장한다.

**결과** (`results/afdb_record_summary.csv`): 25개 기록, 249시간, RR 1,221,650개

| 리듬 | 시간 | 비율 |
| --- | --- | --- |
| N | 149.1 h | 59.9% |
| AFIB | 93.3 h | 37.5% |
| J | 5.2 h | 2.1% (거의 03665) |
| AFL | 1.4 h | 0.6% |

- 범위 밖 RR: 859개 (0.07%), 04126·07879·08455에 몰려 있다.
- 기록별 AF 비율 차이가 크다: 07162·07859는 100%, 06426은 95%, 05091은 0.23%다.

### 2.3 구간 분할과 라벨 (`src/data/segment.py`)

| 규칙 | 값 | 이유 |
| --- | --- | --- |
| 구간 단위 | **60박동**(RR 60개), 고정 박동 수 | 심박수가 달라도 엔트로피·PH 입력 점 개수가 같다. 30박동은 SampEn·PH에 짧다 |
| 겹침 | **없음**, 기록 끝의 남는 부분은 버림 | 구간끼리 독립적이다 |
| 제외 | 범위 밖 RR, AFL, UNKNOWN 리듬이 **하나라도** 있는 구간 | 검출 오류와 애매한 리듬을 뺀다 |
| AF/non-AF | AFIB만 AF, N과 J는 non-AF | |
| `af_frac` | 구간 안 AFIB RR의 비율 | |
| `label_50` | `af_frac ≥ 0.5` → 1 | 기본 라벨 |
| `label_excl` | `≥ 0.9` → 1, `≤ 0.1` → 0, 그 사이 → −1 | 애매한 구간을 뺀 비교용 라벨 |

`T_afdb.npz`의 필드는 다음과 같다.

| 필드 | 형태 | 내용 |
| --- | --- | --- |
| `rr` | (N, 60) float32 | RR (초) |
| `af_frac` | (N,) float32 | AF 비율 |
| `label_50`, `label_excl` | (N,) int8 | 라벨 |
| `record` | (N,) str | 기록(환자) ID, fold의 group |
| `start_samp`, `end_samp` | (N,) int | 원신호에서 구간의 시작·끝 박동 샘플 위치 |
| `beat_source` | (N,) str | `qrsc` 또는 `qrs` |

### 2.4 기록 07162 제외

- `compare_detection.py`에서 07162만 두 리드 모두 Se 30%, PPV 32%였다.
- 시간이 한 방향으로 밀린 것이 아니라 기준 박동이 앞뒤로 평균 약 230 ms 흩어져 있었다.
- 신호에 겹쳐 그려 보면 neurokit2 검출은 실제 QRS와 맞고, 기준 `.qrs`가 어긋나 있다 (`results/figures/07162_beats.png`).
- 따라서 07162의 RR은 실제 박동 간격이 아니라고 보고 `build_afdb.py`의 `EXCLUDED_RECORDS`로 구간 데이터셋에서 뺐다. RR 요약표와 검출 비교표에는 근거로 남겨 둔다.

**최종 구간 데이터셋**: 24개 기록, 18,925개 구간, AF(`label_50`) 40.8%, 애매한 구간(`label_excl = −1`) 2.0%

### 2.5 환자 단위 fold (`src/data/folds.py`)

- AFDB는 기록 하나가 환자 한 명이다. 같은 환자가 학습과 시험에 섞이지 않게 기록 단위로 나눈다.
- 기록마다 AF 비율 차이가 커서 그냥 GroupKFold를 쓰면 AF가 거의 없는 fold가 생길 수 있다. 그래서 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)`를 `label_50`으로 층화해서 쓴다.
- 신호가 없는 00735, 03665도 RR만 쓰므로 포함한다.
- 결과 `results/folds_afdb.json`(`{record: fold}`)을 커밋해서 A와 B가 같은 fold를 쓴다. fold별 결과는 Wilcoxon 대응 검정에 쓴다.

| fold | 기록 | 구간 | AF % |
| --- | --- | --- | --- |
| 0 | 00735 05091 07859 07910 08215 | 3,572 | 46.2 |
| 1 | 04048 06426 06453 06995 08219 | 3,983 | 39.4 |
| 2 | 04043 04908 05121 08455 | 3,755 | 41.5 |
| 3 | 03665 04126 04746 05261 08405 | 3,924 | 36.5 |
| 4 | 04015 04936 07879 08378 08434 | 3,691 | 41.0 |

07162를 빼기 전의 fold와는 24개 중 18개 기록의 배정이 다르다.

### 2.6 자동 검출과 기준 박동 비교 (`src/data/detection.py`)

2주차 PPG 처리에서 쓸 "자동 검출 → 간격" 방식이 믿을 만한지 미리 확인한다.

| 항목 | 설정 |
| --- | --- |
| 검출기 | `nk.ecg_clean` + `nk.ecg_peaks` (neurokit2 기본값) |
| 리드 | ECG1, ECG2 각각 따로 |
| 기준 | RR 추출에 쓴 박동 (`.qrsc` 또는 `.qrs`), 같은 범위 |
| 매칭 | 150 ms 이내 (ANSI/AAMI EC57), `wfdb.processing.compare_annotations` |
| 리듬별 | AF = AFIB, non-AF = N·J. AFL 박동은 리듬별 지표에서만 뺀다 |
| RR 오차 | 연속한 두 기준 박동이 모두 매칭된 경우, 검출 RR과 기준 RR의 차이 (ms) |

**결과** (`results/afdb_detection_compare.csv`, 신호가 있는 23개 기록, 07162 포함, 기록별 중앙값)

| 리드 | 기준 | Se | PPV | Se(AF) | Se(non-AF) | RR 오차 중앙값 / 95% |
| --- | --- | --- | --- | --- | --- | --- |
| ECG1 | `.qrs` | 99.5% | 99.6% | 99.3% | 99.6% | 0 / 12 ms |
| ECG2 | `.qrs` | 99.4% | 99.6% | 99.2% | 99.7% | 0 / 28 ms |
| ECG1 | `.qrsc` | 95.1% | 98.9% | 95.5% | 99.2% | 12 / 58 ms |
| ECG2 | `.qrsc` | 99.3% | 99.5% | 99.0% | 99.9% | 2 / 36 ms |

- 대부분 기록에서 neurokit2 검출이 기준 박동과 거의 일치한다.
- 한 리드만 낮은 기록이 있다: 04043 ECG2(Se 72%), 08378 ECG1(Se 72%), 04908 ECG1의 AF 구간(Se 60%).
- 한계: `.qrs` 자체가 검수되지 않은 자동 검출 결과라서, 검출 정확도를 제대로 잴 수 있는 건 `.qrsc`가 있는 두 기록뿐이다. 나머지는 두 검출 방법의 일치도다.

## 3. Zenodo 손목 PPG 데이터

### 3.1 데이터셋 버전

- 레코드 **23921320** (2026-11-03 공개, concept DOI 10.5281/zenodo.5815073). 이전 계획에 적힌 11242869는 예전 버전이다.
- 이번 버전부터 공개(open) 접근이라 접근 요청이 필요 없다. 그래서 Plan B(CinC 2017)는 쓰지 않기로 했다 (2026-11-24).
- 환자 39명(PPG는 35명), 261일, 62.7 GB, MAT v7.3(HDF5), CC BY-NC-SA 4.0.
- 이번 버전에서 AF 주석을 다시 검토했고, PPG를 ECG에 다시 동기화했고, 환자 ID를 새로 붙였다.
- 파일 형식과 환자 006 확인 결과는 [docs/zenodo_format.md](docs/zenodo_format.md)에 정리했다.

### 3.2 다운로드 (`scripts/download_zenodo.py`)

| 동작 | 구현 |
| --- | --- |
| 파일 목록 | Zenodo API `api/records/{record}`에서 이름, 크기, MD5를 받는다. 레코드는 23921320으로 고정한다 (`--record`로 변경) |
| 환자 선택 | `--subjects 006 012`, `--subjects 001-010`, `--all`. 없는 번호면 받을 수 있는 번호 목록을 보여 주고 중단한다 |
| 받는 파일 | 환자마다 ECG와 PPG를 항상 함께 받는다 (AF 라벨이 ECG 파일에만 있음). `subject_info.xlsx`, `LICENSE.txt`는 항상 받는다 |
| PPG 없는 환자 | 024, 025, 034, 037은 ECG만 받고 경고를 출력한다 |
| 디스크 공간 | 남은 다운로드 용량이 여유 공간보다 크면 시작 전에 중단한다 |
| 건너뛰기 | 크기와 MD5가 맞는 파일은 건너뛴다 |
| 이어받기 | 덜 받은 파일은 HTTP `Range`로 이어받는다 (Zenodo는 206으로 응답). 서버가 200으로 응답하면 처음부터 다시 받는다 |
| 재시도 | 연결이 끊기면 최대 5번까지 간격을 늘려 가며 다시 시도한다. 006 ECG를 받을 때 실제로 두 번 끊겼다 |
| 검증 | 받은 뒤 MD5가 다르면 파일을 지우고 중단한다 |
| `--dry-run` | 받지 않고 파일 목록과 용량만 출력한다 |

외부 패키지 없이 표준 라이브러리만 쓴다.

### 3.3 2주차에 결정할 것

- **용량과 저장 위치**: 전체 62.7 GB, 로컬 여유 공간 약 16 GB → 쓸 환자 선택과 샘플링 규칙
- **`coarse` segment**: 시간 오차가 약 ±1초라 박동 단위 라벨에 쓸지
- **PPG 극성**: `PPG_GREEN` 값이 음수 → 맥파 peak 검출 전에 반전 여부 확인
- **RR 필터**: AFDB와 같은 0.25–2.5초 기준 적용 여부 (006에서 범위 밖 0.56%)
- **의존성**: 로더를 만들 때 `h5py`를 `requirements.txt`에 추가

## 4. 테스트

`python -m pytest -q tests`로 실행한다 (16개).

| 파일 | 확인 내용 |
| --- | --- |
| `test_afdb.py` | 리듬 주석 파싱, 정렬, 알 수 없는 라벨 거부, 리듬 경계 |
| `test_segment.py` | 겹침 없는 분할, 남는 부분 버림, `af_frac`과 두 라벨, AFL·범위 밖 RR 구간 제외 |
| `test_folds.py` | 모든 기록이 정확히 한 fold에 속함, 시드 고정 시 결과 동일 |
| `test_detection.py` | Se·PPV, 리듬별 지표, RR 오차, 150 ms 경계 |
| `test_download_zenodo.py` | 환자 번호·범위 해석, 파일 선택, PPG 없는 환자, 없는 번호 |

새로 clone한 환경에서 설치부터 `build_afdb.py`까지 실행해, `afdb_rr.npz`, `T_afdb.npz`, `results/` 파일이 로컬 결과와 완전히 같은지 확인했다 (2026-11-24).

## 5. 결정 기록

| 날짜 | 결정 | 근거 |
| --- | --- | --- |
| 11-23 | AFDB는 S3 미러에서 받는다 | PhysioNet `/content/` 502 오류 |
| 11-23 | 60박동, 겹침 없음 | 일정 문서 권장안 |
| 11-23 | AFL 포함 구간 제외 | 0.6%로 적고 AF/non-AF 어느 쪽에도 애매함 |
| 11-23 | 범위 밖 RR 포함 구간 제외 | 검출 오류가 특징을 왜곡, 영향은 0.07% |
| 11-23 | 5-fold StratifiedGroupKFold, 25개 기록 전체 | LOSO는 100% AF 기록 fold에서 AUROC 계산 불가 |
| 11-23 | 검출기 neurokit2, 두 리드 따로, 150 ms | 2주차 PPG 처리와 같은 라이브러리, AAMI 관례 |
| 11-23 | 07162 제외 | 기준 `.qrs`가 신호와 어긋남 |
| 11-24 | Zenodo 23921320 사용, CinC 2017 건너뜀 | 공개 접근으로 Plan B 불필요 |
| 11-24 | Zenodo 다운로드는 환자 선택형 | 전체 62.7 GB, 로컬 공간 부족 |
