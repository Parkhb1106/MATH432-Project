# MATH432-Project
POSTECH [MATH432] Mathematical Data Science Final Project

Title: 생체신호 및 음성 신호 분류를 위한 비선형 동역학 특징

Abstract: 생체신호와 음성 신호는 비선형 동역학계에서 나오기 때문에, 기존의 스펙트럼 특징만으로는 그 구조를 충분히 잡아내기 어렵다.
본 프로젝트는 이런 신호를 설명하는 상호보완적인 두 가지 방법, 즉 복잡도를 재는 엔트로피 기반 지표와 동역학 구조를 기술하는 위상적 특징을 다룬다. 각 신호 구간에서 sample entropy와 multiscale permutation entropy를 계산해 불규칙성과 예측 가능성을 정량화한다. 이어서 Takens 지연 임베딩으로 위상공간을 재구성하고, 그 점구름에 persistent homology를 적용한다. 얻은 persistence diagram은 persistence image와 landscape로 벡터화하여, 루프의 개수와 수명 같은 위상적 특징을 머신러닝 모델의 입력으로 쓸 수 있게 한다.
주 데이터는 PhysioNet의 MIT-BIH 부정맥 데이터베이스이며, 심박 구간을 정상과 부정맥으로 분류한다. (i) 엔트로피 특징, (ii) 위상 특징, (iii) 두 특징의 결합, (iv) 기준선인 MFCC·스펙트럼 특징으로 각각 학습한 분류기(SVM, 랜덤포레스트)의 성능을 비교한다. 또한 각 특징이 가산 노이즈에 얼마나 강건한지, 임베딩 파라미터(지연 τ, 차원 d)에 얼마나 민감한지도 분석한다. 확장 실험으로 같은 파이프라인을 감정 음성 데이터(예: RAVDESS)에 적용해, 결과가 다른 종류의 신호에도 성립하는지 확인한다.
우리는 엔트로피 특징과 위상 특징이 각각 동역학의 불규칙성과 반복되는 기하 구조라는 서로 다른 측면을 포착하며, 두 특징을 결합하면 간결하고 해석 가능하며 노이즈에 강건한 시계열 분류 표현을 얻을 수 있다고 가설을 세운다.

## 환경 설정

모든 명령은 **저장소 최상위 폴더**에서 실행한다. 스크립트가 `data/`, `results/` 상대 경로를 사용한다.

### 1. 설치

Python 3.13에서 확인했다.

```bash
git clone https://github.com/Parkhb1106/MATH432-Project.git
cd MATH432-Project
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

설치 확인:

```bash
python -m pytest -q tests
```

### 2. MIT-BIH AFDB 준비

데이터(약 0.6 GB)는 git에 포함되지 않으므로 직접 받는다.

```bash
python scripts/download_afdb.py
python scripts/extract_rr.py
python scripts/build_afdb.py
```

| 단계 | 결과물 |
| --- | --- |
| `download_afdb.py` | `data/mitbih_afdb/` (PhysioNet 원본, SHA256 검증) |
| `extract_rr.py` | `data/processed/afdb_rr.npz`, `results/afdb_record_summary.csv` |
| `build_afdb.py` | `data/processed/T_afdb.npz` (60박동 구간), `results/folds_afdb.json` (환자 단위 fold) |

`results/` 파일은 저장소에 커밋되어 있으며, 위 명령을 다시 실행해도 같은 내용이 만들어진다.

### 3. Zenodo 손목 PPG 데이터 준비

전체는 39명, 62.7 GB이므로 필요한 환자만 받을 수 있다. 각 환자의 ECG와 PPG 파일, `subject_info.xlsx`, `LICENSE.txt`를 받는다.

```bash
python scripts/download_zenodo.py --subjects 006 012 028   # 환자 번호 지정
python scripts/download_zenodo.py --subjects 001-010       # 범위 지정
python scripts/download_zenodo.py --all                    # 전체
python scripts/download_zenodo.py --subjects 001-010 --dry-run   # 받지 않고 파일 목록과 용량만 확인
```

- 기본 저장 위치는 `data/zenodo_ppg_af/`이며 `--out 경로`로 바꿀 수 있다.
- 이미 받은 파일은 건너뛰고, 중간에 끊긴 파일은 이어받는다. 디스크 공간이 부족하면 시작 전에 중단한다.
- 데이터는 CC BY-NC-SA 4.0 라이선스이므로 git에 올리지 않는다 (`/data/`는 `.gitignore`에 포함).

설계와 구현 세부 사항은 [DESIGN.md](DESIGN.md), Zenodo 파일 형식은 [docs/zenodo_format.md](docs/zenodo_format.md)를 참고한다.
