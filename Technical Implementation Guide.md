# Mine Detection — Technical Implementation Guide

---

## 1. Project Overview

이 프로젝트는 해저 이미지 데이터로부터 **기뢰(Mine)** 와 **비기뢰(Terrain)** 를 구분하기 위한 **통합 자동화 파이프라인**입니다.  
입력은 이미지와 XML 어노테이션이며, 출력은 학습된 분류 모델 및 성능 리포트입니다.

전체 처리 과정은 아래와 같습니다:

```
Raw Images + XML
      │
      ▼
Terrain XML Generation ─▶ Cropping ─▶ Label CSV Generation
      │
      ▼
Dataset Splitting ─▶ Data Augmentation (Mine only)
      │
      ▼
Feature Extraction (HOG, LBP, Gabor, SFS)
      │
      ▼
Model Training & Evaluation (SVC, RandomForest)
      │
      ▼
Report Generation (per-model & summary CSV)
```

---

## 2. Environment and Dependencies

### 2.1. Python Environment

- **Python Version:** ≥ 3.10  
- **Supported OS:** Windows 10/11, Ubuntu 20.04+  
- **Hardware:** CPU or GPU (CUDA 지원 시 Torch 가속 가능)

### 2.2. Core Dependencies

의존성은 `requirements.txt` 및 `setup.py`, `pyproject.toml`을 기반으로 관리됩니다.  
다음 명령으로 환경을 구성할 수 있습니다:

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# 패키지 설치
pip install -r requirements.txt
```

**핵심 의존성 목록 (요약)**

| Category | Packages |
|-----------|-----------|
| **Data Handling** | numpy==1.26.4, pandas==2.3.2, scipy==1.13.1 |
| **Image Processing** | opencv-python==4.11.0.86, scikit-image==0.24.0, imageio==2.37.0, pillow==11.3.0 |
| **Machine Learning** | scikit-learn==1.6.1, torch==2.2.2, torchvision==0.17.2, torchaudio==2.2.2 |
| **Visualization** | matplotlib==3.9.4, seaborn==0.13.2 |
| **Utilities** | tqdm==4.67.1, psutil==7.0.0, joblib==1.5.2, pyyaml==6.0.2 |
| **Geospatial / Domain Tools** | pyproj==3.6.1, pyxtf==1.4.2 |
| **Misc.** | colorama, jinja2, markupsafe, fsspec, fonttools 등 |

> 모든 버전은 `requirements.txt`와 동기화되어 있으며, `pyproject.toml`은 `setuptools` 기반으로 패키징됩니다.

### 2.3. Package Metadata (from setup.py)

- **Package Name:** `ecm-hanwha`  
- **Version:** 0.1.0  
- **Package Root:** `src/`  
- **Python Requirement:** ≥ 3.6  
- **Build Backend:** `setuptools.build_meta`

---

## 3. Directory Structure

```
project/
├─ src/
│  ├─ process/
│  │  ├─ terrain.py
│  │  ├─ cropping.py
│  │  ├─ labeling.py
│  │  ├─ splitting.py
│  │  └─ augmentation.py
│  ├─ features/
│  │  ├─ combined_feature_extractor.py
│  │  ├─ hog_extractor.py
│  │  ├─ lbp_extractor.py
│  │  └─ gabor_extractor.py
│  └─ pipeline/
│     ├─ data_loader.py
│     ├─ models.py
│     └─ reporting.py
│
├─ scripts/
│  └─ end_to_end_pipeline.py
├─ config/
│  └─ tuned_params.json
├─ datasets/
│  ├─ raw/original/
│  ├─ xml_with_terrain/
│  ├─ crops/original/
│  └─ augmented/original/
└─ results/
   ├─ labels/
   ├─ splits/
   ├─ aug_labels/
   ├─ features/
   ├─ models/
   └─ reports/
```

---

## 4. Pipeline Workflow

| Step | Module | Input | Output | Description |
|------|---------|--------|---------|-------------|
| 1 | terrain.py | 원본 XML, 이미지 | Terrain XML | 비기뢰 패치 자동 샘플링 |
| 2 | cropping.py | XML, 이미지 | 패치 이미지 | 클래스별 크롭 생성 |
| 3 | labeling.py | 크롭 이미지 | CSV | 라벨링 정보 생성 |
| 4 | splitting.py | CSV | train/test CSV | 기뢰: 비율 / Terrain: 정량 |
| 5 | augmentation.py | train CSV | 증강 이미지, CSV | 기뢰만 증강 |
| 6 | combined_feature_extractor.py | 이미지 | .npz 파일 | HOG, LBP, Gabor, SFS 특징 추출 |
| 7 | models.py, reporting.py | .npz | 모델, 리포트 | 학습/평가 및 성능 보고 |

---

## 5. Execution Guide

### Standard Execution

```bash
python end_to_end_pipeline.py
```

- **기본 세트명:** `original`
- **출력 경로:**
  - `results/features/original/{train,test}`
  - `results/models/original`
  - `results/reports/original`

---

## 6. Input / Output Specification

### Input
| Type | Path | Format |
|------|------|---------|
| 원본 이미지 | datasets/raw/original/*.jpg | RGB or grayscale |
| XML Annotation | datasets/raw/original/*.xml | Pascal VOC 형태 |
| (선택) 튜닝 파라미터 | config/tuned_params.json | JSON |

### Output
| Type | Path | Description |
|------|------|-------------|
| Terrain XML | datasets/xml_with_terrain/original | 비기뢰 패치 정보 포함 |
| 크롭 이미지 | datasets/crops/original/{mine,terrain} | 클래스별 크롭 패치 |
| 라벨 CSV | results/labels/original/*.csv | 이미지-클래스 매핑 |
| 분할 CSV | results/splits/original/*.csv | train/test 구분 |
| 증강 이미지 | datasets/augmented/original/train/mine | 증강된 기뢰 이미지 |
| 특징 벡터 | results/features/original/{train,test}/*.npz | HOG/LBP/Gabor/SFS 특징 |
| 모델 파일 | results/models/original/{feature}/*.pkl | 학습된 모델 |
| 리포트 | results/reports/original/{feature}/*_report.csv | 정밀도/재현율/F1 |
| 요약 CSV | results/reports/original/summary_csv/ | 전체 성능 집계 |

---

## 7. Model Training and Evaluation

- **SVC (Support Vector Classifier)**  
  - 커널: RBF / Linear  
  - 주요 하이퍼파라미터: C, gamma  
- **Random Forest**  
  - 주요 파라미터: n_estimators, max_depth, min_samples_split  

### 성능 리포트 항목
| Metric | Description |
|---------|--------------|
| Precision | 기뢰로 예측된 중 실제 기뢰 비율 |
| Recall | 실제 기뢰 중 기뢰로 검출된 비율 |
| F1-score | Precision과 Recall의 조화 평균 |
| Accuracy | 전체 정확도 |

---

## 8. Feature Extraction Overview

실제 프로젝트에는 SfS(Shape from Shading)을 아래 세가지 기법과 함께 사용했지만 성능이 비교적 낮아 제외하여 제공합니다.

| Feature | Description |
|----------|--------------|
| HOG | 방향 그래디언트 기반 형태 특징 |
| LBP | 픽셀 주변 밝기 패턴 기반 질감 특징 |
| Gabor | 주파수/방향 필터 기반 질감 분석 |

---

## 9. Development Notes

- 모든 경로 상수는 `end_to_end_pipeline.py` 상단 설정 블록에서 일괄 관리됩니다.  
- 데이터 교환 형식은 `CSV → NPZ → Model PKL` 순으로 고정되어 있습니다.  
- 병렬 처리(`max_workers`)와 시드(`SEED=2025`)는 상단 변수로 조정 가능합니다.  
- 모든 로그 메시지는 English로 통일되어 있습니다.  

### 확장 포인트
- 새로운 Feature Extractor → `src/features/`에 추가  
- 새로운 Classifier → `src/pipeline/models.py`의 `build_pipeline_for_feature()` 확장  
- 새로운 데이터셋 → `datasets/raw/{new_set}` 추가 후 `SET` 변경  

---

## 10. Version Control and Build

```bash
python -m build
pip install dist/ecm_hanwha-0.1.0-py3-none-any.whl
```

### 주요 설정
- **Package Name:** ecm-hanwha  
- **Build System:** setuptools (>=61.0)  
- **Python Version:** >=3.10  
- **License:** MIT  
- **Root:** src/  

---

## 11. Summary

| 항목 | 내용 |
|------|------|
| **프로젝트 목적** | 기뢰 탐지 자동화 파이프라인 구축 |
| **입력** | 해저 이미지 + XML |
| **출력** | 학습 모델, 특징 벡터, 리포트 |
| **언어/환경** | Python 3.10+, OpenCV, PyTorch, scikit-learn |
| **실행 명령** | python end_to_end_pipeline.py |
| **결과 디렉터리** | results/ 및 datasets/ 하위 구조 |
| **유지 관리자** | 개발팀 리드 또는 ML 담당자 |
