## ⚙️ 설치 및 실행 방법 (Setup & Usage)

### 1️⃣ 사전 요구사항
- **Anaconda** 또는 **Miniconda** 설치 필수  
  👉 [공식 다운로드 페이지](https://www.anaconda.com/download)  
- **Windows PowerShell 5.1 이상** (기본 제공)  
- 프로젝트 루트에 `pyproject.toml` 파일이 있어야 함  
  (없을 경우, 스크립트가 자동으로 임시 README를 생성함)

---

### 2️⃣ 자동 설치 스크립트 실행
프로젝트 루트(즉, `pyproject.toml`이 있는 디렉터리)에서 아래 명령 실행:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-ecm-hanwha.ps1
```

#### 🧩 이 스크립트가 하는 일
1. **Conda 실행 파일 자동 탐색**  
   - PATH, `$CONDA_EXE`, 기본 설치 경로(`miniconda3`, `anaconda3`) 모두 자동 검색  
2. **Conda 가상환경 자동 생성**  
   - 환경 이름: `ecm-hanwha`  
   - Python 버전: `3.11`  
   - 이미 존재할 경우, 재사용  
3. **PyTorch 설치**  
   - 기본: CPU 전용  
   - GPU 사용 시, 스크립트 상단의 `$UseCUDA = $true` 로 변경 가능  
4. **pip / setuptools / wheel / build 업그레이드**  
5. **프로젝트 설치**  
   - 기본: `pip install -e .` (editable mode)  
6. **추가 의존성 설치**  
   - `requirements.txt` 자동 감지 후 설치  
7. **빌드**  
   - `sdist` 및 `wheel` 생성  
8. **설치 검증**  
   - `numpy`, `pandas`, `torch`, `sklearn` 모듈 임포트 테스트  

---

### 3️⃣ 가상환경 활성화 / 비활성화
설치 완료 후 아래 명령으로 환경을 활성화합니다:

```powershell
conda activate ecm-hanwha
```

작업을 마친 후 비활성화하려면:

```powershell
conda deactivate
```

---

### 4️⃣ 스크립트 옵션 (setup-ecm-hanwha.ps1 상단에서 수정 가능)

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `$EnvName` | 생성할 Conda 환경 이름 | `"ecm-hanwha"` |
| `$PythonVersion` | Python 버전 | `"3.11"` |
| `$UseCUDA` | CUDA GPU 버전(PyTorch GPU 빌드) 사용 여부 | `$false` |
| `$CudaMajor` | CUDA 버전 (예: `"12.4"`, `"11.8"`) | `"12.4"` |
| `$Editable` | 프로젝트를 editable 모드(`-e`)로 설치할지 여부 | `$true` |
| `$SkipBuild` | 빌드 단계 생략 여부 | `$false` |

---

### 5️⃣ 설치 확인
가상환경 활성화 후 아래 명령으로 설치 상태를 확인합니다:

```powershell
python -m pip list
```

또는

```powershell
python -c "import torch, pandas, numpy; print(torch.__version__)"
```

---

### ✅ 설치 완료 후 사용
- `.venv` 대신 Conda 환경(`ecm-hanwha`)이 사용됩니다.  
- 모든 Python 실행 명령은 해당 환경에서 수행해야 합니다:

```powershell
conda activate ecm-hanwha
python main.py
```

---

### 🧠 참고
- 스크립트는 **conda init** 여부와 상관없이  
  `conda.exe`를 직접 찾아 실행하므로, PATH 설정이 필요 없습니다.  
- GPU를 사용할 경우 `$UseCUDA = $true` 로 변경 후 다시 실행하세요.  
- 기존 환경을 재설치 없이 업데이트하려면,  
  `setup-ecm-hanwha.ps1`을 다시 실행하면 됩니다.
