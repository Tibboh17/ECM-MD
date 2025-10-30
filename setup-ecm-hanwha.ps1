#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ====================== CONFIG ======================
$EnvName       = "ecm-hanwha"  # conda env name
$PythonVersion = "3.11"        # 3.10 또는 3.11 권장
$UseCUDA       = $false        # GPU 사용 시 $true
$CudaMajor     = "12.4"        # 예: "12.4" 또는 "11.8" (UseCUDA=true일 때만 사용)
$Editable      = $true         # true: pip install -e .
$SkipBuild     = $false        # true: build 단계 생략
# ====================================================

function Fail($msg) { Write-Error $msg; exit 1 }

# --- Locate conda.exe robustly ---
function Get-CondaExe {
  # 1) PATH
  $cmd = Get-Command conda -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Path }

  # 2) ENV
  if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }

  # 3) Common install paths
  $candidates = @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe"
  )
  foreach ($p in $candidates) {
    if (Test-Path $p) { return $p }
  }
  return $null
}

# --- Run a command inside the conda env (no global activate needed) ---
function CondaRun([string[]]$Args, [string]$CondaExe, [string]$EnvName) {
  # 실행 및 출력 캡처
  $allArgs = @('run','-n',$EnvName) + $Args
  $out = & $CondaExe @allArgs 2>&1
  $code = $LASTEXITCODE
  if ($out) { $out | Write-Host }
  if ($code -ne 0) {
    Fail ("Command failed: conda {0}" -f ($allArgs -join ' '))
  }
}

Write-Host "==> Checking project root..." -ForegroundColor Cyan
if (-not (Test-Path -Path "pyproject.toml")) { Fail "pyproject.toml not found. Run this script at project root." }
if (-not (Test-Path -Path "README.md")) {
  Write-Host "==> README.md missing, creating a placeholder..." -ForegroundColor Yellow
  @"
# ECM-Hanwha

Temporary README created by setup script.
"@ | Out-File -Encoding UTF8 README.md
}

Write-Host "==> Locating Conda..." -ForegroundColor Cyan
$CondaExe = Get-CondaExe
if (-not $CondaExe) {
  Fail "Conda not found. Open 'Anaconda Prompt (PowerShell)' OR run 'conda init powershell' and reopen the shell."
}
Write-Host ("Using conda at: {0}" -f $CondaExe)

Write-Host "==> Creating conda env '$EnvName' (python=$PythonVersion)..." -ForegroundColor Cyan
$envList = (& $CondaExe env list) -join "`n"
if ($LASTEXITCODE -ne 0) { Fail "Failed to list conda envs." }
if ($envList -notmatch ("^\s*{0}\s" -f [regex]::Escape($EnvName))) {
  & $CondaExe create -y -n $EnvName ("python={0}" -f $PythonVersion)
  if ($LASTEXITCODE -ne 0) { Fail "Failed to create conda env '$EnvName'." }
} else {
  Write-Host "Environment already exists. Skipping create."
}

Write-Host "==> Installing PyTorch (conda)..." -ForegroundColor Cyan
if ($UseCUDA) {
  & $CondaExe install -y -n $EnvName pytorch torchvision torchaudio ("pytorch-cuda={0}" -f $CudaMajor) -c pytorch -c nvidia
  if ($LASTEXITCODE -ne 0) { Fail "Failed to install PyTorch with CUDA via conda." }
} else {
  & $CondaExe install -y -n $EnvName pytorch torchvision torchaudio cpuonly -c pytorch
  if ($LASTEXITCODE -ne 0) { Fail "Failed to install CPU PyTorch via conda." }
}

Write-Host "==> Upgrading pip/setuptools/wheel/build..." -ForegroundColor Cyan
CondaRun @("python","-m","pip","install","--upgrade","pip") $CondaExe $EnvName
CondaRun @("python","-m","pip","install","setuptools>=77","wheel","build") $CondaExe $EnvName

Write-Host "==> Installing project (pyproject.toml)..." -ForegroundColor Cyan
if ($Editable) {
  CondaRun @("python","-m","pip","install","-e",".") $CondaExe $EnvName
} else {
  CondaRun @("python","-m","pip","install",".") $CondaExe $EnvName
}

if (Test-Path "requirements.txt") {
  Write-Host "==> Installing requirements.txt ..." -ForegroundColor Cyan
  CondaRun @("python","-m","pip","install","-r","requirements.txt") $CondaExe $EnvName
}

if (-not $SkipBuild) {
  Write-Host "==> Building sdist & wheel..." -ForegroundColor Cyan
  CondaRun @("python","-m","build") $CondaExe $EnvName
}

Write-Host "==> Verifying key imports..." -ForegroundColor Cyan
$verify = @'
import importlib, sys
mods = ["numpy","pandas","torch","sklearn"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "unknown")
        print(f"[OK] {m} {v}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {m}: {e}", file=sys.stderr)
sys.exit(0 if ok else 1)
'@
$tempPy = Join-Path $PWD "._verify_imports_tmp.py"
$verify | Out-File -Encoding UTF8 $tempPy
try {
  CondaRun @("python",$tempPy) $CondaExe $EnvName
} finally {
  Remove-Item -Force $tempPy -ErrorAction SilentlyContinue | Out-Null
}

Write-Host "==> Done." -ForegroundColor Green
Write-Host "Activate with:"
Write-Host ("    conda activate {0}" -f $EnvName)
Write-Host "To deactivate:"
Write-Host "    conda deactivate"
