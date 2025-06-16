# Image Deblurring AI - Setup Script
# This script sets up the environment and dependencies for the deblurring application

Write-Host "🖼️ Image Deblurring AI - Setup Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersionNumber = (python -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>$null)
if ($pythonVersionNumber) {
    $major, $minor = $pythonVersionNumber -split ' '
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 8)) {
        Write-Host "❌ Python 3.8 or higher is required. Current version: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}

# Create virtual environment
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment created" -ForegroundColor Green

# Activate virtual environment
Write-Host "`n🔧 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`n⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`n📥 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green

# Create necessary directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Yellow
$directories = @("uploads", "outputs", "data\samples")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    }
}

# Check GPU availability
Write-Host "`n🖥️ Checking GPU availability..." -ForegroundColor Yellow
$gpuCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>$null
if ($gpuCheck) {
    Write-Host $gpuCheck -ForegroundColor Cyan
} else {
    Write-Host "GPU check failed, but CPU processing will work fine" -ForegroundColor Yellow
}

# Test model loading
Write-Host "`n🧠 Testing model initialization..." -ForegroundColor Yellow
$modelTest = python -c "
import sys
sys.path.append('.')
try:
    from model.deblurgan import DeblurGANv2
    model = DeblurGANv2()
    print('✅ Model loaded successfully')
except Exception as e:
    print(f'⚠️ Model test: {e}')
" 2>$null

if ($modelTest) {
    Write-Host $modelTest -ForegroundColor Cyan
}

# Create launch script
Write-Host "`n🚀 Creating launch script..." -ForegroundColor Yellow
$launchScript = @"
# Launch Image Deblurring AI
Write-Host "🖼️ Starting Image Deblurring AI..." -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Launch Streamlit app
Write-Host "Opening web browser..." -ForegroundColor Green
Write-Host "If the browser doesn't open automatically, go to: http://localhost:8501" -ForegroundColor Yellow

streamlit run app\streamlit_app.py
"@

$launchScript | Out-File -FilePath "launch.ps1" -Encoding UTF8

Write-Host "`n🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Run: .\launch.ps1" -ForegroundColor White
Write-Host "   OR" -ForegroundColor Yellow
Write-Host "2. Run manually:" -ForegroundColor White
Write-Host "   venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "   streamlit run app\streamlit_app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "The app will open in your web browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Yellow
