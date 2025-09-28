Set-Location ml-service
if (!(Test-Path venv)) {
    python -m venv venv
}
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
