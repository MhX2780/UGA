# 1. Define Python version and paths
$version = "3.13.1"
$url = "https://python.org"
$installerPath = "$env:TEMP\python-installer.exe"

# 2. Download the installer
Invoke-WebRequest -Uri $url -OutFile $installerPath

# 3. Run silent installation and automatically add Python to PATH
Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait

# 4. Clean up installer file
Remove-Item $installerPath
