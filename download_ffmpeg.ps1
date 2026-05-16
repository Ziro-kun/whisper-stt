param([string]$TargetDir)

$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zip = Join-Path $TargetDir "ffmpeg.zip"
$tmp = Join-Path $TargetDir "tmp"

try {
    Write-Host "Downloading ffmpeg..."
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

    Write-Host "Extracting..."
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    $exe = Get-ChildItem $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if (-not $exe) { throw "ffmpeg.exe not found in archive" }

    Copy-Item $exe.FullName (Join-Path $TargetDir "ffmpeg.exe")
    Remove-Item $tmp -Recurse -Force
    Remove-Item $zip -Force

    Write-Host "ffmpeg installed."
} catch {
    Write-Host "Error: $_"
    exit 1
}
