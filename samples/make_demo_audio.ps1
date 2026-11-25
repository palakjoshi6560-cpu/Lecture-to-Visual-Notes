# Creates a spoken demo recording (WAV) from a text file using the voice built into Windows, plus a copy of the text
# next to it so that tests/test_audio_real.py can score the speech recognition against what was spoken.
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File samples\make_demo_audio.ps1
#   powershell -ExecutionPolicy Bypass -File samples\make_demo_audio.ps1 -InFile evaluation\data\dev\biology_plain.txt -Name demo_biology
param([string]$InFile = "samples\sample_lecture.txt", [string]$Name = "demo_lecture")
Add-Type -AssemblyName System.Speech
$root = Split-Path -Parent $PSScriptRoot
$in = if ([System.IO.Path]::IsPathRooted($InFile)) { $InFile } else { Join-Path $root $InFile }
$outDir = Join-Path $root "samples\demo_audio"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$wav = Join-Path $outDir "$Name.wav"
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile($wav)
$synth.Speak((Get-Content -Raw -Encoding UTF8 $in))
$synth.Dispose()
Copy-Item $in (Join-Path $outDir "$Name.txt") -Force
Write-Host "Created $wav"
