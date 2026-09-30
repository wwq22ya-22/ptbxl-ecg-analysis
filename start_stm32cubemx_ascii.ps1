param(
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'

$cubeMxExe = 'D:\cube\STM32CubeMX.exe'
$javawExe = 'D:\cube\jre\bin\javaw.exe'
$asciiRoot = 'D:\STM32AI'
$asciiProfile = Join-Path $asciiRoot 'User'
$asciiTemp = Join-Path $asciiRoot 'Temp'
$asciiCubeRoot = Join-Path $asciiProfile 'STM32Cube'
$asciiRepository = Join-Path $asciiCubeRoot 'Repository'
$originalProfile = [Environment]::GetFolderPath('UserProfile')
$sourceRepository = Join-Path $originalProfile 'STM32Cube\Repository'

function Assert-DirectoryLink {
    param(
        [Parameter(Mandatory = $true)][string]$LinkPath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    if (-not (Test-Path -LiteralPath $TargetPath -PathType Container)) {
        throw "Original STM32Cube repository was not found: $TargetPath"
    }
    if (Test-Path -LiteralPath $LinkPath) {
        $item = Get-Item -LiteralPath $LinkPath -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            throw "Path exists but is not a directory junction: $LinkPath"
        }
        return
    }
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
}

foreach ($requiredFile in @($cubeMxExe, $javawExe)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required STM32CubeMX file was not found: $requiredFile"
    }
}

New-Item -ItemType Directory -Force -Path $asciiRoot, $asciiProfile, $asciiTemp, $asciiCubeRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $asciiProfile 'AppData\Roaming'), (Join-Path $asciiProfile 'AppData\Local') | Out-Null
Assert-DirectoryLink -LinkPath $asciiRepository -TargetPath $sourceRepository

# These variables apply only to CubeMX launched by this script.
$env:USERPROFILE = $asciiProfile
$env:HOMEDRIVE = 'D:'
$env:HOMEPATH = '\STM32AI\User'
$env:APPDATA = Join-Path $asciiProfile 'AppData\Roaming'
$env:LOCALAPPDATA = Join-Path $asciiProfile 'AppData\Local'
$env:TEMP = $asciiTemp
$env:TMP = $asciiTemp

$stEdgeAi = Join-Path $asciiRepository 'Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe'

if ($CheckOnly) {
    Write-Host "CubeMX: $cubeMxExe"
    Write-Host "Java: $javawExe"
    Write-Host "Isolated profile: $asciiProfile"
    Write-Host "Isolated temp: $asciiTemp"
    Write-Host "ASCII repository: $asciiRepository"
    Write-Host "X-CUBE-AI: $stEdgeAi"
    if (-not (Test-Path -LiteralPath $stEdgeAi -PathType Leaf)) {
        throw "X-CUBE-AI 10.2.1 was not found through the ASCII repository path."
    }
    & $stEdgeAi --version
    if ($LASTEXITCODE -ne 0) {
        throw "stedgeai self-check failed with exit code $LASTEXITCODE"
    }
    Write-Host 'The isolated ASCII environment passed its self-check.'
    exit 0
}

Write-Host 'Starting STM32CubeMX with the isolated ASCII environment...'
$cubeMxArgs = @(
    '-Duser.home=D:\STM32AI\User',
    '-Djavax.net.ssl.trustStoreType=WINDOWS-ROOT',
    '-Dsun.java2d.d3d=false',
    '--add-exports',
    'java.desktop/sun.awt=ALL-UNNAMED',
    '--add-opens',
    'java.desktop/java.awt=ALL-UNNAMED',
    '-Dfile.encoding=UTF8',
    '-classpath',
    'D:\cube\STM32CubeMX.exe;anything',
    'com.st.microxplorer.maingui.STM32CubeMX'
)
Start-Process -FilePath $javawExe -ArgumentList $cubeMxArgs -WorkingDirectory (Split-Path -Parent $cubeMxExe)

