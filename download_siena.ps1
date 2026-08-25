# Download Siena Scalp EEG Database from PhysioNet
# This script downloads all EDF files for seizure detection

$baseUrl = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
$outputDir = "additional_data\siena"

# Create output directory
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-Host "=============================================="
Write-Host "Downloading Siena Scalp EEG Database"
Write-Host "=============================================="
Write-Host ""

# Patient folders and their EDF files
$patients = @{
    "PN00" = @("PN00-1.edf", "PN00-2.edf", "PN00-3.edf", "PN00-4.edf", "PN00-5.edf")
    "PN01" = @("PN01-1.edf", "PN01-2.edf", "PN01-3.edf", "PN01-4.edf")
    "PN03" = @("PN03-1.edf", "PN03-2.edf", "PN03-3.edf", "PN03-4.edf")
    "PN05" = @("PN05-1.edf", "PN05-2.edf")
    "PN06" = @("PN06-1.edf", "PN06-2.edf", "PN06-3.edf", "PN06-4.edf")
    "PN07" = @("PN07-1.edf", "PN07-2.edf", "PN07-3.edf")
    "PN09" = @("PN09-1.edf", "PN09-2.edf")
    "PN10" = @("PN10-1.edf", "PN10-2.edf", "PN10-3.edf")
    "PN11" = @("PN11-1.edf", "PN11-2.edf", "PN11-3.edf")
    "PN12" = @("PN12-1.edf", "PN12-2.edf", "PN12-3.edf", "PN12-4.edf")
    "PN13" = @("PN13-1.edf", "PN13-2.edf", "PN13-3.edf")
    "PN14" = @("PN14-1.edf", "PN14-2.edf", "PN14-3.edf", "PN14-4.edf")
}

$totalFiles = 0
$downloadedFiles = 0

foreach ($patient in $patients.Keys) {
    $patientDir = Join-Path $outputDir $patient
    New-Item -ItemType Directory -Force -Path $patientDir | Out-Null
    
    foreach ($file in $patients[$patient]) {
        $totalFiles++
        $url = "$baseUrl/$patient/$file"
        $localPath = Join-Path $patientDir $file
        
        if (Test-Path $localPath) {
            Write-Host "  Already exists: $patient/$file"
            $downloadedFiles++
            continue
        }
        
        Write-Host "  Downloading: $patient/$file..." -NoNewline
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $localPath -TimeoutSec 300
            Write-Host " OK"
            $downloadedFiles++
        }
        catch {
            Write-Host " FAILED"
        }
    }
}

# Also download seizure annotations
Write-Host ""
Write-Host "Downloading seizure annotations..."

foreach ($patient in $patients.Keys) {
    $patientDir = Join-Path $outputDir $patient
    $annotFile = "$patient-Seizures-list-grouped.txt"
    $url = "$baseUrl/$patient/$annotFile"
    $localPath = Join-Path $patientDir $annotFile
    
    if (-not (Test-Path $localPath)) {
        try {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $localPath -TimeoutSec 60 -ErrorAction SilentlyContinue
            Write-Host "  Downloaded: $annotFile"
        }
        catch {
            # Annotation file might not exist for all patients
        }
    }
}

Write-Host ""
Write-Host "=============================================="
Write-Host "Download complete: $downloadedFiles / $totalFiles files"
Write-Host "Output directory: $outputDir"
Write-Host "=============================================="
