# Hash Integrity Verification Script for Edendale GXP SignalsPack
# Verifies that CSV file hashes match the embedded hashes in report JSONs
# Requires signals_manifest.toml to exist and checks for all expected epochs/files

param(
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "outputs_latest"
)

if (-not (Test-Path $OutputDir)) {
    Write-Host "Error: Output directory not found: $OutputDir" -ForegroundColor Red
    exit 1
}

Write-Host "Verifying hash integrity in: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Step 1: Require signals_manifest.toml
$manifestPath = Join-Path $OutputDir "signals_manifest.toml"
if (-not (Test-Path $manifestPath)) {
    Write-Host "Error: signals_manifest.toml not found in $OutputDir" -ForegroundColor Red
    Write-Host "The manifest is required to determine expected epochs and files." -ForegroundColor Red
    exit 1
}

# Step 2: Parse manifest to extract expected epochs and file paths
Write-Host "Parsing signals_manifest.toml..." -ForegroundColor Cyan

$manifestContent = Get-Content $manifestPath -Raw
$expectedEpochs = @()
$expectedFiles = @{}  # Key: epoch, Value: hashtable with file paths

# Extract epoch sections: [epoch.YEAR]
$epochPattern = '\[epoch\.(\d+)\]'
$epochMatches = [regex]::Matches($manifestContent, $epochPattern)

foreach ($match in $epochMatches) {
    $year = [int]$match.Groups[1].Value
    $expectedEpochs += $year
    
    # Find the content between this epoch section and the next (or end)
    $startPos = $match.Index + $match.Length
    $nextMatch = $null
    for ($i = 0; $i -lt $epochMatches.Count; $i++) {
        if ($epochMatches[$i].Index -gt $match.Index) {
            $nextMatch = $epochMatches[$i]
            break
        }
    }
    
    $endPos = if ($nextMatch) { $nextMatch.Index } else { $manifestContent.Length }
    $epochContent = $manifestContent.Substring($startPos, $endPos - $startPos)
    
    # Extract file paths from this epoch section
    $epochFiles = @{}
    
    # Extract gxp_hourly_csv
    if ($epochContent -match 'gxp_hourly_csv\s*=\s*"([^"]+)"') {
        $path = $matches[1]
        # Remove "outputs_latest/" prefix if present (paths in manifest are relative to repo root)
        $path = $path -replace '^outputs_latest/', ''
        $epochFiles['gxp_hourly_csv'] = Join-Path $OutputDir $path
    }
    
    # Extract gxp_hourly_report_json
    if ($epochContent -match 'gxp_hourly_report_json\s*=\s*"([^"]+)"') {
        $path = $matches[1]
        $path = $path -replace '^outputs_latest/', ''
        $epochFiles['gxp_hourly_report_json'] = Join-Path $OutputDir $path
    }
    
    # Extract grid_emissions_csv
    if ($epochContent -match 'grid_emissions_csv\s*=\s*"([^"]+)"') {
        $path = $matches[1]
        $path = $path -replace '^outputs_latest/', ''
        $epochFiles['grid_emissions_csv'] = Join-Path $OutputDir $path
    }
    
    # Extract grid_emissions_report_json
    if ($epochContent -match 'grid_emissions_report_json\s*=\s*"([^"]+)"') {
        $path = $matches[1]
        $path = $path -replace '^outputs_latest/', ''
        $epochFiles['grid_emissions_report_json'] = Join-Path $OutputDir $path
    }
    
    $expectedFiles[$year] = $epochFiles
}

$expectedEpochs = $expectedEpochs | Sort-Object

# Step 3: Check for missing files
Write-Host "Checking for expected files..." -ForegroundColor Cyan
$missingFiles = @()
$foundEpochs = @()

foreach ($year in $expectedEpochs) {
    $epochFiles = $expectedFiles[$year]
    $epochComplete = $true
    
    foreach ($fileKey in @('gxp_hourly_csv', 'gxp_hourly_report_json', 'grid_emissions_csv', 'grid_emissions_report_json')) {
        if ($epochFiles.ContainsKey($fileKey)) {
            $filePath = $epochFiles[$fileKey]
            if (-not (Test-Path $filePath)) {
                $missingFiles += [PSCustomObject]@{
                    Epoch = $year
                    File = Split-Path $filePath -Leaf
                    Type = $fileKey
                }
                $epochComplete = $false
            }
        }
    }
    
    if ($epochComplete) {
        $foundEpochs += $year
    }
}

# Step 4: Report summary
Write-Host ""
Write-Host "Expected epochs: $($expectedEpochs -join ', ')" -ForegroundColor Cyan
Write-Host "Found epochs: $($foundEpochs -join ', ')" -ForegroundColor $(if ($foundEpochs.Count -eq $expectedEpochs.Count) { "Green" } else { "Yellow" })

$missingEpochs = $expectedEpochs | Where-Object { $_ -notin $foundEpochs }
if ($missingEpochs.Count -gt 0) {
    Write-Host "Missing epochs: $($missingEpochs -join ', ')" -ForegroundColor Red
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing files:" -ForegroundColor Red
    foreach ($missing in $missingFiles) {
        Write-Host "  [MISSING] Epoch $($missing.Epoch): $($missing.File)" -ForegroundColor Red
    }
}

# Step 5: Exit if files are missing
if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Result: FAIL - Missing expected files" -ForegroundColor Red
    exit 1
}

# Step 6: Proceed with hash verification if all files exist
Write-Host ""
Write-Host "All expected files present. Proceeding with hash verification..." -ForegroundColor Green
Write-Host ""

$allPassed = $true
$results = @()

foreach ($year in $expectedEpochs) {
    $epochFiles = $expectedFiles[$year]
    
    # Verify GXP hourly CSV
    if ($epochFiles.ContainsKey('gxp_hourly_csv') -and $epochFiles.ContainsKey('gxp_hourly_report_json')) {
        $csvPath = $epochFiles['gxp_hourly_csv']
        $reportPath = $epochFiles['gxp_hourly_report_json']
        
        if ((Test-Path $csvPath) -and (Test-Path $reportPath)) {
            try {
                $report = Get-Content $reportPath -Raw | ConvertFrom-Json
                $csvHash = (Get-FileHash -Path $csvPath -Algorithm SHA256).Hash.ToLower()
                
                # Get expected hash from nested files structure (preferred) or top-level alias
                $expectedHash = $null
                if ($report.files -and $report.files.gxp_hourly_csv -and $report.files.gxp_hourly_csv.sha256) {
                    $expectedHash = $report.files.gxp_hourly_csv.sha256.ToLower()
                } elseif ($report.file_hash_sha256) {
                    $expectedHash = $report.file_hash_sha256.ToLower()
                }
                
                if ($expectedHash) {
                    $match = ($csvHash -eq $expectedHash)
                    $results += [PSCustomObject]@{
                        File = Split-Path $csvPath -Leaf
                        Type = "GXP hourly"
                        Match = $match
                        CSVHash = $csvHash.Substring(0, 16)
                        ExpectedHash = $expectedHash.Substring(0, 16)
                    }
                    
                    if (-not $match) {
                        $allPassed = $false
                    }
                } else {
                    Write-Host "Warning: Could not find hash in report for $csvPath" -ForegroundColor Yellow
                    $allPassed = $false
                }
            }
            catch {
                Write-Host "Error processing $csvPath : $($_.Exception.Message)" -ForegroundColor Red
                $allPassed = $false
            }
        }
    }
    
    # Verify Emissions CSV
    if ($epochFiles.ContainsKey('grid_emissions_csv') -and $epochFiles.ContainsKey('grid_emissions_report_json')) {
        $csvPath = $epochFiles['grid_emissions_csv']
        $reportPath = $epochFiles['grid_emissions_report_json']
        
        if ((Test-Path $csvPath) -and (Test-Path $reportPath)) {
            try {
                $report = Get-Content $reportPath -Raw | ConvertFrom-Json
                $csvHash = (Get-FileHash -Path $csvPath -Algorithm SHA256).Hash.ToLower()
                
                # Get expected hash from nested files structure
                $expectedHash = $null
                if ($report.files -and $report.files.emissions_csv -and $report.files.emissions_csv.sha256) {
                    $expectedHash = $report.files.emissions_csv.sha256.ToLower()
                }
                
                if ($expectedHash) {
                    $match = ($csvHash -eq $expectedHash)
                    $results += [PSCustomObject]@{
                        File = Split-Path $csvPath -Leaf
                        Type = "Emissions"
                        Match = $match
                        CSVHash = $csvHash.Substring(0, 16)
                        ExpectedHash = $expectedHash.Substring(0, 16)
                    }
                    
                    if (-not $match) {
                        $allPassed = $false
                    }
                } else {
                    Write-Host "Warning: Could not find hash in report for $csvPath" -ForegroundColor Yellow
                    $allPassed = $false
                }
            }
            catch {
                Write-Host "Error processing $csvPath : $($_.Exception.Message)" -ForegroundColor Red
                $allPassed = $false
            }
        }
    }
}

# Display hash verification results grouped by type
if ($results.Count -gt 0) {
    Write-Host "Hash verification results:" -ForegroundColor Cyan
    $grouped = $results | Group-Object Type
    
    foreach ($group in $grouped) {
        Write-Host "$($group.Name):" -ForegroundColor Cyan
        foreach ($result in $group.Group) {
            $status = if ($result.Match) { "PASS" } else { "FAIL" }
            $color = if ($result.Match) { "Green" } else { "Red" }
            Write-Host "  $($result.File): $status (CSV: $($result.CSVHash)..., Report: $($result.ExpectedHash)...)" -ForegroundColor $color
        }
        Write-Host ""
    }
}

# Final summary
Write-Host "Expected epochs: $($expectedEpochs -join ', ')" -ForegroundColor Cyan
Write-Host "Found epochs: $($foundEpochs -join ', ')" -ForegroundColor $(if ($foundEpochs.Count -eq $expectedEpochs.Count) { "Green" } else { "Yellow" })
if ($missingEpochs.Count -gt 0) {
    Write-Host "Missing epochs: $($missingEpochs -join ', ')" -ForegroundColor Red
}

if ($allPassed -and $foundEpochs.Count -eq $expectedEpochs.Count) {
    Write-Host "Result: PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Result: FAIL" -ForegroundColor Red
    exit 1
}
