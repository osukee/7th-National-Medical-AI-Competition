# Monitor exp_002 completion and trigger exp_003
# Usage: .\monitor_and_trigger.ps1

$repo = "osukee/7th-National-Medical-AI-Competition"
$workflowFile = "train-kaggle.yml"
$checkInterval = 60  # seconds

Write-Host "=== Experiment Workflow Monitor ===" -ForegroundColor Cyan
Write-Host "Monitoring exp_002_augmentation (Run #30) completion..."
Write-Host "Will auto-trigger exp_003_architecture when complete."
Write-Host ""

# Function to check workflow status via browser
function Check-WorkflowStatus {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Checking workflow status..." -ForegroundColor Yellow
    
    # Open browser to check (user will see status)
    Start-Process "https://github.com/$repo/actions/workflows/$workflowFile"
    
    $response = Read-Host "Enter status (running/success/failure) or 'q' to quit"
    return $response
}

# Main monitoring loop
$completed = $false
while (-not $completed) {
    Write-Host ""
    Write-Host "Press Enter to check status, or type 'done' if exp_002 completed:" -ForegroundColor Green
    $input = Read-Host
    
    if ($input -eq "done" -or $input -eq "success") {
        Write-Host ""
        Write-Host "=== exp_002 COMPLETED ===" -ForegroundColor Green
        
        # Prompt for metrics
        $ssim = Read-Host "Enter SSIM value (or press Enter to skip)"
        $psnr = Read-Host "Enter PSNR value (or press Enter to skip)"
        
        Write-Host ""
        Write-Host "Results recorded:" -ForegroundColor Cyan
        if ($ssim) { Write-Host "  SSIM: $ssim" }
        if ($psnr) { Write-Host "  PSNR: $psnr" }
        
        $completed = $true
    }
    elseif ($input -eq "failed" -or $input -eq "failure") {
        Write-Host "exp_002 failed. Check logs and retry." -ForegroundColor Red
        exit 1
    }
    elseif ($input -eq "q" -or $input -eq "quit") {
        Write-Host "Monitoring stopped." -ForegroundColor Yellow
        exit 0
    }
    else {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Still running... Will check again." -ForegroundColor Gray
        Write-Host "Opening GitHub Actions page..." -ForegroundColor Gray
        Start-Process "https://github.com/$repo/actions/workflows/$workflowFile"
    }
}

# Trigger exp_003_architecture
Write-Host ""
Write-Host "=== Triggering exp_003_architecture ===" -ForegroundColor Cyan

# Switch to exp_003 branch and push to trigger workflow
Write-Host "Switching to experiment/exp_003_architecture branch..."
git checkout experiment/exp_003_architecture

if ($LASTEXITCODE -eq 0) {
    Write-Host "Branch switched successfully. Pushing to trigger workflow..."
    git push origin experiment/exp_003_architecture
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=== exp_003_architecture TRIGGERED ===" -ForegroundColor Green
        Write-Host "Check status at: https://github.com/$repo/actions/workflows/$workflowFile"
        Start-Process "https://github.com/$repo/actions/workflows/$workflowFile"
    }
    else {
        Write-Host "Failed to push. Try manually triggering via GitHub Actions." -ForegroundColor Red
    }
}
else {
    Write-Host "Branch not found. Creating exp_003_architecture branch..." -ForegroundColor Yellow
    Write-Host "Please create the branch manually or use the quick_experiment.py script."
}

Write-Host ""
Write-Host "=== Monitoring Complete ===" -ForegroundColor Cyan
