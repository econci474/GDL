# Batch probe generation for heterophilous datasets
# Minesweeper and Roman-empire: K=0-8, seeds 0-1, splits 0-9

$datasets = @('Minesweeper', 'Roman-empire')
$K_values = 0..8
$seeds = 0..1
$model = 'GCN'

$total = $datasets.Count * $K_values.Count * $seeds.Count
$completed = 0
$failed = 0

Write-Host "=" * 60
Write-Host "Running probes for heterophilous datasets"
Write-Host "Datasets: $($datasets -join ', ')"
Write-Host "K values: $($K_values -join ', ')"
Write-Host "Seeds: $($seeds -join ', ')"
Write-Host "Total configurations: $total (each processes 10 splits)"
Write-Host "=" * 60

foreach ($dataset in $datasets) {
    foreach ($K in $K_values) {
        foreach ($seed in $seeds) {
            Write-Host "`nProcessing: $dataset K=$K seed=$seed (10 splits)..." -NoNewline
            
            $cmd = "python src/probe.py --dataset $dataset --model $model --K $K --seed $seed"
            
            try {
                $result = Invoke-Expression $cmd 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host " ✓" -ForegroundColor Green
                    $completed++
                } else {
                    Write-Host " ✗" -ForegroundColor Red
                    $failed++
                }
            } catch {
                Write-Host " ✗ (error)" -ForegroundColor Red
                $failed++
            }
        }
    }
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "COMPLETE:"
Write-Host "  Completed: $completed / $total"
Write-Host "  Failed: $failed"
Write-Host ("=" * 60)
