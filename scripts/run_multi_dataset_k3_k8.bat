@echo off
REM Batch script to run full pipeline for K=3 and K=8 across all datasets
REM Datasets: PubMed, Roman-empire, Minesweeper (Cora already done/in progress)

echo ============================================================
echo Running pipeline for K=3 and K=8 across multiple datasets  
echo ============================================================

set MODEL=GCN

REM Loop through datasets
for %%D in (PubMed Roman-empire Minesweeper) do (
    echo.
    echo ############################################################
    echo # Processing Dataset: %%D
    echo ############################################################
    echo.
    
    REM Loop through K values (3 and 8)
    for %%K in (3 8) do (
        echo.
        echo ========== %%D, K=%%K ==========
        echo.
        
        REM Loop through seeds
        for %%S in (0 1 2) do (
            echo.
            echo [Dataset=%%D, K=%%K, Seed=%%S]
            
            REM Train
            echo [1/4] Training...
            conda run -n gdl python -m src.train_gnn --dataset %%D --model %MODEL% --K %%K --seed %%S
            
            REM Extract embeddings
            echo [2/4] Extracting embeddings...
            conda run -n gdl python -m src.extract_embeddings --dataset %%D --model %MODEL% --K %%K --seed %%S
            
            REM Probe
            echo [3/4] Probing...
            conda run -n gdl python -m src.probe --dataset %%D --model %MODEL% --K %%K --seed %%S
        )
        
        REM Separability analysis (once per K, all seeds)
        echo [4/4] Running separability analysis for %%D, K=%%K...
        conda run -n gdl python -m src.separability_metrics --dataset %%D --model %MODEL% --K %%K --seed all
    )
)

echo.
echo ============================================================
echo Multi-dataset pipeline complete!
echo ============================================================
pause
