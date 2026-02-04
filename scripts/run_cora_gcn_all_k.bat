@echo off
REM Batch script to run full pipeline for Cora GCN K=0-7
REM K=8 is already complete

echo ============================================================
echo Running full pipeline for Cora/GCN, K=0 to 7
echo ============================================================

set DATASET=Cora
set MODEL=GCN

REM Loop through K values 0-7
for %%K in (0 1 2 4 5 6 7) do (
    echo.
    echo ############################################################
    echo # Processing K=%%K
    echo ############################################################
    echo.
    
    REM Loop through seeds
    for %%S in (0 1 2) do (
        echo.
        echo ========== K=%%K, Seed=%%S ==========
        
        REM Train
        echo [1/4] Training...
        conda run -n gdl python -m src.train_gnn --dataset %DATASET% --model %MODEL% --K %%K --seed %%S
        
        REM Extract embeddings
        echo [2/4] Extracting embeddings...
        conda run -n gdl python -m src.extract_embeddings --dataset %DATASET% --model %MODEL% --K %%K --seed %%S
        
        REM Probe
        echo [3/4] Probing...
        conda run -n gdl python -m src.probe --dataset %DATASET% --model %MODEL% --K %%K --seed %%S
       
    )
    
    REM Separability analysis (once per K, all seeds)
    echo [4/4] Running separability analysis for K=%%K...
    conda run -n gdl python -m src.separability_metrics --dataset %DATASET% --model %MODEL% --K %%K --seed all
)

echo.
echo ============================================================
echo Pipeline complete!
echo ============================================================
