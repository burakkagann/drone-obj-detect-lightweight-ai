@echo off
REM ==================================================================
REM Phase 1 Baseline Training Pipeline - YOLOv5n + VisDrone
REM Protocol: extended_phase_protocol.md - Complete Phase 1 Execution
REM 
REM CRITICAL: NO AUGMENTATION - This is the true baseline
REM ==================================================================

echo.
echo ===============================================================
echo PHASE 1 BASELINE TRAINING PIPELINE - YOLOv5n
echo ===============================================================
echo Protocol: NO AUGMENTATION (True Baseline)
echo Dataset: VisDrone (Clean Training + Synthetic Weather Testing)
echo Expected Duration: 8-12 hours total
echo ===============================================================
echo.

REM Set script directory 
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..\..\..\..\..

echo [%TIME%] Using pre-activated virtual environment
echo.

REM Change to script directory
cd /d "%SCRIPT_DIR%"

REM ===========================================
REM STEP 1: BASELINE TRAINING
REM ===========================================
echo ===============================================================
echo STEP 1/3: BASELINE TRAINING (NO AUGMENTATION)
echo ===============================================================
echo Expected Duration: 6-8 hours
echo GPU Required: NVIDIA GPU recommended
echo.

echo [%TIME%] Starting Phase 1 baseline training...
python train_baseline.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Baseline training failed!
    echo Check logs and resolve issues before proceeding.
    echo.
    pause
    exit /b 1
)

echo.
echo [%TIME%] ✓ Baseline training completed successfully
echo.

REM ===========================================
REM STEP 2: BASELINE VALIDATION
REM ===========================================
echo ===============================================================
echo STEP 2/3: BASELINE VALIDATION (CLEAN TEST SET)
echo ===============================================================
echo Expected Duration: 15-30 minutes
echo Purpose: Establish baseline performance metrics
echo.

echo [%TIME%] Starting baseline validation...
python validate_baseline.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Baseline validation failed!
    echo Training may have issues - check weights and data.
    echo.
    pause
    exit /b 1
)

echo.
echo [%TIME%] ✓ Baseline validation completed successfully
echo.

REM ===========================================
REM STEP 3: WEATHER CONDITIONS TESTING
REM ===========================================
echo ===============================================================
echo STEP 3/3: WEATHER DEGRADATION ANALYSIS
echo ===============================================================
echo Expected Duration: 1-2 hours
echo Testing Conditions: Clean, Fog, Rain, Night, Mixed
echo Purpose: Measure performance degradation for thesis
echo.

echo [%TIME%] Starting weather conditions testing...
python test_all_conditions.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Weather testing failed!
    echo Some conditions may have failed - check individual results.
    echo.
    pause
    exit /b 1
)

echo.
echo [%TIME%] ✓ Weather conditions testing completed successfully
echo.

REM ===========================================
REM COMPLETION SUMMARY
REM ===========================================
echo ===============================================================
echo PHASE 1 BASELINE PIPELINE COMPLETED SUCCESSFULLY!
echo ===============================================================
echo.
echo ✓ Baseline training: COMPLETE (NO augmentation)
echo ✓ Baseline validation: COMPLETE (Clean test set)
echo ✓ Weather testing: COMPLETE (All synthetic conditions)
echo.
echo Results Location:
echo - Training: %PROJECT_ROOT%\runs\phase1\yolov5n_baseline\
echo - Validation: %PROJECT_ROOT%\runs\phase1\yolov5n_baseline\validation\
echo - Weather Testing: %PROJECT_ROOT%\runs\phase1\yolov5n_baseline\weather_testing\
echo.
echo Key Outputs:
echo ✓ Model weights: best.pt, last.pt
echo ✓ Training metrics: results.csv
echo ✓ Validation report: phase1_baseline_summary.md
echo ✓ Weather analysis: weather_degradation_report.md
echo ✓ Performance plots: weather_performance_analysis.png
echo.
echo Next Steps:
echo 1. Review weather degradation results
echo 2. Analyze performance drops by condition
echo 3. Use findings to justify Phase 2 augmentation
echo 4. Proceed to Phase 2 training with weather augmentation
echo.
echo ===============================================================
echo THESIS CONTRIBUTION: Baseline performance established
echo RESEARCH VALUE: Weather vulnerability quantified  
echo ===============================================================

REM Log completion
echo [%TIME%] Phase 1 baseline pipeline completed successfully > phase1_completion.log
echo Pipeline completed at %DATE% %TIME% >> phase1_completion.log

echo.
echo Press any key to view results summary...
pause > nul

REM Try to open results directory
if exist "%PROJECT_ROOT%\runs\phase1\yolov5n_baseline\weather_testing\weather_degradation_report.md" (
    echo Opening weather degradation report...
    start "" "%PROJECT_ROOT%\runs\phase1\yolov5n_baseline\weather_testing\weather_degradation_report.md"
)

if exist "%PROJECT_ROOT%\runs\phase1\yolov5n_baseline\weather_testing\weather_performance_analysis.png" (
    echo Opening performance visualization...
    start "" "%PROJECT_ROOT%\runs\phase1\yolov5n_baseline\weather_testing\weather_performance_analysis.png"
)

echo.
echo Phase 1 baseline training pipeline completed!
echo Check the opened files for detailed results.
echo.
pause