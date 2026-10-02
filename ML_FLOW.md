You're right. Let me revise to include the **model comparison and finalization** step that happens in the EDA/tuning phase:

```text
New listings data
       ↓
EDA: inspect data quality, distributions, patterns
       ↓
Compare multiple models (baseline, tuning, feature engineering)
       ↓
Select best model approach
       ↓
Finalize training pipeline (Impute → Scale/OneHot → LinearRegression)
       ↓
Train model on full dataset
       ↓
Evaluate (RMSE, MAE, R²)
       ↓
Model registry checks quality gate (R² drop threshold)
       ↓
Promote to production if acceptable
       ↓
FastAPI loads and serves predictions
       ↓
Angular UI sends input → receives predicted price
       ↓
Scheduler repeats: fetch new data → retrain → evaluate → promote
```

The key phases:
1. **EDA** — understand the data
2. **Model comparison** — test different approaches (in `experiments/compare_models.py`)
3. **Model finalization** — lock in the best pipeline (`train.py`)
4. **Training & evaluation** — fit and measure performance
5. **Quality gate** — only promote if R² doesn't degrade
6. **Production serving** — API uses the vetted model
7. **Continuous retrain loop** — scheduler keeps it fresh with new data

This matches the README's structure: experiments are for EDA/tuning/comparison, and `train.py` implements the finalized pipeline.
