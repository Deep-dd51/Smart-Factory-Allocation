"""
Model Comparison
----------------

Compares the three trained model results:

1. Linear Regression
2. Random Forest
3. Gradient Boosting

Lower MAE/RMSE is better.
Higher R2 is better.
"""

import pandas as pd


# -------------------------------------------------------------------
# Model results
# -------------------------------------------------------------------

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
        ],
        "MAE": [
            0.8343,
            0.6277,
            0.8512,
        ],
        "RMSE": [
            1.0510,
            0.8272,
            1.0356,
        ],
        "R2": [
            0.6575,
            0.7878,
            0.6675,
        ],
    }
)


# -------------------------------------------------------------------
# Display results
# -------------------------------------------------------------------

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(index=False)
)


# -------------------------------------------------------------------
# Best models
# -------------------------------------------------------------------

best_mae = results.loc[
    results["MAE"].idxmin()
]

best_rmse = results.loc[
    results["RMSE"].idxmin()
]

best_r2 = results.loc[
    results["R2"].idxmax()
]


print("\n" + "=" * 70)
print("BEST MODEL BY METRIC")
print("=" * 70)

print(
    f"Best MAE  : "
    f"{best_mae['Model']} "
    f"({best_mae['MAE']:.4f})"
)

print(
    f"Best RMSE : "
    f"{best_rmse['Model']} "
    f"({best_rmse['RMSE']:.4f})"
)

print(
    f"Best R²   : "
    f"{best_r2['Model']} "
    f"({best_r2['R2']:.4f})"
)


# -------------------------------------------------------------------
# Overall winner
# -------------------------------------------------------------------

winner = results.loc[
    results["R2"].idxmax()
]


print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)

print(
    f"Selected model: {winner['Model']}"
)

print(
    f"MAE  : {winner['MAE']:.4f} days"
)

print(
    f"RMSE : {winner['RMSE']:.4f} days"
)

print(
    f"R²   : {winner['R2']:.4f}"
)


# -------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)