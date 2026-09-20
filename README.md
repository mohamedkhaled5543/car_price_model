# Used Car Price Predictor

Predicts used car prices from 8K+ listings (CarDekho data).

## Approach
- Cleaned data, engineered `car_age`, `power_per_engine`, `brand`
- One-hot encoding + scaling
- Compared 4 models

## Results
| Model | MAE (₹) | R² |
|---|---|---|
| Random Forest | 73.8K | 0.919216 |
| Ridge (log) | 87.7K | 0.870017 |
| Linear (log) | 87.7K | 0.869979 |
| Linear (raw) | 137.8K | 0.662531 |

## Findings
- Log-transforming price cut error by ~36%.
- Clipping outliers hurt accuracy (R² 0.92 → 0.886). High-end cars carry price signal.

## Run
pip install -r requirements.txt
streamlit run app.py
