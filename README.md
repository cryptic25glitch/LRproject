# FWI Prediction Using Ridge Regression

This project uses **Ridge Regression** to predict the **Fire Weather Index (FWI)** from the **Algerian Forest Fires dataset**. The model learns from meteorological and environmental features such as temperature, humidity, wind speed, and drought codes.

---

## Dataset

- Source: UCI ML Repository  
- Total Records: 244 (from two regions in Algeria)  
- Target: `FWI` (Fire Weather Index)  
- Features: Temperature, RH, WS, Rain, FFMC, DMC, DC, ISI, etc.

---

## Model

- **Model Used**: Ridge Regression (L2 regularized linear regression)
- **Scaler**: StandardScaler for feature normalization
- **Validation**: K-Fold Cross-Validation
- **Tuning**: GridSearchCV for optimal alpha (λ)

---

## Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score
