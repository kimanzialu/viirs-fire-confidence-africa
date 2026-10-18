##viirs-fire-confidence-africa

In this project, I compare traditional machine learning (scikit-learn) and deep learning (TensorFlow/Keras) for classifying the confidence level (low / nominal / high) that NASA assigns to VIIRS satellite fire detections in five African countries.

Data:
- **Source:** NASA FIRMS, VIIRS S-NPP 375 m active-fire detections (https://firms.modaps.eosdis.nasa.gov/download/)
- **Target:** `confidence` (l = low, n = nominal, h = high)

What the notebook does:
1. Loads and explores the data
2. Cleans it and samples up to 40,000 rows per country-year
3. Splits by year: train 2023 / validation 2024 / test 2025
4. Engineers features and runs 12 experiments: baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, two feature ablations, hyperparameter tuning, and four Keras MLPs
5. Evaluates the final models once on the 2025 test set and analyses errors by country, day/night and pixel saturation
6. Saves models, tables, figures and TensorBoard logs

The main metric is Macro-F1 because the classes are imbalanced.
All metrics are in `results/results_test.csv`. The full experiment table is in `results/results_all.csv`.
