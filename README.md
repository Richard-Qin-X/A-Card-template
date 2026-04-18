# A-Card Template

## Introduction
A-Card Template is a project for building highly interpretable credit scoring cards based on the German Credit Data dataset. Designed for internal use in banking and financial institutions, this template emphasizes transparency and explainability in credit risk assessment, providing a structured workflow for data preprocessing, feature engineering, and scorecard development.

## Directory Structure
```
data/      # Data files
config/    # Settings & mappings
src/       # Core modules (preprocessing, binning, woe, model, scorecard)
notebooks/ # EDA & demos
reports/   # Model reports
main.py    # One-click pipeline
```

## Main Modules
- `preprocessing.py`: Data cleaning and target definition
- `binning.py`: Automatic and manual binning
- `woe_transformer.py`: WOE/IV calculation
- `model.py`: Logistic regression with cost matrix
- `scorecard.py`: Probability-to-score conversion

## License
[MIT License](LICENSE)