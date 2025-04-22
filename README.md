# R_MiniProject

## Overview

This project focuses on **Mine Detection using Machine Learning (ML)**. It involves preprocessing data, building predictive models, and evaluating their performance to identify landmines effectively.

## 📦 Environment Setup with `renv` and Execution

This project uses [`renv`](https://rstudio.github.io/renv/) for R package management and reproducibility.

To activate the current virtual environment, please execute the following command in your terminal:

```bash
Rscript -e 'renv::activate()'
```

To check the state of project and update the deps:

```bash
Rscript -e "renv::restore()"
```

```bash
Rscript -e 'renv::status()'
```
To view the list of all packages installed, run the following command:

```bash
Rscript -e 'pkgs <- installed.packages()[, c("Package", "Version")]; apply(pkgs, 1, paste, collapse = " ")'
```

To save the newly installed packages, run:

```bash
Rscript -e 'renv::snapshot()' 
```

To run an script, execute it in the following manner:

```bash
Rscript Task3/xgboost.R
```


## ⚙️ Models Evaluated

| Model                | Accuracy| 
|---------------------|----------|
| MLP                 | 0.6119   
| Extra Trees         | 0.4776   
| Random Forest       | 0.4328   
| SVM                 | 0.4627   
| Logistic Regression | 0.2985   
| CatBoost            | 0.5672   

---

## Project Structure
- **Data Preprocessing**: Cleaning and transforming the dataset.
- **Model Building**: Training and evaluating ML models.
- **Visualizations**: Insights through plots like heatmaps, PCA, and histograms.
