# :dart: Scoring Model
Use historical loan application data to predict whether or not an applicant will be able to repay a loan.

**Expected output**: a probability of default prediction, between 0 and 1.

# :card_index_dividers: Dataset
The dataset is provided by [Home Credit](https://www.kaggle.com/c/home-credit-default-risk)

# :scroll: Tasks
- :heavy_check_mark: Exploratory Data Analysis (EDA);
- :heavy_check_mark: Data cleaning;
- :heavy_check_mark: Feature engineering;
- :heavy_check_mark: Imbalanced classes management;
- :heavy_check_mark: Model training: Naïve Bayes, Logistic Regression, Stochastic Gradient Descent, Random Forest, LightGBM
- :heavy_check_mark: Model evaluation: AUC (Area Under the Curve), Recall, F1-Score
- :heavy_check_mark: Hyperparameters optimization;
- :heavy_check_mark: Evaluation of variable importance with SHAP (Shapley Additive exPlanations).

## Distribution of the classes
<img src=".\pictures\imbalanced_classes_distribution.png">

## Performance of lightGBM
<img src=".\pictures\roc_auc_lightgbm.png">

## Global explanation with SHAP
<img src=".\pictures\shap_global_explanation.png">

# :computer: Dependencies
scikit-learn, LightGBM, SHAP