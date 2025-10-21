import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Performs the complete multi-label modeling pipeline using pre-split and pre-scaled data.
    """

    # ======================================================================
    # 2. INTERNAL EVALUATION AND ANALYSIS FUNCTIONS
    # ======================================================================

    def analyze_individual_labels(y_true, y_score, label_names, model_name):
        label_scores = []
        for i, label_name in enumerate(label_names):
            try:
                auc = roc_auc_score(y_true.iloc[:, i], y_score[:, i])
            except ValueError:
                auc = np.nan
            label_scores.append((label_name, auc))
        return pd.DataFrame(label_scores, columns=['Label', model_name]).dropna()

    def evaluate_multilabel_model_internal(model, X_test_data, y_test_data):
        y_pred_proba = model.predict_proba(X_test_data)
        if isinstance(y_pred_proba, list):
            y_pred_proba_final = np.column_stack([proba[:, 1] for proba in y_pred_proba])
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == y_test_data.shape[1] * 2:
            y_pred_proba_final = y_pred_proba[:, 1::2]
        else:
            y_pred_proba_final = y_pred_proba

        if y_pred_proba_final.shape != y_test_data.shape:
            raise ValueError(f"Shape mismatch: Predicted {y_pred_proba_final.shape} vs Actual {y_test_data.shape}")

        roc_auc_macro = roc_auc_score(y_test_data, y_pred_proba_final, average='macro')
        detailed_df = analyze_individual_labels(y_test_data, y_pred_proba_final, y_test_data.columns, 'AUC-ROC')
        return roc_auc_macro, detailed_df

    def print_final_summary(summary_df, detailed_dfs):
        print("\n" + "="*80)
        print("                     MODEL PERFORMANCE SUMMARY (MACRO AUC-ROC)")
        print("="*80)
        summary_df = summary_df.sort_values(by='Macro AUC-ROC', ascending=False)
        print(summary_df.to_string(float_format="{:.4f}".format))

        print("\n" + "="*80)
        print("           DETAILED LABEL ANALYSIS (TOP 5 & BOTTOM 5 AUC-ROC ACROSS MODELS)")
        print("="*80)
        final_detailed_df = detailed_dfs[0]
        for df in detailed_dfs[1:]:
            final_detailed_df = pd.merge(final_detailed_df, df, on='Label', how='outer')
        final_detailed_df.columns = ['Label', 'Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
        final_detailed_df['Average AUC'] = final_detailed_df.iloc[:, 1:].mean(axis=1)

        print("\n--- TOP 5 LABELS (Easiest Classes on Average) ---")
        print(final_detailed_df.sort_values(by='Average AUC', ascending=False).head(5)
              .drop(columns=['Average AUC']).to_string(index=False, float_format="{:.4f}".format))

        print("\n--- BOTTOM 5 LABELS (Hardest Classes on Average) ---")
        print(final_detailed_df.sort_values(by='Average AUC', ascending=True).head(5)
              .drop(columns=['Average AUC']).to_string(index=False, float_format="{:.4f}".format))

    # ======================================================================
    # 3. MODEL TRAINING & OPTIMIZATION
    # ======================================================================

    # --- Tree-based models (use UNSCALED data) ---
    print("\n--- Optimizing Random Forest... ---")
    rf_base = RandomForestClassifier(random_state=0)
    rf_multilabel_wrapper = MultiOutputClassifier(rf_base, n_jobs=-1)
    param_grid_rf = {'estimator__n_estimators': [50, 100], 'estimator__max_depth': [5, 10]}
    grid_search_rf = GridSearchCV(estimator=rf_multilabel_wrapper, param_grid=param_grid_rf, cv=3, scoring='roc_auc_ovr', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    print(f"RF - Best Hyperparameters: {grid_search_rf.best_params_}")

    print("--- Training XGBoost... ---")
    xgb_base = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=0)
    multilabel_model_xgb = MultiOutputClassifier(xgb_base, n_jobs=-1)
    multilabel_model_xgb.fit(X_train, y_train)

    # --- Scale-sensitive models (use SCALED data) ---
    print("--- Training Logistic Regression... ---")
    multilabel_model_lr = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=0))
    multilabel_model_lr.fit(X_train_scaled, y_train)

    print("--- Training SVM... (Warning: May be very slow) ---")
    multilabel_model_svm = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
    multilabel_model_svm.fit(X_train_scaled, y_train)

    # ======================================================================
    # 4. RESULTS
    # ======================================================================

    print("\n\n--- EVALUATION AND RESULTS COLLECTION ---")
    results = {}
    all_detailed_dfs = []

    # Evaluate tree-based models on UNSCALED test data
    macro_rf, detailed_rf = evaluate_multilabel_model_internal(best_rf_model, X_test, y_test)
    results['Random Forest'] = {'Macro AUC-ROC': macro_rf, 'Hyperparameters': str(grid_search_rf.best_params_)}
    all_detailed_dfs.append(detailed_rf.rename(columns={'AUC-ROC': 'Random Forest'}))

    macro_xgb, detailed_xgb = evaluate_multilabel_model_internal(multilabel_model_xgb, X_test, y_test)
    results['XGBoost'] = {'Macro AUC-ROC': macro_xgb, 'Hyperparameters': 'Defaults'}
    all_detailed_dfs.append(detailed_xgb.rename(columns={'AUC-ROC': 'XGBoost'}))

    # Evaluate scale-sensitive models on SCALED test data
    macro_lr, detailed_lr = evaluate_multilabel_model_internal(multilabel_model_lr, X_test_scaled, y_test)
    results['Logistic Regression'] = {'Macro AUC-ROC': macro_lr, 'Hyperparameters': 'C=1.0 (Default), Solver=liblinear'}
    all_detailed_dfs.append(detailed_lr.rename(columns={'AUC-ROC': 'Logistic Regression'}))

    macro_svm, detailed_svm = evaluate_multilabel_model_internal(multilabel_model_svm, X_test_scaled, y_test)
    results['SVM'] = {'Macro AUC-ROC': macro_svm, 'Hyperparameters': 'Kernel=linear, C=1.0 (Default)'}
    all_detailed_dfs.append(detailed_svm.rename(columns={'AUC-ROC': 'SVM'}))

    summary_df = pd.DataFrame(results).T
    summary_df['Hyperparameters'] = summary_df['Hyperparameters'].astype(str)

    # FINAL OUTPUT
    print_final_summary(summary_df, all_detailed_dfs)
