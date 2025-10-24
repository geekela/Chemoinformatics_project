import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier



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
        
def print_final_summary(summary_df, detailed_dfs, feat):
        if feat:
            summary_df = summary_df.copy()  # avoid modifying original
            summary_df.insert(0, 'Features', feat)
        print("\n" + "="*80)
        print(f"                     MODEL PERFORMANCE SUMMARY")
        if feat:
            print(f"                     Features: {feat}")
        print("="*80)

        summary_df = summary_df.sort_values(by='Macro AUC-ROC', ascending=False)
        print(summary_df.to_string(float_format="{:.4f}".format))

        # Detailed label analysis
        final_detailed_df = detailed_dfs[0]
        for df in detailed_dfs[1:]:
            final_detailed_df = pd.merge(final_detailed_df, df, on='Label', how='outer')
        final_detailed_df.columns = ['Label', 'Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
        final_detailed_df['Average AUC'] = final_detailed_df.iloc[:, 1:].mean(axis=1)

        print("\n--- TOP 5 LABELS ---")
        print(final_detailed_df.sort_values(by='Average AUC', ascending=False).head(5)
          .drop(columns=['Average AUC']).to_string(index=False, float_format="{:.4f}".format))

        print("\n--- BOTTOM 5 LABELS ---")
        print(final_detailed_df.sort_values(by='Average AUC', ascending=True).head(5)
          .drop(columns=['Average AUC']).to_string(index=False, float_format="{:.4f}".format))      



def evaluate_multilabel_model_internal_cv(model, X_test, y_test):
    """Compute per-label and macro AUC-ROC for multilabel classifier."""
    try:
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)

        # MultiOutputClassifier returns list of arrays
        if isinstance(y_pred_proba, list):
            y_pred_proba = np.vstack([p[:, 1] for p in y_pred_proba]).T

        aucs = []
        label_scores = []

        for i, label_name in enumerate(y_test.columns):
            y_true_label = y_test.iloc[:, i]
            y_pred_label = y_pred_proba[:, i]

            # Skip labels with only one class in test fold
            if len(np.unique(y_true_label)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_true_label, y_pred_label)

            aucs.append(auc)
            label_scores.append((label_name, auc))

        # Compute mean of valid AUCs
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else np.nan

        detailed_df = pd.DataFrame(label_scores, columns=['Label', 'AUC-ROC'])
        return macro_auc, detailed_df

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return np.nan, pd.DataFrame(columns=['Label', 'AUC-ROC'])

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feat,verbose=True):
    """
    Performs the complete multi-label modeling pipeline using pre-split and pre-scaled data.
    """

    # ======================================================================
    # 2. INTERNAL EVALUATION AND ANALYSIS FUNCTIONS
    # ======================================================================
    binary_like = np.isin(X_train.values.ravel()[: min(10000, X_train.size)], [0, 1]).all()
    
    X_train_scaled = X_train
    X_test_scaled = X_test
    scaler = None

    if not binary_like:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if verbose:
            print("Applied StandardScaler (continuous features detected).")
    elif verbose:
        print("Skipping scaling (binary features detected).")
    # ======================================================================
    # 3. MODEL TRAINING & OPTIMIZATION
    # ======================================================================

    # --- Tree-based models (use UNSCALED data) ---
    print("\n--- Hyperparameter search: Random Forest... ---")
    rf_base = RandomForestClassifier(random_state=0)
    rf_multilabel_wrapper = MultiOutputClassifier(rf_base, n_jobs=-1)
    param_grid_rf = {'estimator__n_estimators': [10, 50, 100,200, 300], 'estimator__max_depth': [5, 10,15, 20,25]}
    grid_search_rf = GridSearchCV(estimator=rf_multilabel_wrapper, param_grid=param_grid_rf, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
    grid_search_rf.fit(X_train_scaled, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    print(f"RF - Best Hyperparameters: {grid_search_rf.best_params_}")

    """print("--- Training XGBoost... ---")
    xgb_base = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=0)
    multilabel_model_xgb = MultiOutputClassifier(xgb_base, n_jobs=-1)
    multilabel_model_xgb.fit(X_train, y_train)"""
    
    print("\n--- Hyperparameter search: XGBoost ---")
    xgb_base = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=0)
    xgb_multi = MultiOutputClassifier(xgb_base, n_jobs=-1)
    param_grid_xgb = {'estimator__n_estimators': [10, 50, 100, 200, 300],
                      'estimator__max_depth': [5, 10, 15, 20,25]}
    grid_xgb = GridSearchCV(xgb_multi, param_grid_xgb, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
    grid_xgb.fit(X_train_scaled, y_train)
    best_xgb_model = grid_xgb.best_estimator_


    # --- Scale-sensitive models (use SCALED data) ---
    """print("--- Training Logistic Regression... ---")
    multilabel_model_lr = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=0))
    multilabel_model_lr.fit(X_train_scaled, y_train)"""
    
    # --- Logistic Regression gridsearch ---
    print("\n--- Hyperparameter search: Logistic Regression ---")
    lr_base = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=0))
    param_grid_lr = {'estimator__C': [0.1, 1.0, 2]}
    grid_lr = GridSearchCV(lr_base, param_grid_lr, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
    grid_lr.fit(X_train_scaled, y_train)
    best_lr_model = grid_lr.best_estimator_

    # --- SVM ---
    print("\n--- Hyperparameter search: SVM ---")
    svm_base = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
    param_grid_svm = {'estimator__C': [0.1, 1.0, 10]}
    grid_svm = GridSearchCV(svm_base, param_grid_svm, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
    grid_svm.fit(X_train_scaled, y_train)
    best_svm_model = grid_svm.best_estimator_
    """print("--- Training SVM... (Warning: May be very slow) ---")
    multilabel_model_svm = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
    multilabel_model_svm.fit(X_train_scaled, y_train)"""

    # ======================================================================
    # 4. RESULTS
    # ======================================================================

    print("\n\n--- EVALUATION AND RESULTS COLLECTION ---")
    results = {}
    all_detailed_dfs = []

    # Evaluate tree-based models on  test data
    macro_rf, detailed_rf = evaluate_multilabel_model_internal(best_rf_model, X_test_scaled, y_test)
    results['Random Forest'] = {'Macro AUC-ROC': macro_rf, 'Hyperparameters': str(grid_search_rf.best_params_)}
    all_detailed_dfs.append(detailed_rf.rename(columns={'AUC-ROC': 'Random Forest'}))

    macro_xgb, detailed_xgb = evaluate_multilabel_model_internal( best_xgb_model, X_test_scaled, y_test)
    results['XGBoost'] = {'Macro AUC-ROC': macro_xgb, 'Hyperparameters': 'Defaults'}
    all_detailed_dfs.append(detailed_xgb.rename(columns={'AUC-ROC': 'XGBoost'}))


    macro_lr, detailed_lr = evaluate_multilabel_model_internal(best_lr_model, X_test_scaled, y_test)
    results['Logistic Regression'] = {'Macro AUC-ROC': macro_lr, 'Hyperparameters': 'C=1.0 (Default), Solver=liblinear'}
    all_detailed_dfs.append(detailed_lr.rename(columns={'AUC-ROC': 'Logistic Regression'}))

    macro_svm, detailed_svm = evaluate_multilabel_model_internal(best_svm_model, X_test_scaled, y_test)
    results['SVM'] = {'Macro AUC-ROC': macro_svm, 'Hyperparameters': 'Kernel=linear, C=1.0 (Default)'}
    all_detailed_dfs.append(detailed_svm.rename(columns={'AUC-ROC': 'SVM'}))

    summary_df = pd.DataFrame(results).T
    summary_df['Hyperparameters'] = summary_df['Hyperparameters'].astype(str)

    # FINAL OUTPUT
    print_final_summary(summary_df, all_detailed_dfs,feat)
    return {
        "Features": feat,
        "Random Forest": best_rf_model,
        "XGBoost": best_xgb_model,
        "Logistic Regression": best_lr_model,
        "SVM": best_svm_model,
        "Summary": summary_df,
        "Detailed": all_detailed_dfs
    }




#funciton for running cross validation across multilabel models
def cross_validate_multilabel_models(X, y, feat=None, verbose=True, n_splits=10, random_state=0):
    """
    10-fold cross-validation for multi-label classification.
    
    Models: Random Forest, XGBoost, Logistic Regression, SVM.
    Returns dict with summary, detailed per-label AUCs, per-fold results, and last fold trained models.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {model: [] for model in ["Random Forest", "XGBoost", "Logistic Regression", "SVM"]}
    detailed_results = {model: [] for model in results.keys()}

    fold_idx = 1
    for train_idx, test_idx in kf.split(X):
        if verbose:
            print(f"\n{'='*30} FOLD {fold_idx}/{n_splits} {'='*30}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- Detect binary features and scale if needed ---
        binary_like = np.isin(X_train.values.ravel()[: min(10000, X_train.size)], [0, 1]).all()
        if not binary_like:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            if verbose:
                print("Applied StandardScaler (continuous features detected).")
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
            if verbose:
                print("Skipping scaling (binary features detected).")

        # --- Random Forest ---
        rf = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1))
        rf.fit(X_train_scaled, y_train)
        macro_rf, detailed_rf = evaluate_multilabel_model_internal_cv(rf, X_test_scaled, y_test)
        results["Random Forest"].append(macro_rf)
        detailed_results["Random Forest"].append(
            detailed_rf.rename(columns={'AUC-ROC': f'Fold_{fold_idx}'}))

        # --- XGBoost ---
        xgb = MultiOutputClassifier(XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', n_estimators=200,
            max_depth=15, random_state=random_state, n_jobs=-1))
        xgb.fit(X_train_scaled, y_train)
        macro_xgb, detailed_xgb =  evaluate_multilabel_model_internal_cv(xgb, X_test_scaled, y_test)
        results["XGBoost"].append(macro_xgb)
        detailed_results["XGBoost"].append(
            detailed_xgb.rename(columns={'AUC-ROC': f'Fold_{fold_idx}'}))

        # --- Logistic Regression ---
        lr = OneVsRestClassifier(LogisticRegression(
            C=1.0, solver='liblinear', random_state=random_state, max_iter=1000))
        lr.fit(X_train_scaled, y_train)
        macro_lr, detailed_lr = evaluate_multilabel_model_internal_cv(lr, X_test_scaled, y_test)
        results["Logistic Regression"].append(macro_lr)
        detailed_results["Logistic Regression"].append(
            detailed_lr.rename(columns={'AUC-ROC': f'Fold_{fold_idx}'}))

        # --- SVM ---
        svm = OneVsRestClassifier(SVC(
            kernel='linear', C=1.0, probability=True, random_state=random_state))
        svm.fit(X_train_scaled, y_train)
        macro_svm, detailed_svm = evaluate_multilabel_model_internal_cv(svm, X_test_scaled, y_test)
        results["SVM"].append(macro_svm)
        detailed_results["SVM"].append(
            detailed_svm.rename(columns={'AUC-ROC': f'Fold_{fold_idx}'}))

        fold_idx += 1

    # -------------------------------------------------------------------------
    # Aggregate results across folds
    # -------------------------------------------------------------------------
    avg_results = {model: np.nanmean(scores) for model, scores in results.items()}
    summary_df = pd.DataFrame.from_dict(avg_results, orient='index', columns=['Macro AUC-ROC'])

    # Merge per-label AUCs across folds
    all_detailed_dfs = []
    for model, dfs in detailed_results.items():
        merged_df = dfs[0].copy()
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='Label', how='outer')
        merged_df['Mean AUC'] = merged_df.iloc[:, 1:].mean(axis=1)
        all_detailed_dfs.append(merged_df[['Label', 'Mean AUC']].rename(columns={'Mean AUC': model}))

    # Print summary
    print_final_summary(summary_df, all_detailed_dfs, feat)

    return {
        "Features": feat,
        "Summary": summary_df,
        "Detailed": all_detailed_dfs,
        "PerFoldResults": results,
        "TrainedModels": {"RF": rf, "XGB": xgb, "LR": lr, "SVM": svm}
    }