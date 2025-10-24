from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform
from src.sider_baseline_models import evaluate_multilabel_model_internal

def optimize_svm(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Optimizes a multi-label SVM model using RandomizedSearchCV on pre-scaled training data.
    """

    svm_base = OneVsRestClassifier(
        SVC(probability=True, random_state=0)
    )

    param_distributions = {
        'estimator__kernel': ['rbf', 'linear'],
        'estimator__C': [0.1, 1, 10, 50, 100],
        'estimator__gamma': ['scale', 0.001, 0.01, 0.1]
    }

    random_search_svm = RandomizedSearchCV(
        estimator=svm_base,
        param_distributions=param_distributions,
        n_iter=8,
        cv=5,
        scoring='roc_auc_ovr',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    print("\n--- Optimizing SVM with RandomizedSearchCV... ---")

    random_search_svm.fit(X_train_scaled, y_train)

    print(f"\nSVM - Best Hyperparameters found: {random_search_svm.best_params_}")
    best_svm=random_search_svm.best_estimator_
    macro_auc, detailed_df = evaluate_multilabel_model_internal(best_svm, X_test_scaled, y_test)
    print(f"\nSVM Test Macro AUC-ROC: {macro_auc:.4f}")


    return best_svm, macro_auc, detailed_df