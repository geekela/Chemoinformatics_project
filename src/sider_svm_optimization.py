from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform

def optimize_svm(X_train_scaled, y_train):
    """
    Optimizes a multi-label SVM model using RandomizedSearchCV on pre-scaled training data.
    """
    
    # --- 1. Base Model Setup ---
    svm_base = OneVsRestClassifier(
        SVC(probability=True, random_state=0)
    )

    # --- 2. Hyperparameter Distributions for Randomized Search ---
    param_distributions = {
        'estimator__kernel': ['rbf'],
        'estimator__C': [0.1, 1, 10, 50, 100],
        'estimator__gamma': ['scale', 0.001, 0.01, 0.1]
    }

    # --- 3. Randomized Search Setup ---
    random_search_svm = RandomizedSearchCV(
        estimator=svm_base,
        param_distributions=param_distributions,
        n_iter=8,
        cv=3,
        scoring='roc_auc_ovr',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    print("\n--- Optimizing SVM with RandomizedSearchCV... ---")
    
    # --- 4. Run the Search ---
    # Fit on the pre-split and pre-scaled training data
    random_search_svm.fit(X_train_scaled, y_train)

    print(f"\nSVM - Best Hyperparameters found: {random_search_svm.best_params_}")
    
    # Return the best model found
    return random_search_svm.best_estimator_
