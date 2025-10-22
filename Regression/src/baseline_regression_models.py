from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

def run_base_regressors(features_dict,verbose=True):
    #Run multiple regressors(RandomForest, XGBoost, SVR) for each featurizer dataset.
    results = []
    trained_models = {}
    
    for feat_name, df_feat in features_dict.items():
        if verbose:
            print(f"Running models for featurizer: {feat_name}")

        #print(df_feat.columns)

        X = df_feat.drop(columns=["canonical_smiles", "pIC50"],axis=1).values
        y = df_feat["pIC50"].values

        #print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        #check if features are binary or not
        binary_like = np.isin(X_train.ravel()[: min(10000, X_train.size)], [0, 1]).all()

        if not binary_like:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if verbose:
                print("Applied Standard scaling (continuous features detected).")
        else:
            scaler = None
            if verbose:
                print(" Skipping scaling (binary fingerprint detected).")




        #standardize measurements
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        models = {
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
            "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.01, random_state=0),
            "SVR": SVR(kernel="rbf", C=3.0, epsilon=0.1)
        }

        trained_models[feat_name] = {}

        for model_name, model in models.items():
            if verbose:
                print(f"Training {model_name}...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append({
                "Featurizer": feat_name,
                "Model": model_name,
                "R2": r2,
                "RMSE": rmse
            })

            trained_models[feat_name][model_name] = {
                "model": model,
                "scaler": scaler,
                "r2": r2,
                "rmse": rmse
            }

    results_df = pd.DataFrame(results).sort_values(["Featurizer", "R2"], ascending=[True, False])

    if verbose:
        print("Finished training all models.")
        print(results_df)

    return results_df, trained_models


def run_base_regressors_kv(features_dict,n_splits=10,verbose=True):
    results = []
    best_models = {}

    model_defs = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=0),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=0, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=3.0, epsilon=0.1)
    }

    for feat_name, df_feat in features_dict.items():
        if verbose:
            print(f" Featurizer: {feat_name}")

        X = df_feat.drop(columns=["canonical_smiles", "pIC50"], axis=1).values
        y = df_feat["pIC50"].values

        # Detect binary features (fingerprints)
        binary_like = np.isin(X.ravel()[: min(10000, X.size)], [0, 1]).all()
        use_scaler = not binary_like

        if use_scaler and verbose:
            print("Continuous descriptors:using StandardScaler.")
        elif verbose:
            print("Binary fingerprints: no scaling applied.")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_r2 = -np.inf
        best_model_name = None
        best_model_pipe = None

        # Run CV for each regressor
        for model_name, base_model in model_defs.items():
            model = Pipeline([
                ('scaler', StandardScaler()) if use_scaler else ('identity', 'passthrough'),
                ('regressor', clone(base_model))
            ])

            r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
            rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1))

            mean_r2 = np.mean(r2_scores)
            mean_rmse = np.mean(rmse_scores)

            results.append({
                "Featurizer": feat_name,
                "Model": model_name,
                "R2_mean": mean_r2,
                "R2_std": np.std(r2_scores),
                "RMSE_mean": mean_rmse,
                "RMSE_std": np.std(rmse_scores)
            })

            if verbose:
                print(f" {model_name}: R²={mean_r2:.3f}, RMSE={mean_rmse:.3f}")

            # Track best model
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_model_name = model_name
                best_model_pipe = clone(model)

        # Refit best model on all available data
        best_model_pipe.fit(X, y)
        best_models[feat_name] = {
            "model_name": best_model_name,
            "model": best_model_pipe,
            "use_scaler": use_scaler,
            "r2_cv": best_r2
        }

        if verbose:
            print(f" Best model for {feat_name}: {best_model_name} (R²={best_r2:.3f})")

    results_df = pd.DataFrame(results).sort_values(["Featurizer", "R2_mean"], ascending=[True, False])

    if verbose:
        print("Cross-validation complete.\n")
        print(results_df)

    return results_df, best_models


def run_gridsearch(features_dict, verbose=True):
    
    results = []
    trained_models = {}

    # Models + parameter grids
    models = {
        "RandomForest": (RandomForestRegressor(random_state=0),
                         {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 10, 20]}),
        "XGBoost": (XGBRegressor(random_state=0, verbosity=0),
                    {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.01, 0.1], "model__max_depth": [None,10, 20]}),
        "SVR": (SVR(kernel="rbf"),
                {"model__C": [1.0, 3.0], "model__epsilon": [0.1, 0.2], "model__gamma": ["scale", "auto"]})
    }

    for feat_name, df_feat in features_dict.items():
        if verbose:
            print(f" Featurizer: {feat_name}")

        X = df_feat.drop(columns=["canonical_smiles", "pIC50"], axis=1).values
        y = df_feat["pIC50"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        binary_like = np.isin(X_train.ravel()[: min(10000, X_train.size)], [0, 1]).all()
        scaler = StandardScaler() if not binary_like else None
        if verbose:
            print("Scaling applied." if scaler else "Skipping scaling (binary features).")

        trained_models[feat_name] = {}

        for model_name, (model, param_grid) in models.items():
            steps = [("scaler", scaler)] if scaler else []
            steps.append(("model", model))
            pipeline = Pipeline(steps)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="r2",
            )
            grid.fit(X_train, y_train)
            estimator = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = estimator.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append({
                "Featurizer": feat_name,
                "Model": model_name,
                "Best_Params": best_params,
                "R2": r2,
                "RMSE": rmse
            })

            trained_models[feat_name][model_name] = {"model": estimator, "r2": r2, "rmse": rmse}

    results_df = pd.DataFrame(results).sort_values(["Featurizer", "R2"], ascending=[True, False])
    if verbose:
        print(" Finished GridSearchCV for all models.")
        print(results_df)

    return results_df, trained_models

    
