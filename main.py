from tkinter.ttk import Treeview

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skopt.space import Real, Integer, Categorical
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from xgboost import XGBClassifier
import catboost as cb

from DataHandler import DataHandler
from NNModel import NNModel
from SubmissionGenerator import SubmissionGenerator

def print_top_n(bayes_cv, name, n=3):
    # Extract arrays
    means = bayes_cv.cv_results_["mean_test_score"]
    params = bayes_cv.cv_results_["params"]
    # Get indices of top-n scores
    top_idxs = sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:n]
    print(f"\n=== Top {n} for {name} ===")
    for rank, idx in enumerate(top_idxs, start=1):
        print(f"{rank}. mean_accuracy={means[idx]:.4f}, params={params[idx]}")

def test_LG():
    dh = DataHandler()
    X, Y = dh.get_train_data()
    Y = Y["salary_category"].to_numpy().ravel().astype(int)

    # 1) build a Pipeline with PCA → LogisticRegression
    pipe = Pipeline([
        ("pca", PCA()),                 # ← new PCA step
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5_000,
            random_state=0,
        )),
    ])

    # 2) define search space, including PCA n_components and Logistic C
    n_features = X.shape[1]
    search_space = {
        "pca__n_components": Integer(5, min(200, n_features)),   # tune how many PCs
        "clf__C": Real(1e-4, 1e1, prior="log-uniform"),
    }

    # 3) wrap in BayesSearchCV
    bayes_cv = BayesSearchCV(
        pipe,
        search_space,
        n_iter=20,
        cv=10,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2,
    )

    # 4) fit & report
    bayes_cv.fit(X, Y)
    print("Best CV score:", bayes_cv.best_score_)
    print("Best params:", bayes_cv.best_params_)
    print("Total features before PCA:", n_features)

def test_svm(X, Y):
    # === Top 3 for SVM ===
    # 1. mean_accuracy=0.6281, params=OrderedDict({'clf__C': 0.00313459903396157, 'clf__gamma': 0.0002546972022055807, 'clf__kernel': 'linear', 'pca__n_components': 289})
    # 2. mean_accuracy=0.6258, params=OrderedDict({'clf__C': 0.0033459686205026175, 'clf__gamma': 0.0011742805370085377, 'clf__kernel': 'linear', 'pca__n_components': 281})
    # 3. mean_accuracy=0.6203, params=OrderedDict({'clf__C': 0.002742386020789277, 'clf__gamma': 0.0001700997291746021, 'clf__kernel': 'linear', 'pca__n_components': 211})

    n_features = X.shape[1]
    pipe = Pipeline([
        ("pca", PCA()),
        ("clf", SVC(decision_function_shape="ovr", probability=True, random_state=0)),
    ])
    search_space = {
        "pca__n_components": Integer(5, n_features),
        "clf__C":          Real(1e-3, 1, prior="log-uniform"),
        "clf__kernel":     Categorical(["rbf", "linear", "poly"]),
        "clf__gamma":      Real(1e-4, 1e-1, prior="log-uniform"),
    }
    bayes = BayesSearchCV(pipe, search_space, n_iter=20, cv=10,
                          scoring="accuracy", random_state=0, n_jobs=-1, verbose=2)
    bayes.fit(X, Y)
    print_top_n(bayes, "SVM")

def test_random_forest(X, Y):
    #=== Top 3 for RandomForest ===
    # 1. mean_accuracy=0.7148, params=OrderedDict({'clf__max_depth': 14, 'clf__min_samples_leaf': 3, 'clf__n_estimators': 378})
    # 2. mean_accuracy=0.7148, params=OrderedDict({'clf__max_depth': 13, 'clf__min_samples_leaf': 3, 'clf__n_estimators': 500})
    # 3. mean_accuracy=0.7094, params=OrderedDict({'clf__max_depth': 46, 'clf__min_samples_leaf': 2, 'clf__n_estimators': 403})

    pipe = Pipeline([
        ("clf", RandomForestClassifier(random_state=0, n_jobs=-1)),
    ])
    search_space = {
        "clf__n_estimators":   Integer(50, 750),
        "clf__max_depth":      Integer(2, 75),
        "clf__min_samples_leaf": Integer(1, 10),
    }
    bayes = BayesSearchCV(pipe, search_space, n_iter=20, cv=10,
                          scoring="accuracy", random_state=0, n_jobs=-1, verbose=2)
    bayes.fit(X, Y)
    print_top_n(bayes, "RandomForest")

class LGBMClassifierWrapper(lgb.LGBMClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_names_in = None

    @property
    def feature_names_in_(self):
        return self._feature_names_in

    @feature_names_in_.setter
    def feature_names_in_(self, value):
        self._feature_names_in = value

    @feature_names_in_.deleter
    def feature_names_in_(self):
        self._feature_names_in = None

def test_lightgbm(X, Y):
    pipe = Pipeline([
        ("pca", PCA()),
        ("clf", LGBMClassifierWrapper(
            objective="multiclass",
            random_state=0,
            n_jobs=-1,
            importance_type='gain',
            verbosity=-1,
            force_row_wise=True
        )),
    ])

    search_space = {
        # PCA parameters
        "pca__n_components": Integer(5, X.shape[1]),  # Search for optimal components

        # LightGBM parameters
        "clf__n_estimators": Integer(50, 500),
        "clf__learning_rate": Real(1e-3, 1e0, prior="log-uniform"),
        "clf__num_leaves": Integer(16, 256),
        "clf__max_depth": Integer(3, 20),
        "clf__reg_alpha": Real(0.0, 1.0),
        "clf__reg_lambda": Real(0.0, 1.0),
        "clf__min_child_samples": Integer(5, 50),
        "clf__subsample": Real(0.5, 1.0),
        "clf__colsample_bytree": Real(0.5, 1.0),
        "clf__class_weight": Categorical(['balanced', None])
    }

    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    bayes = BayesSearchCV(
        pipe,
        search_space,
        n_iter=20,
        cv=cv,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2
    )

    if hasattr(X, 'to_pandas'):
        X = X.to_pandas()

    bayes.fit(X, Y)

    print(f"\nLightGBM with PCA — Best CV score: {bayes.best_score_:.4f}")
    print("Best parameters:")
    for param, value in bayes.best_params_.items():
        print(f"  {param}: {value}")

    return bayes.best_estimator_

def test_knn(X, Y):
    #=== Top 3 for KNN ===
    # 1. mean_accuracy=0.5516, params=OrderedDict({'clf__n_neighbors': 20, 'clf__weights': 'distance', 'pca__n_components': 22})
    # 2. mean_accuracy=0.5453, params=OrderedDict({'clf__n_neighbors': 20, 'clf__weights': 'distance', 'pca__n_components': 5})
    # 3. mean_accuracy=0.5406, params=OrderedDict({'clf__n_neighbors': 4, 'clf__weights': 'distance', 'pca__n_components': 97})

    n_features = X.shape[1]
    pipe = Pipeline([
        ("pca", PCA()),
        ("clf", KNeighborsClassifier()),
    ])
    search_space = {
        "pca__n_components": Integer(5, min(200, n_features)),
        "clf__n_neighbors":  Integer(1, 20),
        "clf__weights":      Categorical(["uniform", "distance"]),
    }
    bayes = BayesSearchCV(pipe, search_space, n_iter=20, cv=10,
                          scoring="accuracy", random_state=0, n_jobs=-1, verbose=2)
    bayes.fit(X, Y)
    print_top_n(bayes, "KNN")

def test_xgboost(X, Y):
    #=== Top 3 for XGBoost ===
    # 1. mean_accuracy=0.7406, params=OrderedDict({'clf__colsample_bytree': 0.7643458710377254, 'clf__gamma': 0.28948921846765124, 'clf__learning_rate': 0.010637763908757623, 'clf__max_depth': 6, 'clf__n_estimators': 158, 'clf__subsample': 0.6710635126567753})
    # 2. mean_accuracy=0.7406, params=OrderedDict({'clf__colsample_bytree': 0.7127319006244467, 'clf__gamma': 0.0, 'clf__learning_rate': 0.07588466930755956, 'clf__max_depth': 3, 'clf__n_estimators': 407, 'clf__subsample': 0.5839260344792084})
    # 3. mean_accuracy=0.7398, params=OrderedDict({'clf__colsample_bytree': 0.5, 'clf__gamma': 0.0, 'clf__learning_rate': 0.001, 'clf__max_depth': 7, 'clf__n_estimators': 50, 'clf__subsample': 0.5887405613895313})

    n_features = X.shape[1]
    pipe = Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            random_state=0,
            n_jobs=-1
        ))
    ])
    search_space = {
        "clf__n_estimators":      Integer(50, 500),
        "clf__learning_rate":     Real(1e-3, 1e-0, prior="log-uniform"),
        "clf__max_depth":         Integer(3, 20),
        "clf__gamma":             Real(0, 5),
        "clf__subsample":         Real(0.5, 1.0),
        "clf__colsample_bytree":  Real(0.5, 1.0),
    }
    bayes = BayesSearchCV(
        pipe,
        search_space,
        n_iter=20,
        cv=10,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2,
    )
    bayes.fit(X, Y)
    print_top_n(bayes, "XGBoost")


from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def test_xgboost_improved(X, Y):
    """
    Improved XGBoost implementation with preprocessing, feature selection,
    early stopping, and additional hyperparameters.
    """
    #=== Top 3 for XGBoost (Improved) ===
    # 1. mean_accuracy=0.7539, params=OrderedDict({'clf__colsample_bytree': 0.5, 'clf__gamma': 0.2073497817611189, 'clf__learning_rate': 0.04454710528725294, 'clf__max_depth': 8, 'clf__min_child_weight': 3, 'clf__n_estimators': 186, 'clf__reg_alpha': 0.1558019331286106, 'clf__reg_lambda': 0.17373696059896945, 'clf__subsample': 0.6660284019795022, 'feature_selector__threshold': 'mean'})
    # 2. mean_accuracy=0.7516, params=OrderedDict({'clf__colsample_bytree': 0.5295355555496496, 'clf__gamma': 0.9848537657723972, 'clf__learning_rate': 0.031300533546632696, 'clf__max_depth': 8, 'clf__min_child_weight': 1, 'clf__n_estimators': 185, 'clf__reg_alpha': 0.6626711846877006, 'clf__reg_lambda': 1.0, 'clf__subsample': 0.5437417614677991, 'feature_selector__threshold': 'mean'})
    # 3. mean_accuracy=0.7508, params=OrderedDict({'clf__colsample_bytree': 0.5, 'clf__gamma': 1.0, 'clf__learning_rate': 0.012865904232082646, 'clf__max_depth': 8, 'clf__min_child_weight': 1, 'clf__n_estimators': 500, 'clf__reg_alpha': 1.0, 'clf__reg_lambda': 1.0, 'clf__subsample': 0.8, 'feature_selector__threshold': 'median'})

    # Convert to pandas if X is a polars DataFrame
    if hasattr(X, 'to_pandas'):
        X = X.to_pandas()

    # Create pipeline with optional preprocessing and feature selection
    pipe = Pipeline([
        ("scaler", StandardScaler()),  # Standardize features
        ("feature_selector", SelectFromModel(XGBClassifier(n_estimators=100, random_state=0), threshold="median")),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            random_state=0,
            n_jobs=-1
        ))
    ])

    # Define more focused search space based on previous best results
    search_space = {
        # Use 'feature_selector__threshold' to control feature selection aggressiveness
        "feature_selector__threshold": Categorical(['mean', 'median', '0.75*mean']),

        # Core parameters - narrowed based on previous good values
        "clf__n_estimators": Integer(100, 750),
        "clf__learning_rate": Real(0.001, 0.1, prior="log-uniform"),
        "clf__max_depth": Integer(3, 12),  # Narrower range based on best models

        # Regular parameters from before
        "clf__gamma": Real(0, 2),
        "clf__subsample": Real(0.3, 0.9),  # Narrower range based on best models
        "clf__colsample_bytree": Real(0.3, 0.9),  # Narrower range based on best models

        # Additional parameters to tune
        "clf__min_child_weight": Integer(1, 10),  # Controls overfitting
        "clf__reg_alpha": Real(0, 1),  # L1 regularization
        "clf__reg_lambda": Real(0, 1),  # L2 regularization
    }

    # Use stratified k-fold to maintain class distribution
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # Create BayesSearchCV with early stopping
    bayes = BayesSearchCV(
        pipe,
        search_space,
        n_iter=50,  # Slight increase to explore more combinations
        cv=cv,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2
    )

    bayes.fit(X, Y)

    # Print detailed results
    print_top_n(bayes, "XGBoost (Improved)")

    # Print feature importance information
    best_model = bayes.best_estimator_
    if hasattr(best_model[-1], 'feature_importances_'):
        selected_features = best_model[1].get_support()
        print(f"\nSelected {selected_features.sum()} out of {len(selected_features)} features")

        # Get feature names if available
        if hasattr(X, 'columns'):
            selected_feature_names = X.columns[selected_features]
            importances = best_model[-1].feature_importances_
            top_indices = importances.argsort()[-10:][::-1]

            print("\nTop 10 features by importance:")
            for i in top_indices:
                if i < len(selected_feature_names):
                    print(f"  {selected_feature_names[i]}: {importances[i]:.4f}")

    return bayes.best_estimator_

def nn_test():
    dataHandler = DataHandler()
    X, Y = dataHandler.get_train_data()

    pipe = Pipeline([
        ("net", NeuralNetClassifier(
            module=NNModel,
            module__input_size=X.shape[1],
            module__hidden_size=64,
            module__output_size=3,
            module__num_layers=2,  # match your manual setting
            module__dropout=0.3,  # as in manual
            max_epochs=200,  # give it enough epochs
            lr=3e-4,
            optimizer=torch.optim.Adam,
            batch_size=32,
            verbose=0,
            criterion=torch.nn.BCELoss,
            ordered_classification=True
        )),
    ])

    param_search = {
        "net__lr": Real(1e-5, 1e-3, prior='log-uniform'),
        "net__max_epochs": Integer(5, 300),
        "net__module__hidden_size": Categorical([4, 8, 16, 32, 64, 128]),
        "net__module__num_layers": Integer(1, 5),
        "net__module__dropout": Real(0.1, 0.7),
        "net__optimizer__weight_decay": Real(1e-4, 1e-1, prior='log-uniform'),
        "net__batch_size": Categorical([16, 32, 64, 128]),
    }

    opt = BayesSearchCV(
        pipe,
        param_search,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=3,
    )

    # make sure Y is int64 so skorch produces LongTensor targets
    X_np = X.to_numpy().astype(np.float32)
    Y_np = Y.to_numpy().ravel().astype(np.int64)

    opt.fit(X_np, Y_np)

    # 1) Best parameters found
    print("Best parameters:")
    for param, val in opt.best_params_.items():
        print(f"  {param}: {val}")

    # 2) Best cross-validated accuracy
    print(f"\nBest CV accuracy: {opt.best_score_:.4f}")

    # 3) Number of parameter settings evaluated
    print(f"\nTotal parameter settings evaluated: {len(opt.cv_results_['params'])}")

    # 4) Show the top 5 candidates by mean test accuracy
    results = pd.DataFrame(opt.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    top5 = results.head(5)[[
        "mean_test_score", "std_test_score", "params"
    ]].reset_index(drop=True)

    print("\nTop 5 parameter settings:")
    for i, row in top5.iterrows():
        print(f"\nRank {i + 1}:")
        print(f"  mean_test_accuracy: {row.mean_test_score:.4f}")
        print(f"  std_test_accuracy:  {row.std_test_score:.4f}")
        print(f"  params:            {row.params}")

def test_catboost(X, Y):
    #=== Top 3 for CatBoost ===
    # 1. mean_accuracy=0.7188, params=OrderedDict({'clf__bagging_temperature': 0.5309641649521474, 'clf__depth': 8, 'clf__grow_policy': 'Depthwise', 'clf__iterations': 14, 'clf__l2_leaf_reg': 52.178028219281096, 'clf__learning_rate': 0.8743167287239448, 'clf__random_strength': 1.90847538602676e-08})
    # 2. mean_accuracy=0.7023, params=OrderedDict({'clf__bagging_temperature': 0.36953697740052244, 'clf__depth': 6, 'clf__grow_policy': 'SymmetricTree', 'clf__iterations': 16, 'clf__l2_leaf_reg': 1.3811625072058136, 'clf__learning_rate': 0.16944524387171356, 'clf__random_strength': 0.02197916019583015})
    # 3. mean_accuracy=0.6945, params=OrderedDict({'clf__bagging_temperature': 0.3045464258650724, 'clf__depth': 7, 'clf__grow_policy': 'Depthwise', 'clf__iterations': 19, 'clf__l2_leaf_reg': 1.3597195504224944, 'clf__learning_rate': 1.0, 'clf__random_strength': 1e-09})

    """
    Tests CatBoost classifier with hyperparameter optimization using BayesSearchCV.

    Args:
        X: Feature matrix
        Y: Target vector
    """
    pipe = Pipeline([
        ("clf", cb.CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=0,
            thread_count=-1,
            allow_writing_files=False
        )),
    ])

    search_space = {
        "clf__iterations": Integer(2, 50),
        "clf__learning_rate": Real(1e-2, 1e-1, prior="log-uniform"),
        "clf__depth": Integer(3, 15),
        "clf__l2_leaf_reg": Real(0, 100, prior="log-uniform"),
        "clf__random_strength": Real(1e-9, 10, prior="log-uniform"),
        "clf__bagging_temperature": Real(0, 1),
        "clf__grow_policy": Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
    }

    bayes = BayesSearchCV(
        pipe,
        search_space,
        n_iter=20,
        cv=10,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2,
    )

    bayes.fit(X.to_numpy(), Y.to_numpy())
    print_top_n(bayes, "CatBoost")
    return bayes.best_estimator_

def generate_nn_submission(output_path: str = "./data/submissions/submission_nn.csv"):
    """
    Train NeuralNetClassifier on full training set and generate a submission CSV.
    """
    # 1) Load DataHandler
    dh = DataHandler()
    X_train, Y_train_df = dh.get_train_data()
    Y_train = Y_train_df["salary_category"].to_numpy().ravel().astype(int)

    # 2) Get test data and raw IDs
    raw_test = dh.test_data.clone()
    ids = raw_test["obs"].to_list() if "obs" in raw_test.columns else list(range(len(raw_test)))
    X_test = dh.get_test_data()

    # 3) Setup LabelEncoder (already fitted in DataHandler)
    le = dh.le

    # 4) Setup and train neural net pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", NeuralNetClassifier(
            module=NNModel,
            module__input_size=X_train.shape[1],
            module__hidden_size=128,
            module__output_size=3,
            module__num_layers=2,
            module__dropout=0.5,
            max_epochs=25,
            lr=6e-4,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=0.001,
            batch_size=64,
            train_split=None,
            verbose=0,
            criterion=torch.nn.CrossEntropyLoss,
        ))
    ])
    pipe.fit(X_train.to_numpy().astype(np.float32), Y_train)

    # 5) Print training stats
    from collections import Counter
    preds_train = pipe.predict(X_train.to_numpy().astype(np.float32))
    acc = (preds_train == Y_train).mean()
    dist_train = Counter(le.inverse_transform(preds_train))
    print(f"Training Accuracy: {acc:.4f}")
    print("Training prediction distribution:")
    for label, count in dist_train.items():
        print(f"  {label}: {count}")

    # 6) Submission generation
    sub_gen = SubmissionGenerator(pipeline=pipe, label_encoder=le, id_col="obs")
    sub_gen.generate(X_test=X_test, ids=ids, submission_csv_path=output_path)

    # 7) Test prediction diagnostics
    y_proba = pipe.predict_proba(X_test.to_numpy().astype(np.float32))
    y_pred = pipe.predict(X_test.to_numpy().astype(np.float32))
    dist_test = Counter(le.inverse_transform(y_pred))
    print("\nTest prediction distribution:")
    for label, count in dist_test.items():
        print(f"  {label}: {count}")

    print("\nTop-2 predicted classes for a few test samples:")
    for i in range(10):
        top2_idx = np.argsort(y_proba[i])[::-1][:2]
        top2_labels = le.inverse_transform(top2_idx)
        top2_probs = y_proba[i][top2_idx]
        print(f"Sample {i}: {top2_labels}, probs = {np.round(top2_probs, 3)}")

def generate_xgb_submission(
    output_path: str = "./data/submissions/submission_xgb.csv",
):
    """
    Train XGBoost on the full training set and generate a submission CSV.

    Parameters:
    -----------
    output_path: str
        Path where the submission CSV will be saved.
    """
    # 1) Load DataHandler and raw test IDs
    dh = DataHandler()
    # Extract raw test IDs from original test_data before preprocessing
    raw_test = dh.test_data.clone()
    if "obs" in raw_test.columns:
        ids = raw_test["obs"].to_list()
    else:
        # fallback to index if obs missing
        ids = list(range(len(raw_test)))

    # 2) Prepare training data
    X_train, Y_train_df = dh.get_train_data()
    Y_train = Y_train_df["salary_category"].to_numpy().ravel().astype(int)

    # 3) Prepare test features
    X_test = dh.get_test_data()

    # 4) Configure XGBoost with best parameters
    best_params = {
        'n_estimators': 158,
        'learning_rate': 0.010637763908757623,
        'max_depth': 6,
        'gamma': 0.28948921846765124,
        'subsample': 0.6710635126567753,
        'colsample_bytree': 0.7643458710377254,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'random_state': 0,
        'n_jobs': -1,
    }
    model = XGBClassifier(**best_params)

    # 5) Fit on full training set
    model.fit(
        X_train.to_numpy().astype(np.float32),
        Y_train
    )

    # -- Training prediction stats --
    y_train_pred = model.predict(X_train.to_numpy().astype(np.float32))
    train_acc = np.mean(y_train_pred == Y_train)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    unique_train, counts_train = np.unique(y_train_pred, return_counts=True)
    print("Training prediction distribution:")
    for cls, count in zip(unique_train, counts_train):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    # -- Test prediction stats --
    y_test_pred = model.predict(X_test.to_numpy().astype(np.float32))
    unique_test, counts_test = np.unique(y_test_pred, return_counts=True)
    print("\nTest prediction distribution:")
    for cls, count in zip(unique_test, counts_test):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    y_test_proba = model.predict_proba(X_test.to_numpy().astype(np.float32))
    top2 = np.argsort(-y_test_proba, axis=1)[:, :2]
    print("\nTop-2 predicted classes for a few test samples:")
    for i in range(10):
        print(f"Sample {i}: {dh.le.inverse_transform(top2[i])}, probs = {np.round(y_test_proba[i][top2[i]], 3)}")

    # 6) Generate submission using SubmissionGenerator
    sub_gen = SubmissionGenerator(
        pipeline=model,
        label_encoder=dh.le,
        id_col="obs"
    )
    sub_gen.generate(
        X_test=X_test,
        ids=ids,
        submission_csv_path=output_path,
    )


def generate_improved_xgb_submission(
        output_path: str = "./data/submissions/submission_improved_xgb.csv",
):
    """
    Train the improved XGBoost model with feature selection on the full training set
    and generate a submission CSV.

    Parameters:
    -----------
    output_path: str
        Path where the submission CSV will be saved.
    """
    # 1) Load DataHandler and raw test IDs
    dh = DataHandler()
    # Extract raw test IDs from original test_data before preprocessing
    raw_test = dh.test_data.clone()
    if "obs" in raw_test.columns:
        ids = raw_test["obs"].to_list()
    else:
        # fallback to index if obs missing
        ids = list(range(len(raw_test)))

    # 2) Prepare training data
    X_train, Y_train_df = dh.get_train_data()
    Y_train = Y_train_df["salary_category"].to_numpy().ravel().astype(int)

    # Convert to pandas if X is a polars DataFrame
    if hasattr(X_train, 'to_pandas'):
        X_train = X_train.to_pandas()

    # 3) Prepare test features
    X_test = dh.get_test_data()
    if hasattr(X_test, 'to_pandas'):
        X_test = X_test.to_pandas()

    # 4) Create pipeline with preprocessing and feature selection
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection based on best parameters
    selector = SelectFromModel(
        XGBClassifier(n_estimators=100, random_state=0),
        threshold="mean"  # From the best model
    )
    X_train_selected = selector.fit_transform(X_train_scaled, Y_train)
    X_test_selected = selector.transform(X_test_scaled)

    print(f"Selected {X_train_selected.shape[1]} out of {X_train.shape[1]} features")

    # 5) Configure XGBoost with best parameters from improved model
    best_params = {
        'n_estimators': 186,
        'learning_rate': 0.04454710528725294,
        'max_depth': 8,
        'gamma': 0.2073497817611189,
        'subsample': 0.6660284019795022,
        'colsample_bytree': 0.5,
        'min_child_weight': 3,
        'reg_alpha': 0.1558019331286106,
        'reg_lambda': 0.17373696059896945,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'random_state': 0,
        'n_jobs': -1,
    }
    model = XGBClassifier(**best_params)

    # 6) Fit on full preprocessed training set
    model.fit(
        X_train_selected.astype(np.float32),
        Y_train
    )

    # -- Training prediction stats --
    y_train_pred = model.predict(X_train_selected.astype(np.float32))
    train_acc = np.mean(y_train_pred == Y_train)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    unique_train, counts_train = np.unique(y_train_pred, return_counts=True)
    print("Training prediction distribution:")
    for cls, count in zip(unique_train, counts_train):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    # -- Test prediction stats --
    y_test_pred = model.predict(X_test_selected.astype(np.float32))
    unique_test, counts_test = np.unique(y_test_pred, return_counts=True)
    print("\nTest prediction distribution:")
    for cls, count in zip(unique_test, counts_test):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    y_test_proba = model.predict_proba(X_test_selected.astype(np.float32))
    top2 = np.argsort(-y_test_proba, axis=1)[:, :2]
    print("\nTop-2 predicted classes for a few test samples:")
    for i in range(10):
        print(f"Sample {i}: {dh.le.inverse_transform(top2[i])}, probs = {np.round(y_test_proba[i][top2[i]], 3)}")

    # Calculate confidence metrics
    confidence_scores = np.max(y_test_proba, axis=1)
    print(f"\nPrediction confidence stats:")
    print(f"  Mean confidence: {np.mean(confidence_scores):.4f}")
    print(f"  Median confidence: {np.median(confidence_scores):.4f}")
    print(f"  Min confidence: {np.min(confidence_scores):.4f}")
    print(f"  Max confidence: {np.max(confidence_scores):.4f}")

    # 7) Create a DataFrame for submission
    predictions = []
    for idx, pred_class in zip(ids, y_test_pred):
        label = dh.le.inverse_transform([pred_class])[0]
        predictions.append({
            "obs": idx,
            "salary_category": label
        })

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    return model, selector  # Return the model and selector for potential reuse


def generate_random_forest_submission(
        output_path: str = "./data/submissions/submission_rf_improved.csv",
):
    """
    Train the improved RandomForest model on the full training set
    and generate a submission CSV.
    """
    # 1) Load DataHandler and raw test IDs
    dh = DataHandler()
    raw_test = dh.test_data.clone()
    if "obs" in raw_test.columns:
        ids = raw_test["obs"].to_list()
    else:
        ids = list(range(len(raw_test)))

    # 2) Prepare training data
    X_train, Y_train_df = dh.get_train_data()
    Y_train = Y_train_df["salary_category"].to_numpy().ravel().astype(int)

    if hasattr(X_train, 'to_pandas'):
        X_train = X_train.to_pandas()

    # 3) Prepare test features
    X_test = dh.get_test_data()
    if hasattr(X_test, 'to_pandas'):
        X_test = X_test.to_pandas()

    # 4) Create preprocessing pipeline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=0),
        threshold="mean"  # Assuming this is optimal based on previous results
    )
    X_train_selected = selector.fit_transform(X_train_scaled, Y_train)
    X_test_selected = selector.transform(X_test_scaled)

    print(f"Selected {X_train_selected.shape[1]} out of {X_train.shape[1]} features")

    # 5) Configure RandomForest with best parameters (from your previous results)
    best_params = {
        'n_estimators': 378,
        'max_depth': 14,
        'min_samples_leaf': 3,
        'min_samples_split': 2,  # Default value
        'max_features': 'sqrt',  # Common best practice
        'bootstrap': True,
        'oob_score': True,
        'random_state': 0,
        'n_jobs': -1,
    }
    model = RandomForestClassifier(**best_params)

    # 6) Fit on full preprocessed training set
    model.fit(
        X_train_selected.astype(np.float32),
        Y_train
    )

    # Print OOB score (out-of-bag estimate of accuracy)
    print(f"\nOut-of-bag score: {model.oob_score_:.4f}")

    # Training prediction stats
    y_train_pred = model.predict(X_train_selected.astype(np.float32))
    train_acc = np.mean(y_train_pred == Y_train)
    print(f"Training Accuracy: {train_acc:.4f}")

    unique_train, counts_train = np.unique(y_train_pred, return_counts=True)
    print("Training prediction distribution:")
    for cls, count in zip(unique_train, counts_train):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    # Test prediction stats
    y_test_pred = model.predict(X_test_selected.astype(np.float32))
    unique_test, counts_test = np.unique(y_test_pred, return_counts=True)
    print("\nTest prediction distribution:")
    for cls, count in zip(unique_test, counts_test):
        label_name = dh.le.inverse_transform([cls])[0]
        print(f"  {label_name}: {count}")

    y_test_proba = model.predict_proba(X_test_selected.astype(np.float32))
    top2 = np.argsort(-y_test_proba, axis=1)[:, :2]
    print("\nTop-2 predicted classes for a few test samples:")
    for i in range(10):
        print(f"Sample {i}: {dh.le.inverse_transform(top2[i])}, probs = {np.round(y_test_proba[i][top2[i]], 3)}")

    # Calculate confidence metrics
    confidence_scores = np.max(y_test_proba, axis=1)
    print(f"\nPrediction confidence stats:")
    print(f"  Mean confidence: {np.mean(confidence_scores):.4f}")
    print(f"  Median confidence: {np.median(confidence_scores):.4f}")
    print(f"  Min confidence: {np.min(confidence_scores):.4f}")
    print(f"  Max confidence: {np.max(confidence_scores):.4f}")

    # 7) Create submission DataFrame
    predictions = []
    for idx, pred_class in zip(ids, y_test_pred):
        label = dh.le.inverse_transform([pred_class])[0]
        predictions.append({
            "obs": idx,
            "salary_category": label
        })

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    return model, selector


from sklearn.ensemble import IsolationForest


def identify_isolation_forest_outliers(X, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=0)
    outlier_labels = iso_forest.fit_predict(X)
    outlier_indices = np.where(outlier_labels == -1)[0]

    print(f"Found {len(outlier_indices)} outliers using Isolation Forest")
    return outlier_indices


def detect_supervised_outliers(X, Y, contamination=0.05):
    """
    Detect outliers considering both features and target variable.
    """
    # 1. Model-based detection
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Get cross-validated predictions to avoid data leakage
    y_pred_proba = np.zeros((len(Y), len(np.unique(Y))))
    for train_idx, val_idx in cv.split(X, Y):
        model.fit(X.iloc[train_idx], Y.iloc[train_idx])
        y_pred_proba[val_idx] = model.predict_proba(X.iloc[val_idx])

    # Find prediction errors
    y_pred = np.argmax(y_pred_proba, axis=1)
    prediction_errors = (y_pred != Y)

    # Find confident errors (high confidence in wrong prediction)
    confidences = np.max(y_pred_proba, axis=1)
    confident_errors = (confidences > 0.8) & prediction_errors

    # 2. Feature-target relationship analysis
    # Combine features with target in isolation forest
    X_with_target = np.column_stack([X, Y.values.reshape(-1, 1)])
    iso = IsolationForest(contamination=contamination, random_state=0)
    outlier_scores = iso.fit_predict(X_with_target)

    # Combine both methods
    combined_outliers = confident_errors | (outlier_scores == -1)
    outlier_indices = np.where(combined_outliers)[0]

    print(f"Found {sum(confident_errors)} confident misclassifications")
    print(f"Found {sum(outlier_scores == -1)} isolation forest outliers")
    print(f"Found {len(outlier_indices)} total outliers after combining methods")

    return outlier_indices

def nn_t_t(
        test_size: float = 0.2,
        random_state: int = 0,
        batch_size: int = 32
) -> nn.Module:
    # 1. Load data
    dataHandler = DataHandler(fill_data=True)
    X, Y = dataHandler.X.to_numpy(), dataHandler.Y

    # 2. Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Train/eval split
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y
    )

    # 4. Convert to PyTorch tensors and create DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 5. Initialize model, loss, optimizer
    model = NNModel(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=len(np.unique(Y)),
        num_layers=2,
        dropout=0.3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 6. Training loop with batching
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_ds)

        if epoch % 10 == 0 or epoch == 1:
            # Evaluate
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_loss += loss.item() * xb.size(0)
                    pred = outputs.argmax(dim=1)
                    correct += (pred == yb).sum().item()
            avg_val_loss = val_loss / len(val_ds)
            val_acc = correct / len(val_ds)

            print(
                f"Epoch {epoch:4d}/{num_epochs} "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val Acc: {val_acc:.4f}"
            )

    return model

def main():
    dh = DataHandler()
    X, Y = dh.get_train_data()
    Y = Y["salary_category"]
    detect_supervised_outliers(X.to_pandas(), Y.to_pandas())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

