
import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    OUTPUT_DIR,
    RANDOM_STATE,
    N_SPLITS,
    EXPECTED_CLASSES,
    RandomForestClassifier,
    SVC,
    KNeighborsClassifier,
    StandardScaler,
    SplineTransformer,
    StratifiedKFold,
    KMeans,
    silhouette_score,
    classification_report,
)
from data_simulation import LongitudinalSimulator
from emlc_classifier import EMLCClassifier
import analysis


# Standardize time and response using training-fold means and SDs.
class LongitudinalStandardizer:

    def __init__(self, columns=("time", "y")):
        self.columns = columns
        self.means_ = {}
        self.stds_ = {}

    def fit(self, df):
        for col in self.columns:
            self.means_[col] = float(df[col].mean())
            sd = float(df[col].std(ddof=0))
            self.stds_[col] = sd if np.isfinite(sd) and sd > 0 else 1.0
        return self

    def transform(self, df):
        out = df.copy()
        for col in self.columns:
            out[col] = (out[col] - self.means_[col]) / self.stds_[col]
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# Subject summaries (mean, SD, min, max, slope) for random forest and SVM.
def extract_summary_features(df):
    rows = []
    for sub_id, sub_df in df.groupby("id"):
        sub_df = sub_df.sort_values("time")
        y = sub_df["y"].to_numpy(dtype=float)
        t = sub_df["time"].to_numpy(dtype=float)
        slope = float(np.polyfit(t, y, 1)[0]) if len(y) > 1 and np.std(t) > 0 else 0.0
        rows.append({
            "id": sub_id,
            "mean_y": float(np.mean(y)),
            "std_y": float(np.std(y)),
            "min_y": float(np.min(y)),
            "max_y": float(np.max(y)),
            "slope": slope,
            "true_class": int(sub_df["class"].iloc[0]),
        })
    return pd.DataFrame(rows)


# Interpolate each subject onto a common grid, plus adjacent differences.
def extract_functional_grid_features(df, grid):
    rows = []
    for sub_id, sub_df in df.groupby("id"):
        sub_df = sub_df.sort_values("time")
        t = sub_df["time"].to_numpy(dtype=float)
        y = sub_df["y"].to_numpy(dtype=float)
        unique_t, unique_idx = np.unique(t, return_index=True)
        unique_y = y[unique_idx]
        if len(unique_t) == 1:
            grid_y = np.repeat(unique_y[0], len(grid))
        else:
            grid_y = np.interp(grid, unique_t, unique_y, left=unique_y[0], right=unique_y[-1])
        diffs = np.diff(grid_y)
        row = {"id": sub_id, "true_class": int(sub_df["class"].iloc[0])}
        row.update({f"grid_y_{i}": float(v) for i, v in enumerate(grid_y)})
        row.update({f"grid_diff_{i}": float(v) for i, v in enumerate(diffs)})
        rows.append(row)
    return pd.DataFrame(rows)


# Build three trajectory phenotypes by k-means on dense-history quadratic descriptors.
def create_dense_parkinsons_phenotypes(csv_path):
    df = pd.read_csv(csv_path)[["subject#", "test_time", "motor_UPDRS"]].dropna()
    df.columns = ["id", "time", "y"]

    coefs, valid_ids = [], []
    for sub_id, sub_df in df.groupby("id"):
        sub_df = sub_df.sort_values("time")
        if len(sub_df) >= 3 and sub_df["time"].std() > 0:
            p = np.polyfit(sub_df["time"], sub_df["y"], 2)
            total_change = sub_df["y"].iloc[-1] - sub_df["y"].iloc[0]
            coefs.append([p[0], p[1], p[2], total_change])
            valid_ids.append(sub_id)
    if len(valid_ids) < 3:
        raise ValueError("Not enough valid Parkinson's subjects for three trajectory phenotypes.")

    coef_df = pd.DataFrame(coefs, columns=["quadratic", "linear", "intercept", "total_change"])
    coef_df.insert(0, "id", valid_ids)
    X = StandardScaler().fit_transform(coef_df[["quadratic", "linear", "intercept", "total_change"]])
    clusters = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=50).fit_predict(X)
    coef_df["cluster"] = clusters

    ordered = coef_df.groupby("cluster")["total_change"].mean().sort_values().index.tolist()
    relabel = {cluster: new_label for new_label, cluster in enumerate(ordered)}
    class_map = {row.id: relabel[row.cluster] for row in coef_df.itertuples(index=False)}

    dense_df = df[df["id"].isin(valid_ids)].copy()
    dense_df["class"] = dense_df["id"].map(class_map).astype(int)
    coef_df["class"] = coef_df["cluster"].map(relabel).astype(int)
    return dense_df, coef_df


# Keep about 10% of each dense history at evenly spaced early, middle, and late points.
def make_sparse_recovery_data(dense_df, retained_fraction=0.10):
    rng = np.random.default_rng(RANDOM_STATE)
    sparse = []
    rows = []
    for sub_id, sub_df in dense_df.groupby("id"):
        sub_df = sub_df.sort_values("time").reset_index(drop=True)
        dense_n = len(sub_df)
        sparse_n = max(4, int(round(dense_n * retained_fraction)))
        sparse_n = min(sparse_n, dense_n)
        if sparse_n == dense_n:
            sampled = sub_df
        else:
            idx = np.unique(np.round(np.linspace(0, dense_n - 1, sparse_n)).astype(int))
            while len(idx) < sparse_n:
                candidates = [i for i in range(dense_n) if i not in idx]
                idx = np.sort(np.append(idx, rng.choice(candidates)))
            sampled = sub_df.iloc[idx].sort_values("time")
        sparse.append(sampled)
        rows.append({
            "id": sub_id,
            "dense_n": dense_n,
            "sparse_n": len(sampled),
            "hidden_fraction": 1.0 - len(sampled) / dense_n,
            "class": int(sub_df["class"].iloc[0]),
        })
    return pd.concat(sparse, ignore_index=True), pd.DataFrame(rows)


# Inertia, silhouette, and minimum cluster size for k = 2 to 6.
def kmeans_diagnostics(coef_df, max_k=6):
    X = StandardScaler().fit_transform(coef_df[["quadratic", "linear", "intercept", "total_change"]])
    rows = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit(X)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)),
            "min_cluster_size": int(pd.Series(km.labels_).value_counts().min()),
        })
    return pd.DataFrame(rows)


# AIC/BIC comparison of linear, quadratic, cubic, and B-spline mean models.
def polynomial_model_selection(dense_df):
    rows = []

    def score_model(name, feature_builder, p_per_class):
        rss = 0.0
        n_obs = 0
        p_total = 0
        for _, sub_df in dense_df.groupby("class"):
            t = sub_df["time"].to_numpy(dtype=float)
            y = sub_df["y"].to_numpy(dtype=float)
            X = feature_builder(t)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - X @ beta
            rss += float(np.dot(residual, residual))
            n_obs += len(y)
            p_total += p_per_class(X)
        sigma2 = max(rss / n_obs, 1e-12)
        aic = n_obs * np.log(sigma2) + 2 * p_total
        bic = n_obs * np.log(sigma2) + p_total * np.log(n_obs)
        rows.append({"model": name, "parameters": p_total, "RSS": rss, "AIC": aic, "BIC": bic})

    for degree in (1, 2, 3):
        score_model(
            f"Polynomial degree {degree}",
            lambda t, degree=degree: np.vander(t, degree + 1, increasing=True),
            lambda X: X.shape[1],
        )

    spline = SplineTransformer(n_knots=5, degree=3, include_bias=True)
    spline.fit(dense_df[["time"]])
    score_model(
        "Cubic B-spline",
        lambda t, spline=spline: spline.transform(pd.DataFrame({"time": t})),
        lambda X: X.shape[1],
    )
    out = pd.DataFrame(rows)
    out["delta_BIC"] = out["BIC"] - out["BIC"].min()
    out["delta_AIC"] = out["AIC"] - out["AIC"].min()
    return out


def engineer_parkinsons_data(csv_path, retained_fraction=0.10):
    dense_df, coef_df = create_dense_parkinsons_phenotypes(csv_path)
    sparse_df, missing_df = make_sparse_recovery_data(dense_df, retained_fraction=retained_fraction)
    min_class_size = sparse_df.groupby("class")["id"].nunique().min()
    if min_class_size < N_SPLITS:
        raise ValueError(f"Smallest phenotype class has {min_class_size} subjects; need {N_SPLITS} for CV.")
    return sparse_df, dense_df, coef_df, missing_df


def summarize_simulation_data(df):
    class_counts = df.groupby("class")["id"].nunique().sort_index()
    visits = df.groupby("id").size()
    return pd.DataFrame([
        {
            "Experiment": "Simulation",
            "Subjects": int(df["id"].nunique()),
            "Class 0": int(class_counts.get(0, 0)),
            "Class 1": int(class_counts.get(1, 0)),
            "Class 2": int(class_counts.get(2, 0)),
            "Observations": int(len(df)),
            "Mean obs/subject": float(visits.mean()),
            "Min obs/subject": int(visits.min()),
            "Max obs/subject": int(visits.max()),
            "Hidden fraction": "NA",
        }
    ])


def summarize_parkinsons_recovery(dense_df, sparse_df, missing_df):
    class_counts = sparse_df.groupby("class")["id"].nunique().sort_index()
    dense_visits = dense_df.groupby("id").size()
    sparse_visits = sparse_df.groupby("id").size()
    return pd.DataFrame([
        {
            "Experiment": "Parkinson dense histories",
            "Subjects": int(dense_df["id"].nunique()),
            "Class 0": int(class_counts.get(0, 0)),
            "Class 1": int(class_counts.get(1, 0)),
            "Class 2": int(class_counts.get(2, 0)),
            "Observations": int(len(dense_df)),
            "Mean obs/subject": float(dense_visits.mean()),
            "Min obs/subject": int(dense_visits.min()),
            "Max obs/subject": int(dense_visits.max()),
            "Hidden fraction": "0.000",
        },
        {
            "Experiment": "Parkinson sparse input",
            "Subjects": int(sparse_df["id"].nunique()),
            "Class 0": int(class_counts.get(0, 0)),
            "Class 1": int(class_counts.get(1, 0)),
            "Class 2": int(class_counts.get(2, 0)),
            "Observations": int(len(sparse_df)),
            "Mean obs/subject": float(sparse_visits.mean()),
            "Min obs/subject": int(sparse_visits.min()),
            "Max obs/subject": int(sparse_visits.max()),
            "Hidden fraction": f"{float(missing_df['hidden_fraction'].mean()):.3f}",
        }
    ])


def export_design_summary(summary_df, basename):
    csv_path = os.path.join(OUTPUT_DIR, f"{basename}.csv")
    summary_df.to_csv(csv_path, index=False)
    return csv_path

def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _extract_model_probs(model, X_test, y_test, expected_classes=EXPECTED_CLASSES):
    pred = model.predict(X_test).astype(int)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    else:
        raw = model.decision_function(X_test)
        raw = np.atleast_2d(raw)
        probs = _softmax(raw)
    out = pd.DataFrame({"true_class": np.asarray(y_test, dtype=int), "pred_class": pred})
    for idx, cls in enumerate(model.classes_):
        out[f"prob_{int(cls)}"] = probs[:, idx]
    for cls in expected_classes:
        if f"prob_{cls}" not in out.columns:
            out[f"prob_{cls}"] = 0.0
    prob_cols = [f"prob_{cls}" for cls in expected_classes]
    row_sums = out[prob_cols].sum(axis=1).replace(0, 1.0)
    out[prob_cols] = out[prob_cols].div(row_sums, axis=0)
    return out


# Fit random forest and RBF-SVM on standardized subject summaries.
def _fit_feature_baselines(train_norm, test_norm):
    train_sum = extract_summary_features(train_norm)
    test_sum = extract_summary_features(test_norm)
    feature_cols = ["mean_y", "std_y", "min_y", "max_y", "slope"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_sum[feature_cols])
    X_test = scaler.transform(test_sum[feature_cols])
    y_train = train_sum["true_class"].astype(int).to_numpy()
    y_test = test_sum["true_class"].astype(int).to_numpy()

    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    svm = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        probability=False,
        random_state=RANDOM_STATE,
    )
    svm.fit(X_train, y_train)
    return _extract_model_probs(rf, X_test, y_test), _extract_model_probs(svm, X_test, y_test)


def _fit_functional_knn_baseline(train_norm, test_norm):
    grid = np.linspace(train_norm["time"].quantile(0.02), train_norm["time"].quantile(0.98), 10)
    train_fun = extract_functional_grid_features(train_norm, grid)
    test_fun = extract_functional_grid_features(test_norm, grid)
    feature_cols = [c for c in train_fun.columns if c not in ("id", "true_class")]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_fun[feature_cols])
    X_test = scaler.transform(test_fun[feature_cols])
    y_train = train_fun["true_class"].astype(int).to_numpy()
    y_test = test_fun["true_class"].astype(int).to_numpy()

    fknn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    fknn.fit(X_train, y_train)
    return _extract_model_probs(fknn, X_test, y_test)


# Subject-level stratified 5-fold CV comparing the proposed MLC with the baselines.
def evaluate_dataset(full_data, experiment_name, output_pdf_name, marker_label="Longitudinal marker"):
    print(f"\n{'=' * 72}\nExecuting {experiment_name} ({N_SPLITS}-fold subject-level CV)\n{'=' * 72}", flush=True)
    subject_data = full_data.groupby("id", as_index=False)["class"].first()
    unique_subjects = subject_data["id"].to_numpy()
    subject_classes = subject_data["class"].astype(int).to_numpy()
    class_counts = pd.Series(subject_classes).value_counts().sort_index().to_dict()
    print(f"Class counts by subject: {class_counts}", flush=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    preds_base, preds_enh, preds_rf, preds_svm, preds_fknn = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_classes), 1):
        print(f"  - Fold {fold}/{N_SPLITS}", flush=True)
        train_subs = unique_subjects[train_idx]
        test_subs = unique_subjects[test_idx]
        train_df = full_data[full_data["id"].isin(train_subs)].copy()
        test_df = full_data[full_data["id"].isin(test_subs)].copy()
        normalizer = LongitudinalStandardizer(columns=("time", "y"))
        train_norm = normalizer.fit_transform(train_df)
        test_norm = normalizer.transform(test_df)

        base_mlc = EMLCClassifier(degree=1, prior="uniform", random_effects="intercept", covariance_shrinkage=0.0)
        enh_mlc = EMLCClassifier(degree=2, prior="uniform", random_effects="intercept_slope", covariance_shrinkage=0.0)
        preds_base.append(base_mlc.fit(train_norm).predict(test_norm))
        preds_enh.append(enh_mlc.fit(train_norm).predict(test_norm))
        rf_preds, svm_preds = _fit_feature_baselines(train_norm, test_norm)
        preds_rf.append(rf_preds)
        preds_svm.append(svm_preds)
        preds_fknn.append(_fit_functional_knn_baseline(train_norm, test_norm))

    results = {
        "Baseline MLC (linear)": pd.concat(preds_base, ignore_index=True),
        "Multiclass MLC (quadratic)": pd.concat(preds_enh, ignore_index=True),
        "Functional kNN": pd.concat(preds_fknn, ignore_index=True),
        "Random forest": pd.concat(preds_rf, ignore_index=True),
        "Support vector machine": pd.concat(preds_svm, ignore_index=True),
    }
    metrics_df = analysis.compute_metrics(results)
    print("\nMetric summary:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    metrics_csv = os.path.join(OUTPUT_DIR, output_pdf_name.replace(".pdf", "_metrics.csv"))
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[{experiment_name}] Metrics saved to {metrics_csv}")
    return results, metrics_df, metrics_csv


# Train/validation/test diagnostic for the proposed MLC within each outer fold.
def evaluate_train_validation_test(full_data):
    subject_data = full_data.groupby("id", as_index=False)["class"].first()
    unique_subjects = subject_data["id"].to_numpy()
    subject_classes = subject_data["class"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_preds, val_preds, test_preds = [], [], []
    split_rows = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_classes), 1):
        trainval_subs = unique_subjects[trainval_idx]
        test_subs = unique_subjects[test_idx]
        trainval_y = subject_classes[trainval_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE + fold)
        train_rel, val_rel = next(splitter.split(trainval_subs, trainval_y))
        train_subs = trainval_subs[train_rel]
        val_subs = trainval_subs[val_rel]

        train_df = full_data[full_data["id"].isin(train_subs)].copy()
        val_df = full_data[full_data["id"].isin(val_subs)].copy()
        inner_norm = LongitudinalStandardizer(columns=("time", "y"))
        train_norm = inner_norm.fit_transform(train_df)
        val_norm = inner_norm.transform(val_df)
        inner_model = EMLCClassifier(degree=2, prior="uniform", random_effects="intercept_slope", covariance_shrinkage=0.0)
        val_preds.append(inner_model.fit(train_norm).predict(val_norm))

        trainval_df = full_data[full_data["id"].isin(trainval_subs)].copy()
        test_df = full_data[full_data["id"].isin(test_subs)].copy()
        final_norm = LongitudinalStandardizer(columns=("time", "y"))
        trainval_norm = final_norm.fit_transform(trainval_df)
        test_norm = final_norm.transform(test_df)
        final_model = EMLCClassifier(degree=2, prior="uniform", random_effects="intercept_slope", covariance_shrinkage=0.0)
        final_model.fit(trainval_norm)
        train_preds.append(final_model.predict(trainval_norm))
        test_preds.append(final_model.predict(test_norm))
        split_rows.append({
            "Fold": fold,
            "Outer training subjects (refit)": len(trainval_subs),
            "Inner training subjects": len(train_subs),
            "Validation subjects": len(val_subs),
            "Test subjects": len(test_subs),
        })

    metrics = analysis.compute_metrics({
        "Training": pd.concat(train_preds, ignore_index=True),
        "Validation": pd.concat(val_preds, ignore_index=True),
        "Test": pd.concat(test_preds, ignore_index=True),
    })
    metrics_path = os.path.join(OUTPUT_DIR, "train_validation_test_metrics.csv")
    splits_path = os.path.join(OUTPUT_DIR, "train_validation_test_splits.csv")
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(split_rows).to_csv(splits_path, index=False)
    print(f"[train/validation/test] Metrics saved to {metrics_path}")
    return metrics, pd.DataFrame(split_rows)


def export_diagnostics(k_df, model_df, missing_df):
    k_csv = os.path.join(OUTPUT_DIR, "kmeans_diagnostics.csv")
    model_csv = os.path.join(OUTPUT_DIR, "trajectory_model_selection.csv")
    missing_csv = os.path.join(OUTPUT_DIR, "sparse_recovery_missingness.csv")
    k_df.to_csv(k_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)
    return k_csv, model_csv, missing_csv

SIMULATION_SAMPLE_SIZES = [50, 100, 300, 500, 1000]
SIMULATION_REPLICATES = 1

def class_counts_for_total(total_n):
    ratios = np.asarray([1.0, 2.0, 3.0]) / 6.0
    raw = ratios * int(total_n)
    counts = np.floor(raw).astype(int)
    remainder = int(total_n - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {0: int(counts[0]), 1: int(counts[1]), 2: int(counts[2])}


def evaluate_proposed_mlc_only(full_data, experiment_name, predictions_path=None):
    print(f"\n{'=' * 72}\nExecuting {experiment_name}: proposed MLC only\n{'=' * 72}", flush=True)
    subject_data = full_data.groupby("id", as_index=False)["class"].first()
    unique_subjects = subject_data["id"].to_numpy()
    subject_classes = subject_data["class"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    preds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_classes), 1):
        print(f"  - Fold {fold}/{N_SPLITS}", flush=True)
        train_subs = unique_subjects[train_idx]
        test_subs = unique_subjects[test_idx]
        train_df = full_data[full_data["id"].isin(train_subs)].copy()
        test_df = full_data[full_data["id"].isin(test_subs)].copy()
        normalizer = LongitudinalStandardizer(columns=("time", "y"))
        train_norm = normalizer.fit_transform(train_df)
        test_norm = normalizer.transform(test_df)
        model = EMLCClassifier(degree=2, prior="uniform", random_effects="intercept_slope", covariance_shrinkage=0.0)
        preds.append(model.fit(train_norm).predict(test_norm))

    results = {"Multiclass MLC (quadratic)": pd.concat(preds, ignore_index=True)}
    if predictions_path is not None:
        results["Multiclass MLC (quadratic)"].to_csv(predictions_path, index=False)
    metrics = analysis.compute_metrics(results)
    return metrics.iloc[0]


# Proposed MLC across simulated sample sizes N = 50 to 1000.
def run_simulation_sample_size_study(sample_sizes=SIMULATION_SAMPLE_SIZES, n_replicates=SIMULATION_REPLICATES):
    design_rows = []
    result_rows = []
    replicate_rows = []

    for total_n in sample_sizes:
        class_counts = class_counts_for_total(total_n)
        obs_counts = []
        mean_visits = []
        metric_reps = []

        for rep in range(1, n_replicates + 1):
            sim_data = LongitudinalSimulator(
                n_samples_per_class=class_counts,
                random_state=RANDOM_STATE + int(total_n) * 100 + rep,
            ).simulate()
            visits = sim_data.groupby("id").size()
            obs_counts.append(int(len(sim_data)))
            mean_visits.append(float(visits.mean()))
            metric_row = evaluate_proposed_mlc_only(
                sim_data,
                f"Synthetic Simulation Data (N={total_n}, replicate={rep})",
                predictions_path=os.path.join(OUTPUT_DIR, f"simulation_n{total_n}_rep{rep}_predictions.csv"),
            )
            rep_dict = {
                "Subjects": int(total_n),
                "Replicate": int(rep),
                "Accuracy": float(metric_row["Accuracy"]),
                "Macro_F1": float(metric_row["Macro_F1"]),
                "MCC": float(metric_row["MCC"]),
                "Macro_AUC": float(metric_row["Macro_AUC"]),
                **{key: float(metric_row[key]) for key in ["Brier", "Macro_Recall", "Macro_Specificity"]},
            }
            replicate_rows.append(rep_dict)
            metric_reps.append(rep_dict)

        design_rows.append({
            "Subjects": int(total_n),
            "Class 0": int(class_counts.get(0, 0)),
            "Class 1": int(class_counts.get(1, 0)),
            "Class 2": int(class_counts.get(2, 0)),
            "Mean observations": float(np.mean(obs_counts)),
            "Mean obs/subject": float(np.mean(mean_visits)),
        })
        metric_df = pd.DataFrame(metric_reps)
        result_rows.append({
            "Subjects": int(total_n),
            "Accuracy": float(metric_df["Accuracy"].mean()),
            "Macro_F1": float(metric_df["Macro_F1"].mean()),
            "MCC": float(metric_df["MCC"].mean()),
            "Macro_AUC": float(metric_df["Macro_AUC"].mean()),
            **{key: float(metric_df[key].mean()) for key in ["Brier", "Macro_Recall", "Macro_Specificity"]},
        })

    design_df = pd.DataFrame(design_rows)
    result_df = pd.DataFrame(result_rows)
    replicate_df = pd.DataFrame(replicate_rows)

    design_csv = os.path.join(OUTPUT_DIR, "simulation_sample_design.csv")
    result_csv = os.path.join(OUTPUT_DIR, "simulation_sample_size_results.csv")
    replicate_csv = os.path.join(OUTPUT_DIR, "simulation_sample_size_replicates.csv")

    design_df.to_csv(design_csv, index=False)
    result_df.to_csv(result_csv, index=False)
    replicate_df.to_csv(replicate_csv, index=False)
    print(f"[simulation sample sizes] Design CSV saved to {design_csv}")
    print(f"[simulation sample sizes] Results CSV saved to {result_csv}")
    return design_df, result_df


def export_manuscript_figures():
    sim_csv = os.path.join(OUTPUT_DIR, "simulation_summary_metrics.csv")
    real_csv = os.path.join(OUTPUT_DIR, "real_data_summary_metrics.csv")
    k_csv = os.path.join(OUTPUT_DIR, "kmeans_diagnostics.csv")
    model_csv = os.path.join(OUTPUT_DIR, "trajectory_model_selection.csv")

    if os.path.exists(sim_csv) and os.path.exists(real_csv):
        sim_metrics = pd.read_csv(sim_csv)
        real_metrics = pd.read_csv(real_csv)
        pdf_path = os.path.join(OUTPUT_DIR, "performance_summary.pdf")
        with PdfPages(pdf_path) as pdf:
            fig = analysis.plot_metric_gain(sim_metrics, real_metrics)
            pdf.savefig(fig)
            plt = __import__("matplotlib.pyplot").pyplot
            plt.close(fig)
        print(f"[summary] Performance figure saved to {pdf_path}")

    write_run_summary()



def export_reproducibility_settings():
    rows = [
        {"Component": "Global", "Setting": "RANDOM_STATE", "Value": "42"},
        {"Component": "Outer validation", "Setting": "Split", "Value": "StratifiedKFold, n_splits=5, shuffle=True, random_state=42"},
        {"Component": "Train-validation-test diagnostic", "Setting": "Split", "Value": "Outer 80/20 stratified fold; 20% of training fold held out by StratifiedShuffleSplit with random_state=42+fold"},
        {"Component": "Simulation data", "Setting": "Sample sizes", "Value": "50, 100, 300, 500, 1000"},
        {"Component": "Simulation data", "Setting": "Class ratio", "Value": "1:2:3, rounded to integer counts"},
        {"Component": "Simulation data", "Setting": "Visit design", "Value": "time_range=(0,2), n_obs_range=(4,8), missing_rate=0.10"},
        {"Component": "Simulation data", "Setting": "Noise/random effects", "Value": "noise_std=0.45, random_intercept_sd=0.70, random_slope_sd=0.25"},
        {"Component": "Simulation data", "Setting": "Sample-size seeds", "Value": "42 + 100*N + replicate; replicate=1 in manuscript"},
        {"Component": "Proposed MLC", "Setting": "Fixed hyperparameters", "Value": "degree=2, random_effects=intercept_slope, prior=uniform, ridge=1e-4, covariance_shrinkage=0.0"},
        {"Component": "Linear MLC baseline", "Setting": "Fixed hyperparameters", "Value": "degree=1, random_effects=intercept, prior=uniform, ridge=1e-4, covariance_shrinkage=0.0"},
        {"Component": "Functional kNN", "Setting": "Fixed hyperparameters", "Value": "n_neighbors=5, weights=distance, interpolation_grid_points=10"},
        {"Component": "Random forest", "Setting": "Fixed hyperparameters", "Value": "n_estimators=160, min_samples_leaf=2, class_weight=balanced_subsample, random_state=42"},
        {"Component": "SVM", "Setting": "Fixed hyperparameters", "Value": "kernel=rbf, C=2.0, gamma=scale, class_weight=balanced, decision_function_shape=ovr"},
        {"Component": "Parkinson phenotype construction", "Setting": "Fixed hyperparameters", "Value": "retained_fraction=0.10, kmeans_k=3, n_init=50, random_state=42"},
        {"Component": "Hyperparameter tuning", "Setting": "Policy", "Value": "No grid search or test-fold tuning; all settings fixed before evaluation"},
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "reproducibility_settings.csv")
    txt_path = os.path.join(OUTPUT_DIR, "reproducibility_settings.txt")
    df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['Component']} - {row['Setting']}: {row['Value']}\n")
    return csv_path, txt_path

def write_run_summary():
    path = os.path.join(OUTPUT_DIR, "run_summary.txt")
    export_reproducibility_settings()
    lines = [
        "Multiclass MLC output folder",
        "Includes manuscript-relevant PDFs, CSV tables, and this text file.",
        "Key outputs: performance_summary.pdf, simulation_sample_size_results.csv, simulation_summary_metrics.csv, real_data_summary_metrics.csv, train_validation_test_metrics.csv, kmeans_diagnostics.csv, trajectory_model_selection.csv.",
        "Reproducibility details: reproducibility_settings.csv and reproducibility_settings.txt document random seeds, split strategy, fixed hyperparameters, and the no test-fold tuning policy.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def clean_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        return
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

# Entry point: run the simulation, Parkinson's, and diagnostic analyses.
def main(experiment="all"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if experiment == "all":
        clean_output_dir()
        script = os.path.abspath(__file__)
        subprocess.run([sys.executable, script, "--experiment", "sim"], check=True)
        subprocess.run([sys.executable, script, "--experiment", "real"], check=True)
        export_manuscript_figures()
        subprocess.run([sys.executable, os.path.join(os.path.dirname(script), "reviewer_analysis.py")], check=True)
        return

    if experiment == "sim":
        run_simulation_sample_size_study()

        sim_data = LongitudinalSimulator(random_state=RANDOM_STATE).simulate()
        export_design_summary(summarize_simulation_data(sim_data), "simulation_design_summary")
        evaluate_dataset(sim_data, "Synthetic Simulation Data (N=300 full model comparison)", "simulation_summary.pdf", marker_label="Simulated marker")
        return

    if experiment == "real":
        parkinsons_path = "parkinsons_updrs.csv"
        if os.path.exists(parkinsons_path):
            sparse_data, dense_data, coef_df, missing_df = engineer_parkinsons_data(parkinsons_path, retained_fraction=0.10)
            export_design_summary(summarize_parkinsons_recovery(dense_data, sparse_data, missing_df), "parkinsons_design_summary")
            k_df = kmeans_diagnostics(coef_df)
            model_df = polynomial_model_selection(dense_data)
            export_diagnostics(k_df, model_df, missing_df)
            avg_hidden = missing_df["hidden_fraction"].mean()
            print(f"Average hidden fraction in sparse recovery task: {avg_hidden:.3f}", flush=True)
            evaluate_dataset(
                sparse_data,
                "Parkinson's Sparse Trajectory Recovery",
                "real_data_summary.pdf",
                marker_label="motor_UPDRS",
            )
            evaluate_train_validation_test(sparse_data)
        else:
            print(f"[INFO] {parkinsons_path} not found. Skipping Parkinson's experiment.")
        return

    raise ValueError("experiment must be one of: all, sim, real")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiclass likelihood contrast experiments.")
    parser.add_argument("--experiment", choices=["all", "sim", "real"], default="all")
    args = parser.parse_args()
    main(args.experiment)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
