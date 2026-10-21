
# Metrics, plotting, and LaTeX export helpers for the experiments.
from config import np, pd, plt, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

CORE_METRICS = ["Accuracy", "Macro_F1", "MCC", "Macro_AUC"]


def _labels_from_results(results_dict):
    labels = set()
    for df in results_dict.values():
        labels.update(df["true_class"].astype(int).unique())
        labels.update(df["pred_class"].astype(int).unique())
    return sorted(labels)


# One-versus-rest precision, recall, specificity, F1, and MCC for each class.
def per_class_metrics(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - cm.sum(axis=1) - fp
    specificity = np.divide(tn, tn + fp, out=np.zeros(len(labels)), where=tn + fp != 0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame(dict(Class=labels, Support=support, Precision=precision,
                             Recall=recall, Specificity=specificity, F1=f1,
                             MCC=[matthews_corrcoef(y_true == k, y_pred == k) for k in labels]))


# Pooled held-out metrics per model: accuracy, macro-F1, MCC, macro-AUC, recall, specificity, Brier.
def compute_metrics(results_dict):
    labels = _labels_from_results(results_dict)
    rows = []
    for name, df in results_dict.items():
        y_true = df["true_class"].astype(int).to_numpy()
        y_pred = df["pred_class"].astype(int).to_numpy()
        prob_cols = [f"prob_{int(label)}" for label in labels]
        auc = np.nan
        brier = np.nan
        if all(col in df.columns for col in prob_cols):
            y_probs = df[prob_cols].astype(float).to_numpy()
            row_sums = y_probs.sum(axis=1, keepdims=True)
            y_probs = np.divide(y_probs, row_sums, out=np.zeros_like(y_probs), where=row_sums != 0)
            target = y_true[:, None] == np.asarray(labels)[None, :]
            # Multiclass Brier score: mean summed squared probability error.
            brier = np.mean(np.sum((y_probs - target) ** 2, axis=1))
            try:
                auc = roc_auc_score(y_true, y_probs, labels=labels, multi_class="ovr", average="macro")
            except Exception:
                auc = np.nan
        classes = per_class_metrics(y_true, y_pred, labels)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro_F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Macro_AUC": auc,
            "Macro_Recall": classes.Recall.mean(),
            "Macro_Specificity": classes.Specificity.mean(),
            "Brier": brier,
        })
    return pd.DataFrame(rows)


# Grouped bar chart comparing models on the core metrics.
def plot_performance_comparison(metrics_df, title="Held-out model comparison"):
    names = metrics_df["Model"].tolist()
    metrics = CORE_METRICS
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(names))
    width = 0.18
    for idx, metric in enumerate(metrics):
        values = metrics_df[metric].fillna(0).to_numpy(dtype=float)
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace("_", "-"))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Held-out score")
    ax.set_ylim(0, 1.08)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.08, right=0.98)
    return fig


# Bar chart of proposed-model scores in the simulation and sparse recovery experiments.
def plot_metric_gain(sim_metrics, real_metrics, title=None):
    rows = []
    for label, df in [("Simulation", sim_metrics), ("Sparse recovery", real_metrics)]:
        row = df[df["Model"].str.contains("quadratic", case=False, na=False)].iloc[0]
        for metric in CORE_METRICS:
            rows.append({"Experiment": label, "Metric": metric.replace("_", "-"), "Score": float(row[metric])})
    long_df = pd.DataFrame(rows)
    metrics = long_df["Metric"].unique().tolist()
    experiments = long_df["Experiment"].unique().tolist()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(metrics))
    width = 0.36
    for idx, experiment in enumerate(experiments):
        vals = [long_df[(long_df["Experiment"] == experiment) & (long_df["Metric"] == metric)]["Score"].iloc[0] for metric in metrics]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width, label=experiment)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.16, top=0.88, left=0.10, right=0.96)
    return fig


# Cluster diagnostics and dense-history mean-model AIC/BIC comparison.
def plot_kmeans_and_model_diagnostics(k_df, model_df, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax1 = axes[0]
    ax1.plot(k_df["k"], k_df["inertia"], marker="o", linewidth=2, label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Within-cluster sum of squares")
    ax1.set_title("Cluster diagnostics")
    ax1.grid(alpha=0.25)
    ax1b = ax1.twinx()
    ax1b.plot(k_df["k"], k_df["silhouette"], marker="s", linestyle="--", linewidth=2, label="Silhouette")
    ax1b.set_ylabel("Silhouette score")
    ax1.axvline(3, linestyle=":", linewidth=1.5)
    ax1.text(3.05, ax1.get_ylim()[1] * 0.92, "chosen k=3", fontsize=9)

    ax2 = axes[1]
    x = np.arange(len(model_df))
    width = 0.35
    delta_aic = model_df["AIC"] - model_df["AIC"].min()
    delta_bic = model_df["BIC"] - model_df["BIC"].min()
    ax2.bar(x - width / 2, delta_aic, width, label="Delta AIC")
    ax2.bar(x + width / 2, delta_bic, width, label="Delta BIC")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_df["model"].str.replace("Polynomial degree ", "Poly ", regex=False), rotation=15, ha="right")
    ax2.set_ylabel("Difference from best model")
    ax2.set_title("Dense-history mean model")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(1.1, max(delta_bic.max(), delta_aic.max()) * 0.80, "BIC favors quadratic", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.subplots_adjust(bottom=0.22, top=0.82, left=0.07, right=0.93, wspace=0.32)
    else:
        fig.subplots_adjust(bottom=0.22, top=0.90, left=0.07, right=0.93, wspace=0.32)
    return fig


# Escape special characters for LaTeX output.
def _latex_escape(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


# Write a model-metrics table as a LaTeX file.
def export_metrics_latex(metrics_df, output_path, caption="Evaluation metrics", label="tab:metrics"):
    source_cols = ["Model"] + CORE_METRICS
    headers = ["Model", "Accuracy", "Macro-F1", "MCC", "Macro-AUC"]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + r"}",
        r"\label{" + _latex_escape(label) + r"}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in metrics_df[source_cols].iterrows():
        vals = []
        for col in source_cols:
            if col == "Model":
                vals.append(_latex_escape(row[col]))
            else:
                vals.append("--" if pd.isna(row[col]) else f"{float(row[col]):.4f}")
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


# Write any DataFrame as a formatted LaTeX table.
def export_dataframe_latex(df, output_path, caption="Table", label="tab:table", float_cols=None):
    float_cols = float_cols or {}
    align = "l" + "r" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + r"}",
        r"\label{" + _latex_escape(label) + r"}",
        r"\begin{tabular}{" + align + r"}",
        r"\toprule",
        " & ".join(_latex_escape(str(c).replace("_", " ")) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if col in float_cols and not pd.isna(val):
                vals.append(f"{float(val):.{int(float_cols[col])}f}")
            elif isinstance(val, (int, np.integer)) and not pd.isna(val):
                vals.append(str(int(val)))
            elif isinstance(val, (float, np.floating)) and not pd.isna(val):
                vals.append(str(int(val)) if float(val).is_integer() else f"{float(val):.4f}")
            else:
                vals.append(_latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


# Line plot of proposed-model performance across simulated sample sizes.
def plot_sample_size_performance(sample_df, title="Simulation sample-size performance"):
    metrics = ["Accuracy", "Macro_F1", "MCC", "Macro_AUC"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for metric in metrics:
        ax.plot(sample_df["Subjects"], sample_df[metric], marker="o", linewidth=2, label=metric.replace("_", "-"))
    ax.set_xlabel("Number of simulated subjects")
    ax.set_ylabel("Held-out score")
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.14, top=0.88, left=0.10, right=0.96)
    return fig
