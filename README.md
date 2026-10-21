# Multiclass likelihood-contrast experiments

Use Python 3.12 and install the tested dependencies:

```bash
python -m pip install -r requirements.txt
python run_experiment.py --experiment all
```

Run commands inside this folder. The all command runs the original simulation sample-size study, both main experiments, and the additional diagnostics. The estimator, main hyperparameters, simulation generator, phenotype construction, and subject-level folds are fixed as supplied.

To run only the diagnostic/comparator analyses:

```bash
python reviewer_analysis.py
```

To export numerical tables and figures into the accompanying LaTeX folder after running all experiments:

```bash
python export_manuscript_assets.py --manuscript ../Project_Draft
```

`simulation_output/` contains the core experimental outputs. `review_output/` contains all added baseline and sensitivity results, subject-level predictions, fold memberships, per-class metrics, confusion matrices, clustering diagnostics, patient curve coordinates, likelihood scores, and publication figures. Saved outputs are included. Class labels are 0, 1, and 2.

Main proposed-model settings: quadratic mean, random intercept/slope covariance, ridge 1e-4, covariance shrinkage 0, and uniform priors. Parameter fitting uses two-stage ridge regression and residual covariance estimation; prediction uses a marginal Gaussian likelihood. `MODEL_ALGORITHMS.md` describes the steps.

Three additional baselines use polynomial descriptors or grid-PCA features; all learned preprocessing uses training subjects only. The one-factor sensitivity settings are recorded in `review_output/settings.json` and the sensitivity CSVs. Default settings are not retuned using these comparisons.

Patient illustrations use class models fitted without the illustrated subject. The dense history is plotted only as reference; the conditional fitted curve uses the sparse record under the predicted class. Representatives are nearest to class centroids in standardized dense-descriptor space, with one lowest-ID error also displayed.

The simulation's nominal 10% integer-floor deletion removes zero visits for its default 4--8 visits per subject. The Parkinson masking experiment retains approximately 10% of dense records across follow-up. These are separate designs. The evaluation is trajectory-group recovery, with dense-history labels constructed across the cohort.

The manuscript uses the supplied `assets/clustering_diagnostics.pdf` unchanged. Its inertia and mean-silhouette values correspond to `review_output/clustering_diagnostics.csv`. Sensitivity results and individual fitted curves are presented in the main manuscript.

The reporting code includes macro sensitivity/recall, macro one-versus-rest specificity, and the unscaled multiclass Brier score (mean summed squared probability error; range 0--2). Per-class reports include one-versus-rest MCC. SVM probabilities use softmax of decision scores and are not calibrated. All eight models have saved held-out predictions. Their sensitivity/recall and specificity are included in manuscript Tables 6 and 7; pooled ROC plots remain in `review_output/`. The supplied Parkinson's variable table is included as `assets/data_variables.tex` and exported to the manuscript.

The sample-size results (Table 2) and training/validation/test diagnostic (Table 3) also report all seven metrics. Sample-size runs save their held-out predictions for checking these values. The original simulation-design CSV and performance-summary PDF remain in the code outputs; the manuscript omits those repeated displays. Baseline comparisons appear last in Results.
