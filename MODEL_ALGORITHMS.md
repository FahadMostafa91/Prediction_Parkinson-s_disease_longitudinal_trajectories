# Multiclass Likelihood Contrast: Algorithms

## Algorithm 1: Subject-level cross-validation

**Input:** Longitudinal table with columns `id`, `time`, `y`, and `class`.

1. Collapse the table to one row per subject: `(id, class)`.
2. Use stratified 5-fold cross-validation on subject IDs, not visits.
3. For each fold:
   - Split subjects into training and testing subjects.
   - Fit the normalizer on training visits only:
     - `time_norm = (time - train_time_mean) / train_time_sd`
     - `y_norm = (y - train_y_mean) / train_y_sd`
   - Apply the same training-fold normalizer to held-out visits.
   - Fit all models on the normalized training fold.
   - Predict labels for held-out subjects.
4. Pool the held-out predictions from all folds.
5. Report accuracy, macro-F1, MCC, and macro-AUC, with sensitivity/recall, specificity, and Brier scores in the relevant comparison tables.

## Algorithm 2: Proposed multiclass mixed-likelihood contrast classifier

For each class `k`:

1. Select training subjects with `class = k`.
2. Estimate the parameters of a class-specific Gaussian mixed-effects scoring model using a two-stage ridge/covariance procedure:
   - Fixed effects: `y ~ time + time_sq`
   - Random effects: subject-level random intercept and random time slope
3. Fit the class mean by ridge regression; fit each training subject's residual intercept/slope by ridge regression. Estimate the covariance of those deviations and the degrees-of-freedom-adjusted residual variance. Apply diagonal shrinkage and numerical variance floors.
4. Store these plug-in parameters and the uniform class prior.

For a new subject with sparse observations `(time_i, y_i)`:

1. Build the fixed-effect design matrix `X = [1, time, time^2]`.
2. Build the random-effect design matrix `Z = [1, time]`.
3. For every class `k`, compute the marginal Gaussian log-likelihood using
   `mu_k = X beta_k` and `V_k = Z D_k Z' + sigma_k^2 I`.
4. Add the log prior.
5. Convert class scores to posterior-like probabilities with a stable softmax.
6. Predict the class with the largest score.

## Algorithm 3: Simulation sample-size study

1. Generate sparse three-class longitudinal data for total sample sizes `N = 50, 100, 300, 500, 1000`.
2. Preserve the original imbalanced 1:2:3 class ratio as closely as integer counts allow.
3. Draw each subject's visit times irregularly on a two-year interval.
4. Generate three nonlinear quadratic trajectory classes with subject-level random effects and Gaussian measurement noise.
5. Retain 4 to 8 visits per subject.
6. For Table 2, evaluate the proposed quadratic multiclass MLC model with stratified subject-level 5-fold cross-validation at each sample size. The sample-size design is described in Section 5.1 and saved in `simulation_output/simulation_sample_design.csv`.
7. Generate a separate N = 300 cohort with seed 42 for the full model comparison (Table 6) against linear MLC, functional kNN, random forest, summary-feature SVM, and the three additional shape-based baselines.

## Algorithm 4: Parkinson's sparse trajectory recovery experiment

This experiment is an algorithmic stress test, not a clinical diagnostic validation.

1. Start from dense Parkinson's motor-UPDRS histories.
2. Keep subjects with at least three valid visits and nonconstant follow-up time.
3. Fit a dense-history quadratic curve for each subject.
4. Extract four dense-history descriptors: quadratic coefficient, linear coefficient, intercept, and total change.
5. Standardize descriptors and cluster them with `k = 3` using k-means.
6. Order the phenotype labels by mean total change for reproducibility.
7. Hide about 90% of each dense history while retaining early, middle, and late points.
8. Evaluate whether models recover the dense-history phenotype from the sparse input.

Default seed counts:

- Dense valid subjects: 42
- Class counts: Class 0 = 10, Class 1 = 10, Class 2 = 22
- Dense observations: 5,875
- Sparse observations: 587
- Mean dense visits per subject: 139.88
- Mean sparse visits per subject: 13.98
- Mean hidden fraction: 0.900

## Baselines

1. **Baseline MLC (linear):** class-specific `y ~ time` mixed model with random intercept.
2. **Functional kNN:** interpolate sparse trajectories on a training-fold grid, standardize the grid values and adjacent differences using training subjects, and use distance-weighted kNN.
3. **Random forest:** subject summaries: mean, standard deviation, minimum, maximum, and linear slope.
4. **Support vector machine:** same subject-summary features as random forest.
5. **Polynomial logistic regression:** quadratic curve coefficients and first-to-last response change.
6. **Polynomial SVM:** the same polynomial descriptors.
7. **Grid-PCA logistic regression:** interpolated values and differences on a training-fold grid, reduced by PCA.

## Diagnostics included in the code

- Exact subject counts and class distributions for each experiment.
- K-means inertia, silhouette score, and minimum cluster size for `k = 2, ..., 6`.
- AIC/BIC comparisons among linear, quadratic, cubic, and cubic B-spline dense-history mean models.
- Sparse recovery missingness summary by subject.

## Methodological safeguards

- Cross-validation splits are by subject, never by visit.
- Normalization is fit on training folds only.
- The Parkinson's experiment is reported as sparse trajectory recovery, not diagnosis or staging.
- Parkinson's labels are derived mathematical phenotypes, not externally assigned clinical outcomes.
- The Gaussian model is treated as an approximation for motor-UPDRS; ordinal GLMM likelihood contrasts are identified as the natural future extension.

## Train-validation-test diagnostic

For the Parkinson's sparse recovery study, the code also exports a diagnostic split summary.
Within each outer 5-fold split, the held-out fold remains the test set.
The remaining subjects are split again, reserving 20% of the training fold as validation subjects.
The approximate inner split is 64% training, 16% validation, and 20% test. Validation uses the inner training fit. The model is then refitted on all outer training subjects (approximately 80%); reported training and test predictions use that refit.
Hyperparameters remain fixed; the validation results are reported as an internal diagnostic, not as a tuning step. The split-count CSV names the outer training set used for refitting separately from the inner training set; inner training, validation, and test counts sum to the cohort size.

## Fixed hyperparameters and random seeds

No grid search or test-fold hyperparameter tuning is used in the reported experiments. The values below are fixed before evaluation.

- Global seed: `RANDOM_STATE = 42`.
- Outer validation: stratified subject-level 5-fold cross-validation, `shuffle=True`, approximately 80% training and 20% held-out test subjects per fold.
- Parkinson train-validation-test diagnostic: outer held-out fold remains the test fold; 20% of the training fold is reserved by stratified subject-level sampling for validation, giving approximately 64%/16%/20% training/validation/test by subject.
- Proposed MLC: `degree=2`, `random_effects='intercept_slope'`, `prior='uniform'`, `ridge=1e-4`, `covariance_shrinkage=0.0`.
- Linear MLC baseline: `degree=1`, `random_effects='intercept'`, `prior='uniform'`, `ridge=1e-4`, `covariance_shrinkage=0.0`.
- Functional kNN: `n_neighbors=5`, `weights='distance'`, 10 interpolation grid points.
- Random forest: `n_estimators=160`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `random_state=42`.
- RBF-SVM: `C=2.0`, `gamma='scale'`, `class_weight='balanced'`, `decision_function_shape='ovr'`.
- Simulation design: time range `(0, 2)`, 4--8 visits per subject, noise standard deviation 0.45, nominal missing rate 0.10 (integer-floor deletion removes zero visits at the 4--8-visit default), random-intercept standard deviation 0.70, random-slope standard deviation 0.25, and 1:2:3 class ratio.
- Parkinson sparse recovery: retained fraction 0.10, k-means `k=3`, `n_init=50`, and seed 42.

The simulation sample-size study uses seed `42 + 100*N + replicate` for each total sample size `N`; the manuscript uses one replicate per sample size.

## Diagnostic and comparison analyses

`reviewer_analysis.py` exports held-out confusion matrices, per-class metrics, one-factor sensitivity, shape-aware baseline comparisons, clustering diagnostics, and individual fitted curves. See `review_output/settings.json` for settings. Sensitivity keeps the main specification fixed and does not tune on test subjects. Classifier fitting and patient-specific conditional visualization are distinct; prediction uses the marginal likelihood.
