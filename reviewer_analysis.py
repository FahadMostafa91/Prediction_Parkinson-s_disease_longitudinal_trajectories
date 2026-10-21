from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import plt, RANDOM_STATE, N_SPLITS, StratifiedKFold
from emlc_classifier import EMLCClassifier
from data_simulation import LongitudinalSimulator
from analysis import compute_metrics, per_class_metrics
from run_experiment import (
    LongitudinalStandardizer, engineer_parkinsons_data,
    extract_functional_grid_features, _extract_model_probs, kmeans_diagnostics,
    _fit_feature_baselines, _fit_functional_knn_baseline,
)

# Output folder, main-model defaults, and plot colors.
OUT = Path('review_output')
DEFAULT = dict(ridge=1e-4, covariance_shrinkage=0.0, prior='uniform')
PROPOSED = 'Multiclass MLC (quadratic)'
COLORS = ['#1764a0', '#d07916', '#238351']


# Per-subject quadratic coefficients and first-to-last change.
def polynomial_features(df):
    rows = []
    for sid, g in df.groupby('id', sort=True):
        g = g.sort_values('time')
        p = np.polyfit(g.time, g.y, 2)
        rows.append([sid, int(g['class'].iloc[0]), *p,
                     float(g.y.iloc[-1] - g.y.iloc[0])])
    return pd.DataFrame(rows, columns=['id', 'true_class', 'quadratic',
                                     'linear', 'intercept', 'change'])


# Subject-level stratified folds with training-fold normalization.
def folds(df):
    subjects = df.groupby('id', as_index=False)['class'].first()
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(subjects.id, subjects['class']), 1):
        train = df[df.id.isin(subjects.id.iloc[tr])].copy()
        test = df[df.id.isin(subjects.id.iloc[te])].copy()
        norm = LongitudinalStandardizer().fit(train)
        yield fold, train, test, norm, norm.transform(train), norm.transform(test)


# Polynomial logistic regression, polynomial SVM, and grid-PCA logistic regression.
def new_baselines(train, test):
    tr, te = polynomial_features(train), polynomial_features(test)
    xcols = ['quadratic', 'linear', 'intercept', 'change']
    models = {
        'Polynomial logistic regression': make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced',
                                                  max_iter=5000, random_state=42)),
        'Polynomial SVM': make_pipeline(StandardScaler(), SVC(
            C=2.0, gamma='scale', class_weight='balanced',
            decision_function_shape='ovr')),
    }
    predictions = {}
    for name, model in models.items():
        model.fit(tr[xcols], tr.true_class)
        predictions[name] = _extract_model_probs(model, te[xcols], te.true_class)
        predictions[name]['id'] = te.id.to_numpy()
    grid = np.linspace(train.time.quantile(.02), train.time.quantile(.98), 10)
    tr, te = (extract_functional_grid_features(d, grid) for d in (train, test))
    xcols = [c for c in tr if c not in ('id', 'true_class')]
    model = make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver='full'),
                          LogisticRegression(C=1.0, class_weight='balanced',
                                             max_iter=5000, random_state=42))
    model.fit(tr[xcols], tr.true_class)
    predictions['Grid-PCA logistic regression'] = _extract_model_probs(
        model, te[xcols], te.true_class)
    predictions['Grid-PCA logistic regression']['id'] = te.id.to_numpy()
    return predictions


# Run all eight models across folds; save predictions, metrics, and confusion matrix.
def evaluate(df, name):
    all_preds, cache, membership = {}, {}, []
    for fold, train, test, norm, tr, te in folds(df):
        model = EMLCClassifier(degree=2, random_effects='intercept_slope', **DEFAULT)
        predictions = {PROPOSED: model.fit(tr).predict(te), **new_baselines(tr, te)}
        linear = EMLCClassifier(degree=1, random_effects='intercept', **DEFAULT)
        rf, svm = _fit_feature_baselines(tr, te)
        predictions.update({
            'Baseline MLC (linear)': linear.fit(tr).predict(te),
            'Functional kNN': _fit_functional_knn_baseline(tr, te),
            'Random forest': rf, 'Support vector machine': svm,
        })
        for label, p in predictions.items():
            if 'id' not in p:
                p['id'] = te.groupby('id').first().index.to_numpy()
            p['fold'] = fold
            all_preds.setdefault(label, []).append(p)
        for sid in test.id.unique():
            cache[sid] = (model, norm, te[te.id == sid].copy())
        membership.extend(dict(fold=fold, id=int(sid), split=split)
                          for split, data in [('train', train), ('test', test)]
                          for sid in sorted(data.id.unique()))
    all_preds = {label: pd.concat(ps, ignore_index=True) for label, ps in all_preds.items()}
    for label, pred in all_preds.items():
        assert pred.id.nunique() == len(pred) == df.id.nunique()
        slug = label.lower().replace(' ', '_').replace('(', '').replace(')', '')
        pred.to_csv(OUT / f'{name}_{slug}_predictions.csv', index=False)
    pd.DataFrame(membership).to_csv(OUT / f'{name}_fold_membership.csv', index=False)
    metrics = compute_metrics(all_preds)
    metrics.to_csv(OUT / f'{name}_extended_metrics.csv', index=False)
    additions = [PROPOSED, 'Polynomial logistic regression', 'Polynomial SVM',
                 'Grid-PCA logistic regression']
    metrics.set_index('Model').loc[additions].reset_index().to_csv(
        OUT / f'{name}_additional_baselines.csv', index=False)
    p = all_preds[PROPOSED]
    cm = confusion_matrix(p.true_class, p.pred_class, labels=[0, 1, 2])
    pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).to_csv(
        OUT / f'{name}_confusion_matrix.csv', index_label='True class')
    report = per_class_metrics(p.true_class.to_numpy(), p.pred_class.to_numpy(), [0, 1, 2])
    report.to_csv(OUT / f'{name}_per_class.csv', index=False)
    return all_preds, cache, cm, report


# One-factor-at-a-time sensitivity for ridge, covariance shrinkage, and prior.
def sensitivity(df, name):
    settings = [('Default', 'default', DEFAULT.copy())]
    variations = [('Ridge', 'ridge', [0., 1e-6, 1e-2, 1.]),
                  ('Shrinkage', 'covariance_shrinkage', [.25, .5, .75, 1.]),
                  ('Prior', 'prior', ['empirical'])]
    for group, parameter, values in variations:
        for value in values:
            settings.append((group, str(value), {**DEFAULT, parameter: value}))
    fold_data = list(folds(df))
    rows = []
    for group, value, parameters in settings:
        preds = []
        for fold, _, _, _, tr, te in fold_data:
            model = EMLCClassifier(degree=2, random_effects='intercept_slope', **parameters)
            p = model.fit(tr).predict(te)
            p['fold'] = fold
            preds.append(p)
        p = pd.concat(preds, ignore_index=True)
        slug = f'{group}_{value}'.lower().replace('.', 'p')
        p.to_csv(OUT / f'{name}_sensitivity_{slug}_predictions.csv', index=False)
        row = compute_metrics({PROPOSED: p}).iloc[0].drop('Model').to_dict()
        rows.append(dict(Experiment=name, Parameter=group, Value=value, **parameters, **row))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f'{name}_sensitivity.csv', index=False)
    return result


def save_figure(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


# Plot held-out confusion matrices with counts and row percentages.
def confusion_figure(matrices):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    for ax, (name, cm) in zip(axes, matrices.items()):
        normalized = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i,j]}\n({normalized[i,j]:.0%})',
                        ha='center', va='center', fontsize=10,
                        color='white' if normalized[i,j] > .55 else 'black')
        ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xlabel='Predicted class',
               ylabel='True class', title=name)
        ax.grid(False)
    save_figure(fig, 'confusion_matrices')


# Save k-means diagnostics and copy the supplied clustering figure.
def clustering_figure(coef):
    kdf = kmeans_diagnostics(coef)
    kdf.to_csv(OUT / 'clustering_diagnostics.csv', index=False)
    import shutil
    shutil.copy2(Path(__file__).resolve().parent / 'assets' / 'clustering_diagnostics.pdf',
                 OUT / 'clustering_diagnostics.pdf')
    return kdf


# Pooled macro-averaged one-versus-rest ROC curves for all models.
def roc_figure(predictions):
    names = ['Baseline MLC (linear)', PROPOSED, 'Functional kNN', 'Random forest',
             'Support vector machine', 'Polynomial logistic regression',
             'Polynomial SVM', 'Grid-PCA logistic regression']
    short = ['Linear MLC', 'Quadratic MLC', 'Functional kNN', 'Random forest',
             'Summary SVM', 'Polynomial LR', 'Polynomial SVM', 'Grid-PCA LR']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8))
    grid = np.linspace(0, 1, 1001)
    for ax, (dataset, results) in zip(axes, predictions.items()):
        for index, (name, label) in enumerate(zip(names, short)):
            p = results[name]
            curves = []
            for cls in [0, 1, 2]:
                fpr, tpr, _ = roc_curve(p.true_class == cls, p[f'prob_{cls}'])
                curves.append(np.interp(grid, fpr, tpr))
            auc = compute_metrics({name: p}).Macro_AUC.iloc[0]
            ax.plot(grid, np.mean(curves, axis=0), color=plt.get_cmap('tab10')(index),
                    lw=1.8, label=f'{label} ({auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title=dataset,
               xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    save_figure(fig, 'roc_curves')


# Plot representative and misclassified patients with class means and conditional fits.
def patient_figure(dense, coef, preds, cache):
    x = StandardScaler().fit_transform(coef[['quadratic', 'linear', 'intercept', 'total_change']])
    selected = []
    for cls in range(3):
        ix = np.flatnonzero(coef['class'].to_numpy() == cls)
        distances = np.linalg.norm(x[ix] - x[ix].mean(axis=0), axis=1)
        selected.append((int(coef.iloc[ix[np.argmin(distances)]].id), 'Class representative'))
    wrong = preds[preds.true_class != preds.pred_class].sort_values('id')
    if not wrong.empty:
        selected.append((int(wrong.id.iloc[0]), 'Misclassified example'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    rows, curves, scores = [], [], []
    for ax, (sid, selection) in zip(axes.flat, selected):
        model, norm, sparse = cache[sid]
        dg = dense[dense.id == sid].sort_values('time')
        pred = preds[preds.id == sid].iloc[0]
        times = np.linspace(dg.time.min(), dg.time.max(), 200)
        tn = (times - norm.means_['time']) / norm.stds_['time']
        ax.plot(dg.time, dg.y, '.', color='0.73', ms=3, label='Dense reference')
        ax.scatter(sparse.time * norm.stds_['time'] + norm.means_['time'],
                   sparse.y * norm.stds_['y'] + norm.means_['y'],
                   s=21, color='black', zorder=4, label='Sparse input')
        for cls in range(3):
            info = model.models[cls]
            mean = model._fixed_design(tn) @ info['beta']
            mean = mean * norm.stds_['y'] + norm.means_['y']
            ax.plot(times, mean, '--', lw=1, color=COLORS[cls], label=f'Class {cls} mean')
            scores.append(dict(id=sid, Class=cls,
                               Log_score=model._calc_marginal_loglike(info, sparse)
                               + np.log(model.priors[cls])))
        cls = int(pred.pred_class)
        info = model.models[cls]
        X, Z = model._fixed_design(sparse.time), model._random_design(sparse.time)
        V = model._nearest_psd(Z @ info['D'] @ Z.T + info['sigma2'] * np.eye(len(Z)))
        b = info['D'] @ Z.T @ np.linalg.solve(V, sparse.y.to_numpy() - X @ info['beta'])
        fit = model._fixed_design(tn) @ info['beta'] + model._random_design(tn) @ b
        fit = fit * norm.stds_['y'] + norm.means_['y']
        ax.plot(times, fit, color=COLORS[cls], lw=2, label='Conditional fitted curve')
        ax.set(title=f'Patient {sid}: true {int(pred.true_class)}, predicted {cls}\n{selection}',
               xlabel='Follow-up time (days)', ylabel='Motor-UPDRS')
        curves.extend(dict(id=sid, time=t, fitted_y=y) for t, y in zip(times, fit))
        rows.append(dict(id=sid, selection=selection, true_class=int(pred.true_class),
                         predicted_class=cls, fold=int(pred.fold), sparse_visits=len(sparse)))
    axes.flat[0].legend(fontsize=7, frameon=True, ncol=2)
    save_figure(fig, 'individual_trajectories')
    pd.DataFrame(rows).to_csv(OUT / 'patient_examples.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'patient_fitted_curves.csv', index=False)
    pd.DataFrame(scores).to_csv(OUT / 'patient_likelihood_scores.csv', index=False)


# Run all diagnostic analyses and write the settings file.
def main():
    OUT.mkdir(exist_ok=True)
    sparse, dense, coef, _ = engineer_parkinsons_data('parkinsons_updrs.csv')
    sim = LongitudinalSimulator(random_state=RANDOM_STATE).simulate()
    matrices, sensitivities, roc_predictions = {}, {}, {}
    for name, df in [('simulation', sim), ('parkinsons', sparse)]:
        print(f'Additional analyses: {name}', flush=True)
        predictions, cache, cm, _ = evaluate(df, name)
        roc_predictions['Simulation' if name == 'simulation' else 'Parkinson recovery'] = predictions
        matrices['Simulation' if name == 'simulation' else 'Parkinson recovery'] = cm
        sensitivities['Simulation' if name == 'simulation' else 'Parkinson recovery'] = sensitivity(df, name)
        if name == 'parkinsons':
            patient_figure(dense, coef, predictions[PROPOSED], cache)
    confusion_figure(matrices)
    clustering_figure(coef)
    roc_figure(roc_predictions)
    settings = dict(seed=42, folds=5, default=DEFAULT, sensitivity='one factor at a time',
                    logistic=dict(C=1.0, class_weight='balanced', max_iter=5000),
                    polynomial_svm=dict(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced'),
                    grid_pca=dict(grid_points=10, variance_retained=.95, differences=True),
                    patient_selection='Nearest standardized dense-descriptor centroid per class; lowest-ID error',
                    brier='Mean summed squared error across three classes; range 0 to 2',
                    specificity='One versus rest; macro average gives equal weight to each class',
                    svm_probabilities='Softmax of decision scores; no probability calibration')
    (OUT / 'settings.json').write_text(json.dumps(settings, indent=2))
    print('Additional analyses complete.', flush=True)


if __name__ == '__main__':
    main()
