# Export result tables and figures into the manuscript folder.
from pathlib import Path
import argparse
import shutil
import pandas as pd


# Build a LaTeX table, optionally marked in blue as a revision.
def table(caption, label, columns, rows, alignment, blue=True):
    color = '\\color{revisionblue}\n' if blue else ''
    caption = '\\rev{' + caption + '}' if blue else caption
    settings = '\\small\\setlength{\\tabcolsep}{4pt}\n' if blue else ''
    return ('\\begin{table}[!ht]\n\\centering\n' + color
            + f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
            + settings
            + f'\\begin{{tabular}}{{{alignment}}}\n\\toprule\n'
            + ' & '.join(columns) + ' \\\\\n\\midrule\n'
            + '\n'.join(' & '.join(map(str, row)) + ' \\\\' for row in rows)
            + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


# Copy assets and write all manuscript tables.
def main(destination):
    destination = Path(destination)
    # Create the tables and figures folders.
    (destination / 'tables').mkdir(parents=True, exist_ok=True)
    (destination / 'figures').mkdir(exist_ok=True)
    output = Path(__file__).resolve().parent
    # Copy the variable table and manuscript figures.
    shutil.copy2(output / 'assets' / 'data_variables.tex', destination / 'tables')
    for name in ['confusion_matrices', 'clustering_diagnostics',
                 'individual_trajectories']:
        source = output / ('assets' if name == 'clustering_diagnostics' else 'review_output') / f'{name}.pdf'
        shutil.copy2(source, destination / 'figures')
    # Sample-size and train/validation/test tables with all seven metrics.
    metrics = ['Accuracy', 'Macro_F1', 'MCC', 'Macro_AUC', 'Brier', 'Macro_Recall', 'Macro_Specificity']
    for filename, first, label, caption in [
        ('simulation_sample_size_results', 'Subjects', 'tab:simulation-sample-size-results',
         'Proposed multiclass MLC performance across simulated sample sizes.'),
        ('train_validation_test_metrics', 'Model', 'tab:train-validation-test',
         "Training, validation, and held-out test performance for the proposed multiclass MLC model in the Parkinson's sparse recovery experiment.")]:
        data = pd.read_csv(output / 'simulation_output' / f'{filename}.csv')
        rows = []
        for _, row in data.iterrows():
            values = [str(int(row[first])) if first == 'Subjects' else row[first]]
            for i, key in enumerate(metrics):
                value = f'{row[key]:.4f}'
                values.append(r'\rev{' + value + '}' if i >= 4 else value)
            rows.append(values)
        headers = ['Subjects' if first == 'Subjects' else r'\rev{Evaluation set}',
                   'Accuracy', r'\shortstack{Macro-\\F1}', 'MCC', r'\shortstack{Macro-\\AUC}',
                   r'\rev{Brier}', r'\rev{\shortstack{Sensitivity/\\recall}}', r'\rev{Specificity}']
        rendered = table(caption, label, headers, rows, 'lrrrrrrr', blue=False)
        rendered = rendered.replace(r'\begin{tabular}', r'\small\setlength{\tabcolsep}{3pt}' + '\n' + r'\begin{tabular}')
        (destination / 'tables' / f'{filename}.tex').write_text(rendered)
    rows = []
    # Per-class performance table for both experiments.
    for dataset, label in [('simulation', 'Simulation'), ('parkinsons', 'Parkinson')]:
        data = pd.read_csv(output / 'review_output' / f'{dataset}_per_class.csv')
        for r in data.itertuples(index=False):
            rows.append([label, r.Class, r.Support, f'{r.Precision:.4f}',
                         f'{r.Recall:.4f}', f'{r.Specificity:.4f}', f'{r.F1:.4f}', f'{r.MCC:.4f}'])
    (destination / 'tables' / 'per_class.tex').write_text(table(
        'Pooled held-out per-class performance of quadratic MLC. MCC treats each class versus the rest.', 'tab:per-class',
        ['Experiment', 'Class', 'Subjects', 'Precision', r'\shortstack{Sensitivity/\\recall}',
         'Specificity', 'F1', 'MCC'], rows, 'llrrrrrr'))
    # Combine original and added baselines into the full model-comparison tables.
    for dataset, filename in [('simulation', 'simulation_summary_metrics'),
                              ('parkinsons', 'real_data_summary_metrics')]:
        old = pd.read_csv(output / 'simulation_output' / f'{filename}.csv')
        new = pd.read_csv(output / 'review_output' / f'{dataset}_additional_baselines.csv').iloc[1:]
        joined = pd.concat([old, new], ignore_index=True)
        extra = pd.read_csv(output / 'review_output' / f'{dataset}_extended_metrics.csv').set_index('Model')
        for metric in ['Macro_Recall', 'Macro_Specificity', 'Brier']:
            joined[metric] = joined.Model.map(extra[metric])
        joined.to_csv(output / 'review_output' / f'{dataset}_all_baselines.csv', index=False)
        title = ('Synthetic Simulation Data (N=300 full model comparison)' if dataset == 'simulation'
                 else "Parkinson's Sparse Trajectory Recovery")
        label = 'tab:simulation-summary-metrics' if dataset == 'simulation' else 'tab:real-data-summary-metrics'
        lines = [r'\begin{table}[!ht]', r'\centering',
                 r'\caption{Held-out subject-level stratified 5-fold cross-validation metrics for ' + title + '.}',
                 r'\label{' + label + '}', r'\small\setlength{\tabcolsep}{3pt}',
                 r'\begin{tabular}{lrrrrrrr}', r'\toprule',
                 r'Model & Accuracy & \shortstack{Macro-\\F1} & MCC & \shortstack{Macro-\\AUC} & \rev{Brier} & \rev{\shortstack{Sensitivity/\\recall}} & \rev{Specificity} \\', r'\midrule']
        for i, row in joined.iterrows():
            # Citation keys for each comparison model.
            citations = {
                'Baseline MLC (linear)': 'klen2020likelihood',
                'Functional kNN': 'ramsay2005functional',
                'Random forest': 'breiman2001random',
                'Support vector machine': 'cortes1995support',
                'Polynomial logistic regression': 'pedregosa2011scikit',
                'Polynomial SVM': 'cortes1995support',
                'Grid-PCA logistic regression': 'pedregosa2011scikit',
            }
            model_name = row.Model
            if dataset == 'simulation' and row.Model in citations:
                model_name += r'~\rev{\citep{' + citations[row.Model] + '}}'
            values = [model_name] + [f'{row[c]:.4f}' for c in ['Accuracy', 'Macro_F1', 'MCC', 'Macro_AUC']]
            if i >= len(old):
                values = [r'\rev{' + v + '}' for v in values]
            values.append(r'\rev{' + f'{row.Brier:.4f}' + '}')
            values.extend(r'\rev{' + f'{row[c]:.4f}' + '}'
                          for c in ['Macro_Recall', 'Macro_Specificity'])
            lines.append(' & '.join(values) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        (destination / 'tables' / f'{filename}.tex').write_text('\n'.join(lines) + '\n')
    rows = []
    # Full one-factor sensitivity table.
    for dataset, label in [('simulation', 'Simulation'), ('parkinsons', 'Parkinson')]:
        data = pd.read_csv(output / 'review_output' / f'{dataset}_sensitivity.csv')
        for r in data.itertuples(index=False):
            rows.append([label, f'{r.ridge:g}', f'{r.covariance_shrinkage:g}', r.prior,
                         f'{r.Accuracy:.4f}', f'{r.Macro_F1:.4f}', f'{r.MCC:.4f}', f'{r.Macro_AUC:.4f}', f'{r.Brier:.4f}'])
    (destination / 'tables' / 'sensitivity_full.tex').write_text(table(
        'One-factor-at-a-time sensitivity results; unchanged factors retain the main specification.',
        'tab:sensitivity-full', ['Experiment', '$\\lambda$', '$\\alpha$', 'Prior',
                                 'Accuracy', 'Macro-F1', 'MCC', 'Macro-AUC', 'Brier'], rows, 'lrrlrrrrr'))


# Command-line entry point: --manuscript gives the target folder.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manuscript', required=True)
    main(parser.parse_args().manuscript)
