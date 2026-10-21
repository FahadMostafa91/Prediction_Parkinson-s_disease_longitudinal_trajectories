
try:
    from config import np, pd, multivariate_normal
except Exception:
    import numpy as np
    import pandas as pd
    from scipy.stats import multivariate_normal


# Multiclass mixed-likelihood contrast classifier for sparse longitudinal data.
class EMLCClassifier:

    # Set model options and validate degree, prior, random effects, and shrinkage.
    def __init__(
        self,
        degree=2,
        prior="uniform",
        random_effects="intercept_slope",
        ridge=1e-4,
        covariance_shrinkage=0.35,
        verbose=False,
        use_priors=None,
    ):
        if use_priors is not None:
            prior = "empirical" if use_priors else None
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        if prior not in ("uniform", "empirical", None):
            raise ValueError("prior must be 'uniform', 'empirical', or None")
        if random_effects not in ("intercept", "intercept_slope"):
            raise ValueError("random_effects must be 'intercept' or 'intercept_slope'")
        if not 0 <= covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must be between 0 and 1")

        self.degree = degree
        self.prior = prior
        self.random_effects = random_effects
        self.ridge = float(ridge)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.verbose = verbose
        self.models = {}
        self.priors = {}
        self.classes_ = []

    # Check required columns, add squared time, and sort visits by subject and time.
    def _prepare_data(self, df):
        required = {"id", "time", "y", "class"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Input data missing required columns: {sorted(missing)}")
        out = df.copy()
        out["time_sq"] = out["time"] ** 2
        return out.sort_values(["id", "time"])

    # Fixed-effect design matrix: [1, t] or [1, t, t^2].
    def _fixed_design(self, t):
        t = np.asarray(t, dtype=float)
        if self.degree == 1:
            return np.column_stack([np.ones(len(t)), t])
        return np.column_stack([np.ones(len(t)), t, t**2])

    # Random-effect design matrix: [1, t] or [1].
    def _random_design(self, t):
        t = np.asarray(t, dtype=float)
        if self.random_effects == "intercept_slope":
            return np.column_stack([np.ones(len(t)), t])
        return np.ones((len(t), 1))

    # Symmetrize a matrix and lift small eigenvalues to keep it positive definite.
    @staticmethod
    def _nearest_psd(matrix, eps=1e-5):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 0:
            matrix = np.array([[float(matrix)]])
        matrix = 0.5 * (matrix + matrix.T)
        try:
            min_eig = float(np.min(np.linalg.eigvalsh(matrix)))
            if min_eig < eps:
                matrix = matrix + np.eye(matrix.shape[0]) * (eps - min_eig)
        except Exception:
            matrix = np.eye(matrix.shape[0]) * eps
        return matrix

    def _ridge_solve(self, X, y):
        penalty = self.ridge * np.eye(X.shape[1])
        return np.linalg.solve(X.T @ X + penalty, X.T @ y)

    # Two-stage fit for one class: ridge mean, subject random effects, covariance, residual variance.
    def _fit_class_model(self, subset):
        t = subset["time"].to_numpy(dtype=float)
        y = subset["y"].to_numpy(dtype=float)
        X = self._fixed_design(t)
        beta = self._ridge_solve(X, y)

        q = self._random_design([0.0]).shape[1]
        random_effects = []
        residual_ss = 0.0
        n_obs = 0

        for _, subject_df in subset.groupby("id"):
            tg = subject_df["time"].to_numpy(dtype=float)
            yg = subject_df["y"].to_numpy(dtype=float)
            Xg = self._fixed_design(tg)
            Zg = self._random_design(tg)
            raw_residual = yg - Xg @ beta
            if len(subject_df) >= q:
                b = self._ridge_solve(Zg, raw_residual)
            else:
                b = np.zeros(q)
                b[0] = float(np.mean(raw_residual)) if len(raw_residual) else 0.0
            random_effects.append(b)
            adjusted = raw_residual - Zg @ b
            residual_ss += float(np.dot(adjusted, adjusted))
            n_obs += len(subject_df)

        B = np.vstack(random_effects) if random_effects else np.zeros((1, q))
        if len(B) > 1:
            D = np.cov(B, rowvar=False)
        else:
            D = np.eye(q) * 1e-4
        if q == 1 and np.ndim(D) == 0:
            D = np.array([[float(D)]])
        diagonal = np.diag(np.diag(D))
        D = (1.0 - self.covariance_shrinkage) * D + self.covariance_shrinkage * diagonal
        D = self._nearest_psd(D)

        df_resid = max(n_obs - len(random_effects) * q - X.shape[1], 1)
        sigma2 = max(float(residual_ss / df_resid), 1e-4)
        return {"beta": beta, "D": D, "sigma2": sigma2, "n_subjects": subset["id"].nunique()}

    # Fit one model and one prior per class on the training subjects.
    def fit(self, train_data):
        train_data = self._prepare_data(train_data)
        self.classes_ = sorted(int(c) for c in train_data["class"].unique())
        total_subjects = train_data["id"].nunique()
        self.models = {}
        self.priors = {}

        for cls in self.classes_:
            subset = train_data[train_data["class"] == cls].copy()
            n_cls = subset["id"].nunique()
            if self.prior == "empirical":
                self.priors[cls] = max(n_cls / total_subjects, 1e-12)
            elif self.prior == "uniform":
                self.priors[cls] = 1.0 / len(self.classes_)
            else:
                self.priors[cls] = 1.0
            self.models[cls] = self._fit_class_model(subset)
        return self

    # Marginal Gaussian log-likelihood with mean X*beta and covariance Z*D*Z' + sigma^2*I.
    def _calc_marginal_loglike(self, model_info, subject_df):
        subject_df = self._prepare_data(subject_df)
        t = subject_df["time"].to_numpy(dtype=float)
        y = subject_df["y"].to_numpy(dtype=float)
        if len(y) == 0:
            return -np.inf
        X = self._fixed_design(t)
        Z = self._random_design(t)
        mu = X @ model_info["beta"]
        V = Z @ model_info["D"] @ Z.T + model_info["sigma2"] * np.eye(len(y))
        V = self._nearest_psd(V)
        try:
            ll = multivariate_normal.logpdf(y, mean=mu, cov=V, allow_singular=True)
            return float(ll) if np.isfinite(ll) else -np.inf
        except Exception:
            return -np.inf

    # Convert class scores to probabilities with a numerically stable softmax.
    @staticmethod
    def _softmax(scores):
        finite = {k: (v if np.isfinite(v) else -1e12) for k, v in scores.items()}
        max_score = max(finite.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in finite.items()}
        denom = sum(exp_scores.values())
        if denom <= 0 or not np.isfinite(denom):
            return {k: 1.0 / len(scores) for k in scores}
        return {k: v / denom for k, v in exp_scores.items()}

    # Score each subject under every class, add the log prior, and return predictions.
    def predict(self, test_data):
        test_data = self._prepare_data(test_data)
        rows = []
        for sub_id, sub_df in test_data.groupby("id"):
            scores = {}
            for cls in self.classes_:
                ll = self._calc_marginal_loglike(self.models[cls], sub_df)
                if self.prior is not None:
                    ll += np.log(max(self.priors.get(cls, 1e-12), 1e-12))
                scores[cls] = ll
            probs = self._softmax(scores)
            pred = int(max(probs, key=probs.get))
            true_cls = int(sub_df["class"].iloc[0])
            row = {
                "id": sub_id,
                "true_class": true_cls,
                "pred_class": pred,
                "true_class_prob": probs.get(true_cls, 0.0),
            }
            for cls in self.classes_:
                row[f"prob_{int(cls)}"] = probs.get(cls, 0.0)
            rows.append(row)
        return pd.DataFrame(rows)
