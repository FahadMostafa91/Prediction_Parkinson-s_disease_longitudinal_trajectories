
# Shared configuration: imports, environment settings, constants, and plot style.
import os
import warnings

# Suppress library warnings in experiment output.
warnings.filterwarnings("ignore")

# Limit numerical libraries to one thread for reproducible runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Use a non-interactive Matplotlib backend and a local font cache.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Core numerical, data, and plotting libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Gaussian density used for the marginal likelihood.
from scipy.stats import multivariate_normal
# Scikit-learn tools for clustering, baselines, metrics, splits, and scaling.
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC

# Global seed, number of CV folds, and expected class labels.
RANDOM_STATE = 42
N_SPLITS = 5
EXPECTED_CLASSES = [0, 1, 2]
# Folder for the core experimental outputs.
OUTPUT_DIR = "simulation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default figure style for all plots.
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
