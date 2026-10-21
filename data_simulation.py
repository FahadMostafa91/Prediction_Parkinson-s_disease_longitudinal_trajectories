
# Shared numerical and data libraries from the project configuration.
from config import np, pd


# Generates sparse, irregular longitudinal data from three quadratic trajectory classes.
class LongitudinalSimulator:

    # Store class sizes, visit design, noise level, missing rate, and random seed.
    def __init__(
        self,
        n_samples_per_class=None,
        time_range=(0.0, 2.0),
        n_obs_range=(4, 8),
        noise_std=0.45,
        missing_rate=0.10,
        random_state=42,
    ):
        # Default class sizes follow a 1:2:3 ratio (N = 300).
        self.n_samples = n_samples_per_class or {0: 50, 1: 100, 2: 150}
        self.time_range = time_range
        self.n_obs_range = n_obs_range
        self.noise_std = noise_std
        self.missing_rate = missing_rate
        self.rng = np.random.default_rng(random_state)

        # Class-specific quadratic mean coefficients: intercept, linear, quadratic.
        self.coeffs = {
            0: [0.0, -0.50, 0.35],
            1: [0.0, -2.40, 1.70],
            2: [0.0, 2.30, -0.95],
        }

    # Simulate one subject's visits for a given class.
    def generate_subject(self, subject_id, class_label):
        # Draw the number of visits and sorted irregular visit times.
        n_obs = int(self.rng.integers(self.n_obs_range[0], self.n_obs_range[1] + 1))
        times = np.sort(self.rng.uniform(self.time_range[0], self.time_range[1], n_obs))
        beta = self.coeffs[class_label]

        # Subject-level random intercept and random slope.
        random_intercept = self.rng.normal(0.0, 0.70)
        random_slope = self.rng.normal(0.0, 0.25)
        mean_curve = beta[0] + beta[1] * times + beta[2] * times**2
        # Response = class mean curve + random effects + Gaussian measurement noise.
        y = mean_curve + random_intercept + random_slope * times + self.rng.normal(0.0, self.noise_std, n_obs)

        df = pd.DataFrame({"id": subject_id, "time": times, "y": y, "class": int(class_label)})
        # Randomly drop a fraction of visits while keeping the minimum visit count.
        if self.missing_rate > 0 and len(df) > self.n_obs_range[0]:
            n_drop = int(np.floor(len(df) * self.missing_rate))
            if n_drop > 0 and len(df) - n_drop >= self.n_obs_range[0]:
                drop_idx = self.rng.choice(df.index.to_numpy(), size=n_drop, replace=False)
                df = df.drop(drop_idx)
        return df

    # Generate all subjects with sequential IDs and combine them into one table.
    def simulate(self):
        frames = []
        subject_id = 0
        for class_label, n_subjects in self.n_samples.items():
            for _ in range(n_subjects):
                frames.append(self.generate_subject(subject_id, class_label))
                subject_id += 1
        return pd.concat(frames, ignore_index=True)
