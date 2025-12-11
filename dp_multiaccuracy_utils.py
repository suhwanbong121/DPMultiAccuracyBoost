"""
Differentially Private Multiaccuracy utilities.

Contains:
- DPLinearAuditor: simple DP-SGD style linear regression auditor
- compute_noisy_correlation: DP Gaussian mechanism for correlation estimate
- run_dp_multiaccuracy_boost: end-to-end DP multiaccuracy post-processing
"""

import numpy as np
import math
from sklearn.metrics import accuracy_score


# dp_multiaccuracy_utils.py

import numpy as np
import math

class DPLinearAuditor(object):
    """
    Simple DP-SGD linear regression auditor for multiaccuracy.
    h(x) = w^T x

    Noise is calibrated from (epsilon, delta) using the Gaussian mechanism
    and basic composition over num_steps.
    """

    def __init__(self,
                 feature_dim,
                 learning_rate=0.05,
                 num_steps=200,
                 batch_size=256,
                 clipping_norm=1.0,
                 seed=42):
        self.d = feature_dim
        self.lr = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.C = clipping_norm          # gradient clipping norm
        self.rng = np.random.RandomState(seed)
        # initialize weights
        self.w = np.zeros(self.d, dtype=np.float32)

    def _clip_gradients(self, grads):
        """
        grads: array of shape (batch_size, d)
        Clip each per-example gradient to L2 norm <= C.
        """
        norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, self.C / norms)
        return grads * scale

    def fit(self, X, r, epsilon, delta):
        """
        DP-SGD training with noise calibrated from (epsilon, delta).

        We use basic composition: epsilon and delta are divided
        equally over num_steps, and each step is a Gaussian mechanism
        on the averaged clipped gradient.

        Args:
            X: features, shape (n, d)
            r: residuals, shape (n,)
            epsilon: total epsilon for the auditor training in this round
            delta: total delta for the auditor training in this round
        """
        n, d = X.shape
        assert d == self.d

        if epsilon <= 0 or not (0 < delta < 1):
            raise ValueError("epsilon must be > 0 and delta in (0,1).")

        indices = np.arange(n)

        # Per-step privacy budget (basic composition)
        eps_step = epsilon / float(self.num_steps)
        del_step = delta / float(self.num_steps)

        if eps_step <= 0 or not (0 < del_step < 1):
            raise ValueError("Invalid per-step epsilon/delta.")

        # Sensitivity of averaged clipped gradient in a minibatch:
        # each per-example grad has norm <= C, so changing one example
        # changes the average by at most 2C / batch_size.
        sensitivity = 2.0 * self.C / float(self.batch_size)

        # Noise scale for each step (Gaussian mechanism)
        sigma_step = sensitivity * math.sqrt(2.0 * math.log(1.25 / del_step)) / eps_step

        for step in range(self.num_steps):
            self.rng.shuffle(indices)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if batch_idx.size == 0:
                    continue

                Xb = X[batch_idx]          # (B, d)
                rb = r[batch_idx]          # (B,)

                preds = Xb.dot(self.w)     # (B,)
                # per-example gradient for squared loss: (pred - r) * x
                per_example_grads = ((preds - rb)[:, None] * Xb)  # (B, d)

                # clip per-example gradients
                clipped = self._clip_gradients(per_example_grads)

                # average
                grad_mean = np.mean(clipped, axis=0)  # (d,)

                # add Gaussian noise with sigma_step
                noise = self.rng.normal(loc=0.0, scale=sigma_step, size=self.d)
                noisy_grad = grad_mean + noise

                # gradient descent update
                self.w -= self.lr * noisy_grad

    def predict(self, X):
        """
        Return h(x) = w^T x.
        X: shape (n, d)
        """
        return X.dot(self.w)



def compute_noisy_correlation(auditor, X, residuals, clipping_bound, epsilon, delta, seed=42):
    """
    Compute a differentially private estimate of the correlation

        E[ h_t(x) * (f_t(x) - y) ]

    using a Gaussian mechanism with clipping.

    We:
        - compute per-example scores c_i = h_t(x_i) * (f_t(x_i) - y_i),
        - clip them to [-B, B],
        - average,
        - add Gaussian noise with variance chosen from (epsilon, delta).

    This matches the Gaussian mechanism analysis for a scalar query.

    Args:
        auditor: Trained DPLinearAuditor instance.
        X: Feature matrix used for auditing, shape (n, d).
        residuals: Residuals f_t(x) - y, shape (n,).
        clipping_bound: Clipping bound B.
        epsilon: Privacy parameter epsilon for this correlation query.
        delta: Privacy parameter delta for this correlation query.
        seed: Random seed for the Gaussian noise.

    Returns:
        noisy_correlation: Scalar DP estimate of the correlation.
    """
    n = len(X)
    if n == 0:
        return 0.0

    if epsilon <= 0 or not (0.0 < delta < 1.0):
        raise ValueError("epsilon must be > 0 and delta must be in (0, 1).")

    # Auditor predictions
    h_predictions = auditor.predict(X)  # (n,)

    # Per-example scores
    scores = h_predictions * residuals  # (n,)

    # Clip scores to [-B, B]
    clipped_scores = np.clip(scores, -clipping_bound, clipping_bound)

    # Average of clipped scores
    avg_correlation = np.mean(clipped_scores)

    # Sensitivity of the average under clipping
    sensitivity = 2.0 * clipping_bound / float(n)

    # Noise scale (standard deviation) from Gaussian mechanism
    sigma_corr = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon

    # Add Gaussian noise
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma_corr)

    noisy_correlation = avg_correlation + noise
    return noisy_correlation


def multiplicative_weights_update(f, h, eta):
    """
    Perform a multiplicative-weights style update on probabilities f(x).

    Update:
        f_{t+1}(x) ∝ f_t(x) * exp(- eta * h_t(x))

    Then project to [0,1] by clipping.

    Args:
        f: Current probabilities, shape (n,).
        h: Auditor predictions, shape (n,).
        eta: Step size.

    Returns:
        Updated probabilities, shape (n,) clipped to [0,1].
    """
    updated = f * np.exp(-eta * h)
    updated = np.clip(updated, 0.0, 1.0)
    return updated


def run_dp_multiaccuracy_boost(
    X_train, y_train,
    X_audit, y_audit,
    X_test, y_test,
    base_model,
    T=5,
    eta=0.5,
    epsilon_round=1.0,
    delta_round=1e-5,
    clipping_grad=1.0,
    clipping_corr=1.0,
    auditor_lr=0.05,
    auditor_steps=200,
    auditor_batch_size=256,
    noise_multiplier=1.0,
    stop_threshold=0.01,
    seed=0,
):
    """
    Run DP multiaccuracy post-processing on top of a fixed base model.

    At each round t:
        1. Compute residuals r_t on the audit set.
        2. Train a DPLinearAuditor on (X_audit, r_t) using DP-SGD style updates.
        3. Compute a DP correlation estimate hat{Δ}_t using Gaussian mechanism.
        4. If | hat{Δ}_t | < stop_threshold (if provided), stop.
        5. Otherwise, update f_t on (train, audit, test) via multiplicative-weights.

    Args:
        X_train, y_train: Training data (only used to propagate updated f_t).
        X_audit, y_audit: Audit data used to fit auditor and compute correlation.
        X_test, y_test: Test data, used only for evaluation.
        base_model: Trained base model with predict_proba(X)[:,1].
        T: Maximum number of boosting rounds.
        eta: Step size for multiplicative-weights update.
        epsilon_round: Per-round epsilon for the DP correlation query.
        delta_round: Per-round delta for the DP correlation query.
        clipping_grad: Gradient clipping norm C for auditor DP-SGD.
        clipping_corr: Clipping bound B for correlation scores.
        auditor_lr: Learning rate for auditor.
        auditor_steps: Number of DP-SGD steps for auditor.
        auditor_batch_size: Mini-batch size for auditor training.
        noise_multiplier: Noise multiplier for DPLinearAuditor (DP-SGD).
        stop_threshold: If not None, stop when |Δ̂_t| < stop_threshold.
        seed: Random seed base.

    Returns:
        A dictionary with:
            "fT_train": final probabilities on train set.
            "fT_audit": final probabilities on audit set.
            "fT_test": final probabilities on test set.
            "history": {
                "delta_hat": list of noisy correlations per round,
                "acc_test": list of test accuracies per round,
            }
    """
    rng = np.random.RandomState(seed)

    # Initialize f_0 using base model probabilities
    f_train = base_model.predict_proba(X_train)[:, 1]
    f_audit = base_model.predict_proba(X_audit)[:, 1]
    f_test = base_model.predict_proba(X_test)[:, 1]

    history = {
        "delta_hat": [],
        "acc_test": [],
    }

    for t in range(T):
        # ------------------------------------------------------------------
        # 1. Compute residuals on audit set
        # ------------------------------------------------------------------
        r_audit = f_audit - y_audit  # f_t(x) - y

        # ------------------------------------------------------------------
        # 2. Train DP auditor on (X_audit, r_audit)
        # ------------------------------------------------------------------
        auditor = DPLinearAuditor(
            feature_dim=X_audit.shape[1],
            learning_rate=auditor_lr,
            num_steps=auditor_steps,
            batch_size=auditor_batch_size,
            clipping_norm=clipping_grad,
            noise_multiplier=noise_multiplier,
            seed=seed + t,
        )
        auditor.fit(X_audit, r_audit)

        # ------------------------------------------------------------------
        # 3. Compute DP correlation estimate hat{Δ}_t on audit set
        #     We split the round budget (epsilon_round, delta_round) evenly
        #     between auditor and correlation in the *theoretical* analysis.
        #     In this code, epsilon_round/2, delta_round/2 are used for the
        #     Gaussian mechanism; the DP-SGD part uses noise_multiplier as
        #     a proxy for privacy strength.
        # ------------------------------------------------------------------
        eps_corr = epsilon_round / 2.0
        delta_corr = delta_round / 2.0

        delta_hat_t = compute_noisy_correlation(
            auditor=auditor,
            X=X_audit,
            residuals=r_audit,
            clipping_bound=clipping_corr,
            epsilon=eps_corr,
            delta=delta_corr,
            seed=seed + 1000 + t,
        )

        history["delta_hat"].append(delta_hat_t)

        # ------------------------------------------------------------------
        # 4. Optional stopping rule
        # ------------------------------------------------------------------
        if stop_threshold is not None and abs(delta_hat_t) < stop_threshold:
            # No significant remaining multiaccuracy violation detected
            break

        # ------------------------------------------------------------------
        # 5. Update f_t on train / audit / test via multiplicative-weights
        # ------------------------------------------------------------------
        h_train = auditor.predict(X_train)
        h_audit = auditor.predict(X_audit)
        h_test = auditor.predict(X_test)

        f_train = multiplicative_weights_update(f_train, h_train, eta)
        f_audit = multiplicative_weights_update(f_audit, h_audit, eta)
        f_test = multiplicative_weights_update(f_test, h_test, eta)

        # Track test accuracy after this round
        y_test_pred = (f_test >= 0.5).astype(int)
        acc_test = accuracy_score(y_test, y_test_pred)
        history["acc_test"].append(acc_test)

    return {
        "fT_train": f_train,
        "fT_audit": f_audit,
        "fT_test": f_test,
        "history": history,
    }

def compute_noisy_correlation(h_vals, residuals, clipping_bound, epsilon, delta, seed=42):
    """
    Differentially private correlation estimate:

        E[ h(x) * (f_t(x) - y) ]

    implemented as a clipped average plus Gaussian noise.

    Args:
        h_vals: auditor predictions on audit set, shape (n,)
        residuals: f_t(x) - y on audit set, shape (n,)
        clipping_bound: clipping bound B
        epsilon: privacy budget for this correlation query
        delta: privacy budget for this correlation query
        seed: random seed

    Returns:
        Noisy correlation estimate (float).
    """
    import math
    rng = np.random.RandomState(seed)

    n = len(h_vals)
    if n == 0:
        return 0.0

    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("epsilon must be > 0 and delta in (0,1).")

    # Per-example scores
    scores = h_vals * residuals  # (n,)

    # Clip scores to [-B, B]
    clipped_scores = np.clip(scores, -clipping_bound, clipping_bound)

    # Average
    avg_corr = np.mean(clipped_scores)

    # Sensitivity of scalar average is 2B / n
    sensitivity = 2.0 * clipping_bound / float(n)

    # Gaussian mechanism noise scale
    sigma_corr = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    noise = rng.normal(loc=0.0, scale=sigma_corr)

    return avg_corr + noise


def sigmoid_multiaccuracy_update(probs, h_vals, eta):
    """
    Same as your multiaccuracy_update: update logits by -eta * h(x)
    and map back with sigmoid. Defined here to keep dp_multiaccuracy_utils
    self-contained; you can delete your local copy if you prefer.
    """
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p / (1.0 - p))
    new_logits = logits - eta * h_vals
    new_probs = 1.0 / (1.0 + np.exp(-new_logits))
    return new_probs


def run_dp_multiaccuracy_boost(
    X_train, y_train,
    X_audit, y_audit,
    X_test, y_test,
    base_model,
    T=5,
    eta=0.5,
    epsilon_round=1.0,
    delta_round=1e-5,
    clipping_grad=1.0,
    clipping_corr=1.0,
    auditor_lr=0.05,
    auditor_steps=200,
    auditor_batch_size=256,
    stop_threshold=0.002,
    seed=0,
):
    """
    DP multiaccuracy boosting with explicit (epsilon_round, delta_round)
    controlling both auditor-training noise and correlation-query noise.

    In each round t:
      - We split (epsilon_round, delta_round) evenly:
          * (epsilon_round/2, delta_round/2) for DP-SGD auditor training
          * (epsilon_round/2, delta_round/2) for the noisy correlation query
      - Train DPLinearAuditor with that budget.
      - Compute noisy correlation hat{Delta}_t under that budget.
      - If | hat{Delta}_t | < stop_threshold, stop.
      - Otherwise update f_t using the auditor predictions on audit/test.
    """
    rng = np.random.RandomState(seed)

    n_audit, d = X_audit.shape

    # Start from base-model probabilities
    f_audit = base_model.predict_proba(X_audit)[:, 1]
    f_test  = base_model.predict_proba(X_test)[:, 1]

    delta_hat_list = []

    for t in range(T):
        residuals = f_audit - y_audit  # (n_audit,)

        # Split per-round budget into audit vs correlation (1/2 each)
        eps_audit = epsilon_round / 2.0
        del_audit = delta_round / 2.0
        eps_corr  = epsilon_round / 2.0
        del_corr  = delta_round / 2.0

        # --- DP-SGD auditor training ---
        auditor = DPLinearAuditor(
            feature_dim=d,
            learning_rate=auditor_lr,
            num_steps=auditor_steps,
            batch_size=auditor_batch_size,
            clipping_norm=clipping_grad,
            seed=rng.randint(10**6),
        )
        auditor.fit(X_audit, residuals, epsilon=eps_audit, delta=del_audit)

        # Auditor predictions on audit and test
        h_audit = auditor.predict(X_audit)
        h_test  = auditor.predict(X_test)

        # --- DP noisy correlation query ---
        delta_hat = compute_noisy_correlation(
            h_vals=h_audit,
            residuals=residuals,
            clipping_bound=clipping_corr,
            epsilon=eps_corr,
            delta=del_corr,
            seed=rng.randint(10**6),
        )
        delta_hat_list.append(delta_hat)
        print(f"[DP MA] Round {t}: noisy correlation delta_hat = {delta_hat:.6f}")

        # Stop based on noisy correlation
        if abs(delta_hat) < stop_threshold:
            print(f"Stopping DP MA early at round {t} (|delta_hat| < {stop_threshold}).")
            break

        # Multiaccuracy update (post-processing) on both audit and test
        f_audit = sigmoid_multiaccuracy_update(f_audit, h_audit, eta)
        f_test  = sigmoid_multiaccuracy_update(f_test,  h_test,  eta)

    return {
        "fT_audit": f_audit,
        "fT_test":  f_test,
        "history": {
            "delta_hat": delta_hat_list,
        },
    }
