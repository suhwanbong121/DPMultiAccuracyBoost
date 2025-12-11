import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import math


def split_data(data, ratio=[1, 0, 0], idxs=None):
    """
    Split data into train/val/test according to ratio.
    data: array or list/tuple of arrays with same first dimension.
    """
    if idxs is None:
        num = data[0].shape[0] if isinstance(data, (list, tuple)) else data.shape[0]
        idx = np.arange(num)
        np.random.shuffle(idx)
        idx_train = idx[:int(ratio[0] * num)]
        idx_val = idx[int(ratio[0] * num):int((ratio[0] + ratio[1]) * num)]
        idx_test = idx[int((ratio[0] + ratio[1]) * num):]
    else:
        idx_train, idx_val, idx_test = idxs

    train, val, test = [], [], []
    if isinstance(data, (list, tuple)):
        for counter in range(len(data)):
            train.append(data[counter][idx_train])
            val.append(data[counter][idx_val])
            test.append(data[counter][idx_test])
    else:
        train = data[idx_train]
        val = data[idx_val]
        test = data[idx_test]
    return (train, val, test), (idx_train, idx_val, idx_test)


def get_session(number=None):
    """
    Create a TF1 session with GPU memory growth.
    """
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    if number is not None:
        device = "/gpu:{}".format(number)
        # device 변수를 실제로 사용하려면 with tf.device(device): 를 밖에서 써야 함.
    return tf.Session(config=config_gpu)


def read_images(files):
    """
    Read a list of image file paths and return numpy array of RGB images (160x160).
    """
    ims = []
    for file in files:
        image_rgb = cv2.imread(file)
        if image_rgb is None:
            continue
        if image_rgb.shape[0] != image_rgb.shape[1]:
            image_rgb = image_rgb[29:-29, 9:-9, ::-1]
        else:
            cut = 1
            image_rgb = cv2.resize(image_rgb[cut:-cut, cut:-cut, :], (160, 160))[:, :, ::-1]
        ims.append(image_rgb)
    return np.array(ims)


# From Facenet Library:
class ImageClass:
    """Stores the paths to images for a given class."""
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    """
    Load dataset structure where subdirectories correspond to classes.
    """
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [p for p in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, p))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    """
    From Facenet library: list all image paths in a directory.
    """
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_image_paths_and_labels(dataset):
    """
    Flatten dataset into (image_paths, labels) lists.
    """
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


# ============================================================
# DP-SGD AUDITOR (FULL-BATCH, THEORETICALLY DP)
# ============================================================

class DPLinearAuditor(object):
    """
    Simple full-batch DP-SGD linear regression auditor for multiaccuracy.

    h(x) = w^T x

    This implementation matches the theoretical analysis:

    - Per-example gradients are clipped to L2 norm <= C.
    - We average over all n examples (full batch).
    - At each step s we add Gaussian noise with std sigma_s chosen from
      the Gaussian mechanism so that the step is (epsilon_s, delta_s)-DP.
    - Over K steps, by basic composition, total privacy is
      (sum epsilon_s, sum delta_s).

    In practice, we will call fit(X, r, epsilon_audit_round, delta_audit_round)
    and split epsilon_audit_round, delta_audit_round equally across K steps.
    """

    def __init__(self,
                 feature_dim,
                 learning_rate=0.1,
                 num_steps=200,
                 clipping_norm=1.0,
                 seed=42):
        self.d = feature_dim
        self.lr = learning_rate
        self.num_steps = num_steps
        self.C = clipping_norm          # gradient clipping norm
        self.rng = np.random.RandomState(seed)
        # initialize weights
        self.w = np.zeros(self.d, dtype=np.float32)

    def _clip_gradients(self, grads):
        """
        grads: array of shape (n, d)
        Clip each per-example gradient to L2 norm <= C.
        """
        norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, self.C / norms)
        return grads * scale

    def fit(self, X, r, epsilon_audit_round, delta_audit_round):
        """
        Train auditor with full-batch DP-SGD.

        X: shape (n, d)  – features (e.g., embeddings)
        r: shape (n,)    – residuals f_t(x) - y
        epsilon_audit_round: total privacy epsilon budget for this round (audit part)
        delta_audit_round: total privacy delta budget for this round (audit part)

        We split the round budget equally across num_steps:
            epsilon_s = epsilon_audit_round / num_steps
            delta_s   = delta_audit_round   / num_steps

        At each step we apply a Gaussian mechanism to the average clipped gradient
        with sensitivity 2C/n.
        """
        n, d = X.shape
        assert d == self.d

        if n == 0:
            return

        if epsilon_audit_round <= 0 or delta_audit_round <= 0 or delta_audit_round >= 1:
            raise ValueError("epsilon_audit_round must be > 0 and delta_audit_round in (0,1)")

        eps_step = float(epsilon_audit_round) / float(self.num_steps)
        delta_step = float(delta_audit_round) / float(self.num_steps)

        # Sensitivity of average clipped gradient: 2C/n (Lemma)
        sensitivity = 2.0 * self.C / float(n)

        for step in range(self.num_steps):
            # Full-batch gradients
            preds = X.dot(self.w)          # (n,)
            per_example_grads = ((preds - r)[:, None] * X)  # (n, d)

            # clip per-example gradients
            clipped = self._clip_gradients(per_example_grads)

            # average
            grad_mean = np.mean(clipped, axis=0)  # (d,)

            # Gaussian noise scale for this step:
            # sigma_step >= sensitivity * sqrt(2 log(1.25/delta_step)) / eps_step
            sigma_step = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_step))) / eps_step

            # add Gaussian noise
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


# ============================================================
# DIFFERENTIALLY PRIVATE CORRELATION QUERY
# ============================================================

def compute_noisy_correlation(auditor,
                              X,
                              residuals,
                              clipping_bound,
                              epsilon_corr,
                              delta_corr,
                              seed=42):
    """
    Implement noisy correlation query:

        Delta_t = E[ h_t(x) * (f_t(x) - y) ]

    with Gaussian noise calibrated for (epsilon_corr, delta_corr)-DP.

    Args:
        auditor: Trained DPLinearAuditor instance (fixed model)
        X: Features (n, d)
        residuals: Residuals f_t(x) - y (n,)
        clipping_bound: Clipping bound B for the scores
        epsilon_corr: Privacy budget epsilon for correlation query
        delta_corr: Privacy budget delta for correlation query
        seed: Random seed for noise generation

    Returns:
        Noisy correlation estimate hat{Delta}_t.
    """
    n = len(X)
    if n == 0:
        return 0.0

    if epsilon_corr <= 0 or delta_corr <= 0 or delta_corr >= 1:
        raise ValueError("epsilon_corr must be > 0 and delta_corr must be in (0, 1)")

    # Compute h_t(x_i) for all examples
    h_predictions = auditor.predict(X)  # (n,)

    # Per-example scores c_i = h_t(x_i) * (f_t(x_i) - y_i)
    scores = h_predictions * residuals  # (n,)

    # Clip scores to [-B, B]
    clipped_scores = np.clip(scores, -clipping_bound, clipping_bound)

    # Average
    avg_correlation = np.mean(clipped_scores)

    # Sensitivity of average clipped score: 2B / n
    sensitivity = 2.0 * clipping_bound / float(n)

    # Noise scale for Gaussian mechanism
    sigma_corr = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta_corr))) / epsilon_corr

    # Add Gaussian noise
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma_corr)

    noisy_correlation = avg_correlation + noise
    return noisy_correlation


# ============================================================
# SIMPLE PRIVACY ACCOUNTANT
# ============================================================

class PrivacyAccountant:
    """
    Simple privacy accountant to track (epsilon, delta) budget across rounds.

    We assume that each round t uses:
        epsilon_t/2, delta_t/2 for auditor training (DP-SGD)
        epsilon_t/2, delta_t/2 for correlation query

    and we require:
        sum_t epsilon_t <= total_epsilon
        sum_t delta_t   <= total_delta.
    """

    def __init__(self, total_epsilon, total_delta):
        """
        Initialize privacy accountant.

        Args:
            total_epsilon: Total privacy budget epsilon
            total_delta: Total privacy budget delta
        """
        self.total_epsilon = float(total_epsilon)
        self.total_delta = float(total_delta)
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        # Each entry: (epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t)
        self.rounds = []

    def allocate_round(self, epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t):
        """
        Allocate privacy budget for one round, splitting between auditor and correlation.

        Args:
            epsilon_audit_t: Privacy budget for auditor training in this round
            epsilon_corr_t: Privacy budget for correlation query in this round
            delta_audit_t: Privacy budget for auditor training in this round
            delta_corr_t: Privacy budget for correlation query in this round

        Returns:
            True if allocation is successful, False if budget would be exceeded.
        """
        epsilon_t = float(epsilon_audit_t) + float(epsilon_corr_t)
        delta_t = float(delta_audit_t) + float(delta_corr_t)

        if self.consumed_epsilon + epsilon_t > self.total_epsilon:
            return False
        if self.consumed_delta + delta_t > self.total_delta:
            return False

        self.consumed_epsilon += epsilon_t
        self.consumed_delta += delta_t
        self.rounds.append((epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t))
        return True

    def allocate_round_split(self, epsilon_t, delta_t):
        """
        Convenience method: allocate a round budget with equal split:

            epsilon_audit_t = epsilon_t / 2
            epsilon_corr_t  = epsilon_t / 2
            delta_audit_t   = delta_t / 2
            delta_corr_t    = delta_t / 2

        Args:
            epsilon_t: Total epsilon for this round
            delta_t: Total delta for this round

        Returns:
            True if allocation is successful, False otherwise.
        """
        epsilon_t = float(epsilon_t)
        delta_t = float(delta_t)
        epsilon_audit_t = epsilon_t / 2.0
        epsilon_corr_t = epsilon_t / 2.0
        delta_audit_t = delta_t / 2.0
        delta_corr_t = delta_t / 2.0
        return self.allocate_round(epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t)

    def get_remaining_budget(self):
        """Get remaining privacy budget as (epsilon, delta)."""
        return (self.total_epsilon - self.consumed_epsilon,
                self.total_delta - self.consumed_delta)

    def get_consumed_budget(self):
        """Get consumed privacy budget as (epsilon, delta)."""
        return (self.consumed_epsilon, self.consumed_delta)

    def can_allocate(self, epsilon_t, delta_t):
        """
        Check if budget can be allocated without actually allocating.

        Args:
            epsilon_t: Total epsilon for the round
            delta_t: Total delta for the round
        """
        epsilon_t = float(epsilon_t)
        delta_t = float(delta_t)
        return (self.consumed_epsilon + epsilon_t <= self.total_epsilon and
                self.consumed_delta + delta_t <= self.total_delta)

    def can_allocate_split(self, epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t):
        """
        Check if budget can be allocated for a round with an explicit split.
        """
        epsilon_t = float(epsilon_audit_t) + float(epsilon_corr_t)
        delta_t = float(delta_audit_t) + float(delta_corr_t)
        return self.can_allocate(epsilon_t, delta_t)
