import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def split_data(data, ratio = [1, 0, 0], idxs=None):
    
    if idxs is None:
        num = data[0].shape[0] if type(data)==list  or type(data)==tuple else data.shape[0]
        idx = np.arange(num)
        np.random.shuffle(idx)
        idx_train = idx[:int(ratio[0]*num)]
        idx_val = idx[int(ratio[0]*num):int((ratio[0]+ratio[1])*num)]
        idx_test = idx[int((ratio[0]+ratio[1])*num):]
    else:
        idx_train,idx_val,idx_test = idxs
    train, val, test = [], [], []
    if type(data)==list or type(data)==tuple:
        for counter in range(len(data)):
            train.append(data[counter][idx_train])
            val.append(data[counter][idx_val])
            test.append(data[counter][idx_test])
    else:
        train = data[idx_train]
        val = data[idx_val]
        test = data[idx_test]
    return (train,val,test),(idx_train,idx_val,idx_test)

def get_session(number=None):
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    if number is not None:
        device = "/gpu{}".format(number)
    return tf.Session(config=config_gpu)

def read_images(files):
    ims = []
    for file in files:
        image_rgb = cv2.imread(file)
        if image_rgb.shape[0]!=image_rgb.shape[1]:
            image_rgb = image_rgb[29:-29, 9:-9,::-1]
        else:
            cut = 1
            image_rgb = cv2.resize(image_rgb[cut:-cut,cut:-cut,:], (160,160))[:,:,::-1]
        ims.append(image_rgb)
    return np.array(ims)

#From Facenet Library:
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
    
def get_dataset(path, has_class_directories=True):
    
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    #From Facenet Library
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

class DPLinearAuditor(object):
    """
    Simple DP-SGD linear regression auditor for multiaccuracy.
    h(x) = w^T x
    """

    def __init__(self,
                 feature_dim,
                 learning_rate=0.1,
                 num_steps=200,
                 batch_size=128,
                 clipping_norm=1.0,
                 noise_multiplier=1.0,
                 seed=42):
        self.d = feature_dim
        self.lr = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.C = clipping_norm          # gradient clipping norm
        self.noise_multiplier = noise_multiplier
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

    def fit(self, X, r):
        """
        X: shape (n, d)  – features (e.g., embeddings)
        r: shape (n,)    – residuals f_t(x) - y
        We minimize 0.5 * (h(x) - r)^2 with DP-SGD.
        """
        n, d = X.shape
        assert d == self.d

        indices = np.arange(n)
    
        # Process multiple batches per epoch for better convergence
        num_batches_per_epoch = max(1, n // self.batch_size)
        
        for step in range(self.num_steps):
            self.rng.shuffle(indices)
            
            # Process all batches in this epoch
            for batch_idx_start in range(0, min(n, num_batches_per_epoch * self.batch_size), self.batch_size):
                batch_idx = indices[batch_idx_start:batch_idx_start + self.batch_size]
                if len(batch_idx) == 0:
                    continue
                    
                Xb = X[batch_idx]          # (B, d)
                rb = r[batch_idx]          # (B,)

                # forward
                preds = Xb.dot(self.w)     # (B,)
                # grad wrt w for each example: (pred - r) * x
                # shape: (B, d)
                per_example_grads = ((preds - rb)[:, None] * Xb)

                # clip per-example gradients
                clipped = self._clip_gradients(per_example_grads)

                # average
                grad_mean = np.mean(clipped, axis=0)  # (d,)

                # add Gaussian noise
                sigma = self.noise_multiplier * self.C / float(len(batch_idx))
                noise = self.rng.normal(loc=0.0, scale=sigma, size=self.d)

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
    Implement noisy correlation query as per Lemma 4.
    
    Computes differentially private correlation estimate
    
    Args:
        auditor: Trained DPLinearAuditor instance
        X: Features (n, d)
        residuals: Residuals f_t(x) - y (n,)
        clipping_bound: Clipping bound B
        epsilon: Privacy budget for correlation query 
        delta: Privacy budget for correlation query 
        seed: Random seed for noise generation
        
    Returns:
        Noisy correlation estimate 
    """
    import math
    
    n = len(X)
    if n == 0:
        return 0.0
    
    # Compute h_t(x_i) for all examples
    h_predictions = auditor.predict(X)  # (n,)
    
    # Compute per-example scores
    scores = h_predictions * residuals  # (n,)
    
    # Clip scores to [-B, B] 
    clipped_scores = np.clip(scores, -clipping_bound, clipping_bound)
    
    # Compute average
    avg_correlation = np.mean(clipped_scores)
    
    # Compute noise scale 
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("epsilon must be > 0, delta must be in (0, 1)")
    
    sensitivity = 2.0 * clipping_bound / float(n)  # From Lemma 3
    sigma_corr = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon
    
    # Add Gaussian noise
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma_corr)
    
    # Return noisy correlation estimate
    noisy_correlation = avg_correlation + noise
    
    return noisy_correlation

# PRIVACY ACCOUNTING FUNCTIONS 

def compute_epsilon_dp_sgd(noise_multiplier, dataset_size, num_steps, batch_size, delta):
    """
    Compute (epsilon, delta)-DP guarantee for DP-SGD using basic composition.
    
    
    Args:
        noise_multiplier: Noise multiplier used in DP-SGD 
        dataset_size: Total size of dataset (n)
        num_steps: Number of training steps (K)
        batch_size: Batch size used in training
        delta: Target delta value per step (delata_s)
        
    Returns:
        Estimated epsilon value for the entire DP-SGD procedure
    """
    import math
    
    
    
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    
    # Epsilon per step 
    epsilon_per_step = (2.0 / noise_multiplier) * math.sqrt(2.0 * math.log(1.25 / delta))
    
  
    epsilon_total = epsilon_per_step * num_steps
    
    return epsilon_total


def compute_epsilon_basic_composition(epsilon_per_round, num_rounds):
    """
    Compute total epsilon using basic composition.
    
    
    Args:
        epsilon_per_round: Epsilon consumed per round
        num_rounds: Number of rounds
        
    Returns:
        Total epsilon
    """
    return epsilon_per_round * num_rounds


def compute_delta_basic_composition(delta_per_round, num_rounds):
    """
    Compute total delta using basic composition.
    
    
    Args:
        delta_per_round: Delta consumed per round
        num_rounds: Number of rounds
        
    Returns:
        Total delta
    """
    return delta_per_round * num_rounds


class PrivacyAccountant:
    """
    tracks privacy budget consumption across multiple DP operations.
    
    """
    
    def __init__(self, total_epsilon, total_delta):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget 
            total_delta: Total privacy budget 
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.rounds = []  # Each entry: (epsilon_audit, epsilon_corr, delta_audit, delta_corr)
    
    def allocate_round(self, epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t):
        """
        Allocate privacy budget for one round, splitting between auditor and correlation.
        
        
        Args:
            epsilon_audit_t: Privacy budget for auditor training 
            epsilon_corr_t: Privacy budget for correlation query 
            delta_audit_t: Privacy budget for auditor training 
            delta_corr_t: Privacy budget for correlation query 
            
        Returns:
            True if allocation is successful, False if budget exceeded
        """
        epsilon_t = epsilon_audit_t + epsilon_corr_t
        delta_t = delta_audit_t + delta_corr_t
        
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
        Convenience method: Allocate round budget with equal split 
        
        Args:
            epsilon_t: Total privacy budget for this round 
            delta_t: Total privacy budget for this round 
            
        Returns:
            True if allocation is successful, False if budget exceeded
        """
        # Split equally
        epsilon_audit_t = epsilon_t / 2.0
        epsilon_corr_t = epsilon_t / 2.0
        delta_audit_t = delta_t / 2.0
        delta_corr_t = delta_t / 2.0
        
        return self.allocate_round(epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t)
    
    def get_remaining_budget(self):
        """Get remaining privacy budget as (epsilon, delta) tuple."""
        return (self.total_epsilon - self.consumed_epsilon,
                self.total_delta - self.consumed_delta)
    
    def get_consumed_budget(self):
        """Get consumed privacy budget as (epsilon, delta) tuple."""
        return (self.consumed_epsilon, self.consumed_delta)
    
    def can_allocate(self, epsilon_t, delta_t):
        """
        Check if budget can be allocated without actually allocating.
        
        Args:
            epsilon_t: Total epsilon for the round 
            delta_t: Total delta for the round 
        """
        return (self.consumed_epsilon + epsilon_t <= self.total_epsilon and
                self.consumed_delta + delta_t <= self.total_delta)
    
    def can_allocate_split(self, epsilon_audit_t, epsilon_corr_t, delta_audit_t, delta_corr_t):
        """Check if budget can be allocated with explicit split."""
        epsilon_t = epsilon_audit_t + epsilon_corr_t
        delta_t = delta_audit_t + delta_corr_t
        return self.can_allocate(epsilon_t, delta_t)

