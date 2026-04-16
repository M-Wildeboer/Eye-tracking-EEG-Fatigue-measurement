"""
Beginner-friendly walkthrough of this script
===========================================

Goal
- We want to classify each search trial as either:
    0 = random search
    1 = systematic search

What is a trial?
- A trial is a sequence of gaze points over time on an image.
- Each point has x and y position.
- Different trials have different lengths (different number of points).

Why feature extraction?
- Most basic machine-learning models expect a fixed-size numeric input per sample.
- But trials are variable-length sequences.
- So we convert each trial into the same set of summary measurements (features).

High-level pipeline
1. Load raw CSV files -> sys, random (& data)
2. Group rows into time-ordered trials (per trial num)
3. Convert each trial into a fixed feature list.
4. Split train/test with class balance. (equal per class 80-20)
5. Train logistic regression; kansberekening van class probabilities.
6. Evaluate accuracy of test data
7. Predict on data/data.csv and export output.csv. <- confidence + prob(sys) per trial
8. Estimate feature importance. (boxplot)

Model intuition (very simple)
- Logistic regression computes a weighted sum of features.
- If the weighted sum is high, prediction moves toward class 1.
- If low, prediction moves toward class 0.
- "Learning" means finding feature weights that best separate classes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the labelled training data and the AOI file
random_raw = pd.read_csv('data/ran.csv')
system_raw = pd.read_csv('data/sys.csv')
aoi = pd.read_csv('data/aoi.csv')

# Data heads are: Exp,Part,trial_index,Stim,time,xL,yL,xL_pro,yL_pro,xR,yR,xR_pro,yR_pro,xF,yF,xF_pro,yF_pro,inside_stimulus
# data heads for aoi: Stimulus,AOI,x,y,w,h

# Group rows into trials, preserving the time order inside each trial.
trial_columns = ['Exp', 'Part', 'trial_index']
random_trials = [trial.sort_values('time').reset_index(drop=True) for _, trial in random_raw.groupby(trial_columns)]
system_trials = [trial.sort_values('time').reset_index(drop=True) for _, trial in system_raw.groupby(trial_columns)]

print(f"Total random trials: {len(random_trials)}")
print(f"Total system trials: {len(system_trials)}")

# Optional debug print: how many gaze points each trial contains.
SHOW_TRIAL_POINT_COUNTS = False
if SHOW_TRIAL_POINT_COUNTS:
    for i, trial in enumerate(random_trials):
        print(f"Random trial {i} has {len(trial)} points")

    for i, trial in enumerate(system_trials):
        print(f"System trial {i} has {len(trial)} points")

def extract_trial_features(trial):
    """
    Convert one trial (a time-ordered list of gaze points) into fixed numeric features.

    - A trial is a path of (x, y) points over time on the image.
    - The model cannot directly compare paths of different lengths.
    - So we summarize each path with a set of measurements that describe:
        1) how much of the image was covered,
        2) how far the gaze moved,
        3) how smooth or zig-zag the movement was.

    Returned feature order (12 values):
    1.  x_std: horizontal spread
    2.  y_std: vertical spread
    3.  bbox_w: width of the explored area
    4.  bbox_h: height of the explored area
    5.  bbox_area: area of explored rectangle
    6. mean_step: average movement distance between consecutive points
    7. std_step: variability of movement distance
    8. step_75_iqr: largest single movement jump of IQR75
    9. mean_abs_dx: average horizontal movement per step
    10. mean_abs_dy: average vertical movement per step
    11. mean_direction_change: average turning amount between steps
    12. std_direction_change: variability of turning amount
        """
    xs = trial['xF'].to_numpy(dtype=float)
    ys = trial['yF'].to_numpy(dtype=float)

    if len(xs) == 0:
                return np.zeros(16, dtype=float)

    # Calculate movement between consecutive gaze points
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_dist = np.sqrt(dx ** 2 + dy ** 2)

    sorted_step_dist = np.sort(step_dist)
    
    # Calculate coverage of the viewed area
    bbox_w = xs.max() - xs.min()
    bbox_h = ys.max() - ys.min()
    bbox_area = bbox_w * bbox_h

    # Calculate other summary measures of scanpath movement
    total_path = step_dist.sum() if len(step_dist) else 0.0
    mean_step = step_dist.mean() if len(step_dist) else 0.0
    std_step = step_dist.std() if len(step_dist) else 0.0
    max_step = step_dist.max() if len(step_dist) else 0.0
    step_75_iqr = sorted_step_dist[int(0.75 * len(sorted_step_dist))] if len(sorted_step_dist) else 0.0
    mean_abs_dx = np.abs(dx).mean() if len(dx) else 0.0
    mean_abs_dy = np.abs(dy).mean() if len(dy) else 0.0

    # Calculate how much the gaze changes direction over time
    if len(dx) > 1:
        direction = np.arctan2(dy, dx)
        direction_change = np.abs(np.diff(direction))
        mean_direction_change = direction_change.mean()
        std_direction_change = direction_change.std()
    else:
        mean_direction_change = 0.0
        std_direction_change = 0.0

    # Final trial summary vector used by the classifier.
    features = np.array([
        #len(xs),
        #xs.mean(),
        #ys.mean(),
        xs.std(),
        ys.std(),
        bbox_w,
        bbox_h,
        bbox_area,
        #total_path,
        mean_step,
        std_step,
        #max_step,
        step_75_iqr,
        mean_abs_dx,
        mean_abs_dy,
        mean_direction_change,
        std_direction_change,
    ], dtype=float)

    return features


FEATURE_NAMES = [
    #'n_points',
    #'x_mean',
    #'y_mean',
    'x_std',
    'y_std',
    'bbox_w',
    'bbox_h',
    'bbox_area',
    #'total_path',
    'mean_step',
    'std_step',
    #'max_step',
    'step_75_iqr',
    'mean_abs_dx',
    'mean_abs_dy',
    'mean_direction_change',
    'std_direction_change',
]


def save_classification_report_table(y_true, y_pred, y_proba, output_path, brier=None, logloss=None):
    """
    Save a nicely formatted classification report table as a PNG image.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=['random', 'system'],
        zero_division=0,
        output_dict=True,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)

    # Confidence = probability assigned to the predicted class for each trial.
    confidences = y_proba[np.arange(len(y_pred)), y_pred]

    rows = ['random', 'system', 'accuracy', 'macro avg', 'weighted avg']
    row_data = []
    for row_name in rows:
        if row_name == 'accuracy':
            row_data.append([
                '',
                '',
                f"{report_dict[row_name]:.3f}",
                int(report_dict['macro avg']['support']),
                f"{confidences.mean():.3f}",
                f"{confidences.std():.3f}",
            ])
        elif row_name in ('macro avg', 'weighted avg'):
            row_data.append([
                f"{report_dict[row_name]['precision']:.3f}",
                f"{report_dict[row_name]['recall']:.3f}",
                f"{report_dict[row_name]['f1-score']:.3f}",
                int(report_dict[row_name]['support']),
                f"{confidences.mean():.3f}",
                f"{confidences.std():.3f}",
            ])
        else:
            cls_idx = 0 if row_name == 'random' else 1
            cls_mask = y_true == cls_idx
            cls_conf = confidences[cls_mask]
            cls_conf_mean = cls_conf.mean() if len(cls_conf) else 0.0
            cls_conf_std = cls_conf.std() if len(cls_conf) else 0.0
            row_data.append([
                f"{report_dict[row_name]['precision']:.3f}",
                f"{report_dict[row_name]['recall']:.3f}",
                f"{report_dict[row_name]['f1-score']:.3f}",
                int(report_dict[row_name]['support']),
                f"{cls_conf_mean:.3f}",
                f"{cls_conf_std:.3f}",
            ])

    # Keep figure height readable even if row count changes.
    fig_h = max(3.8, 0.68 * (len(rows) + 1.8))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.axis('off')

    title = 'Classification Report'
    metrics = []
    if brier is not None:
        metrics.append(f"Brier={brier:.4f}")
    if logloss is not None:
        metrics.append(f"LogLoss={logloss:.4f}")
    if metrics:
        title += " | " + " | ".join(metrics)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    # Create a table image of the model's test performance
    table = ax.table(
        cellText=row_data,
        rowLabels=rows,
        colLabels=['precision', 'recall', 'f1-score', 'support', 'mean_conf', 'std_conf'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1f2937')
            cell.set_text_props(weight='bold', color='white')
        elif c == -1:
            cell.set_facecolor('#e5e7eb')
            cell.set_text_props(weight='bold', color='#111827')
        else:
            cell.set_facecolor('#f9fafb' if r % 2 == 1 else 'white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved evaluation table image to {output_path}")


def plot_bootstrap_feature_importance(X_train, y_train, feature_names, n_bootstrap=200):
    """
    Estimate feature importance stability with bootstrap resampling.

    What this does:
    - Re-trains the model many times on slightly different resampled training sets.
    - Records each feature's absolute coefficient each time.
    - Builds a boxplot per feature to show how stable/unstable its influence is.

    Why useful:
    - With small data, one single model fit can be misleading.
    - The distribution across resamples gives a more honest view of uncertainty.
    """
    rng = np.random.RandomState(42)
    coefficient_samples = []

    class0_idx = np.where(y_train == 0)[0]
    class1_idx = np.where(y_train == 1)[0]

    if len(class0_idx) < 2 or len(class1_idx) < 2:
        print("Not enough samples per class for bootstrap importance plot.")
        return

    for _ in range(n_bootstrap):
        # Stratified bootstrap: resample each class with replacement.
        boot0 = rng.choice(class0_idx, size=len(class0_idx), replace=True)
        boot1 = rng.choice(class1_idx, size=len(class1_idx), replace=True)
        boot_idx = np.concatenate([boot0, boot1])

        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]

        coeff_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ])
        coeff_model.fit(X_boot, y_boot)

        coefs = np.abs(coeff_model.named_steps['clf'].coef_[0])
        coefficient_samples.append(coefs)

    coefficient_samples = np.asarray(coefficient_samples)

    # Save ranked mean absolute coefficients for easy tabular review.
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_coef': coefficient_samples.mean(axis=0),
        'median_abs_coef': np.median(coefficient_samples, axis=0),
        'std_abs_coef': coefficient_samples.std(axis=0),
    }).sort_values('mean_abs_coef', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    print("Saved ranked feature importance table to feature_importance.csv")

    # Horizontal boxplot for readability with many feature names.
    order = np.argsort(coefficient_samples.mean(axis=0))
    ordered_data = [coefficient_samples[:, i] for i in order]
    ordered_names = [feature_names[i] for i in order]

    plt.figure(figsize=(11, 7))
    plt.boxplot(ordered_data, vert=False, labels=ordered_names)
    plt.title('Feature Importance Distribution (Bootstrap Logistic Coefficients)')
    plt.xlabel('Absolute standardized coefficient (higher = more influential)')
    plt.tight_layout()
    plt.savefig('feature_importance_boxplot.png', dpi=200)
    plt.close()
    print("Saved feature importance boxplot to feature_importance_boxplot.png")


def build_dataset(trials, label):
    """
    Turn many trials into model-ready arrays.

    Returns
    - features: one fixed-length feature vector per trial
    - labels: class label per trial (0 random / 1 systematic)
    - groups: participant identifier used to reduce data leakage in splits
    """
    features = [extract_trial_features(trial) for trial in trials]
    labels = [label] * len(trials)
    groups = [f"{trial['Exp'].iloc[0]}::{trial['Part'].iloc[0]}" for trial in trials]
    return features, labels, groups


X_random, y_random, g_random = build_dataset(random_trials, 0)
X_system, y_system, g_system = build_dataset(system_trials, 1)

X = X_random + X_system
y = y_random + y_system
groups = g_random + g_system

# Train a classifier on whole trials, then test on whole trials.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import brier_score_loss, classification_report, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=int)
groups = np.asarray(groups)

def split_by_class_with_groups(y, groups, test_size=0.2, random_state=27):
    """
    Split data so both classes appear in both train and test sets.

    Extra protection:
    - Tries to keep participant groups separated where possible.
    - This reduces "leakage" (model seeing the same person in train and test),
        which can make results look unrealistically good.
    """
    rng = np.random.RandomState(random_state)
    train_indices = []
    test_indices = []

    for class_label in np.unique(y):
        cls_idx = np.where(y == class_label)[0]
        if len(cls_idx) < 2:
            raise ValueError(f"Not enough trials for class {class_label} to split train/test")

        cls_groups = groups[cls_idx]
        unique_cls_groups = np.unique(cls_groups)

        if len(unique_cls_groups) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + int(class_label))
            rel_train, rel_test = next(splitter.split(cls_idx, groups=cls_groups))
            cls_train = cls_idx[rel_train]
            cls_test = cls_idx[rel_test]
        else:
            # Fallback when one class only has one participant/group.
            shuffled = rng.permutation(cls_idx)
            n_test = max(1, int(round(len(shuffled) * test_size)))
            n_test = min(n_test, len(shuffled) - 1)
            cls_test = shuffled[:n_test]
            cls_train = shuffled[n_test:]

        if len(cls_train) == 0 or len(cls_test) == 0:
            raise ValueError(f"Failed to create non-empty split for class {class_label}")

        train_indices.append(cls_train)
        test_indices.append(cls_test)

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)
    return train_idx, test_idx

# Split the labelled trials into training data and test data
train_idx, test_idx = split_by_class_with_groups(y, groups, test_size=0.2)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train trials: {len(X_train)} | Test trials: {len(X_test)}")
print(f"Unique train groups: {len(np.unique(groups[train_idx]))}")
print(f"Unique test groups: {len(np.unique(groups[test_idx]))}")

base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
])

# Why StandardScaler?
# - Features have very different numeric scales (for example x_mean vs bbox_area).
# - Scaling puts features on comparable ranges, which helps logistic regression.

# Why class_weight='balanced'?
# - If class counts are uneven, this gives extra weight to the smaller class.
# - It reduces bias toward the majority class.

# Calibrate probabilities to reduce overconfident 0/1 outputs.
class_counts = np.bincount(y_train)
min_class_count = class_counts.min() if len(class_counts) > 1 else 0
if min_class_count >= 3:
    calibration_cv = min(5, int(min_class_count))
    model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=calibration_cv)
else:
    print("Too few trials per class for calibration; using uncalibrated probabilities.")
    model = base_model

# What is calibration?
# - Raw model probabilities can be overconfident (for example, too many 0.99 values).
# - Calibration adjusts probability outputs so they better match real frequencies.

# Fit the classifier on the training data
model.fit(X_train, y_train)

# Test the classifier on the held-out test set
proba = model.predict_proba(X_test)
pred = model.predict(X_test)
brier = brier_score_loss(y_test, proba[:, 1])
logloss = log_loss(y_test, proba, labels=[0, 1])

print(classification_report(y_test, pred, labels=[0, 1], target_names=['random', 'system'], zero_division=0))
print(f"Brier score (system): {brier:.4f}")
print(f"Log loss: {logloss:.4f}")

# confidences of the models test set:
print(f"Test set predictions with confidence: {list(zip(pred, proba[:, 1]))}")

save_classification_report_table(y_test, pred, proba, 'output_table.png', brier=brier, logloss=logloss)

# Metric intuition:
# - classification_report: precision/recall/F1 per class.
# - Brier score: lower is better; compares predicted probabilities to outcomes.
# - Log loss: lower is better; strongly penalizes confident wrong predictions.

print("Per-trial uncertainty (lower confidence = more uncertain):")
for i, (p, yhat, ytrue) in enumerate(zip(proba, pred, y_test)):
    confidence = float(p[yhat])
    uncertainty = 1.0 - confidence
    print(
        f"trial {i}: true={ytrue}, pred={yhat}, "
        f"confidence={confidence:.3f}, uncertainty={uncertainty:.3f}, "
        f"P(random)={p[0]:.3f}, P(system)={p[1]:.3f}"
    )

# Estimate which features most influence model decisions and visualize uncertainty in that ranking.
plot_bootstrap_feature_importance(X_train, y_train, FEATURE_NAMES, n_bootstrap=200)


# Final inference step:
# - Load trials from data/data.csv (new/unlabeled dataset)
# - Predict class probabilities per trial
# - Save prediction + uncertainty to output.csv for later analysis/reporting
data_raw = pd.read_csv('data/data.csv')
data_trials = [trial.sort_values('time').reset_index(drop=True) for _, trial in data_raw.groupby(trial_columns)]
output_rows = []

total_conf = 0

for i, trial in enumerate(data_trials):
    features = extract_trial_features(trial)
    features = features.reshape(1, -1)
    proba = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    confidence = float(proba[pred])
    uncertainty = 1.0 - confidence
    total_conf += confidence


    output_rows.append({
        'trial_index': i,
        'predicted_label': pred,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'P_random': proba[0],
        'P_system': proba[1],
    })

# Save classifier predictions for all experimental trials
output_df = pd.DataFrame(output_rows)
output_df.to_csv('output.csv', index=False)

print(f"Average confidence: {total_conf/len(data_trials):.4f}")








