import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier

import os

os.makedirs("Figures", exist_ok=True)

CLINICAL_PATH = "StructuredData/clinical_data.json"
PATHOLOGICAL_PATH = "StructuredData/pathological_data.json"
BLOOD_PATH = "StructuredData/blood_data.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 40

TARGETS = ["hpv_association_p16", "primary_tumor_site", "grading"]

LEAKY_COLS = [
    "pT_stage",
    "pN_stage",
    "histologic_type",
    "perinodal_invasion",
    "lymphovascular_invasion_L",
    "vascular_invasion_V",
    "perineural_invasion_Pn",
    "resection_status",
    "resection_status_carcinoma_in_situ",
    "carcinoma_in_situ",
]

LOSS_WEIGHTS = {"hpv_out": 1.0, "site_out": 1.5, "grade_out": 1.0}


# Loading In Data ++++++++++++++++++++++++++++++++++
def load_json_to_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.json_normalize(data)


clinical = load_json_to_df(CLINICAL_PATH)
pathological = load_json_to_df(PATHOLOGICAL_PATH)
blood_raw = load_json_to_df(BLOOD_PATH)

blood_raw.columns = blood_raw.columns.str.strip()
clinical.columns = clinical.columns.str.strip()
pathological.columns = pathological.columns.str.strip()

blood = blood_raw.pivot_table(
    index="patient_id", columns="analyte_name", values="value", aggfunc="mean"
).reset_index()
blood.columns.name = None

df = clinical.merge(pathological, on="patient_id", how="inner").merge(
    blood, on="patient_id", how="inner"
)
print("Merged dataset shape:", df.shape)

df["grading"] = df["grading"].replace("hpv_association_p16", "G_unknown")

print("\n=== Target classes after fix ===")
for col in TARGETS:
    print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")

df = df.dropna(subset=TARGETS)
print(f"\nShape after dropna on targets: {df.shape}")

# Features ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
X = df.drop(columns=TARGETS + ["patient_id"] + LEAKY_COLS, errors="ignore")
y = df[TARGETS]

# Train Test Split ++++++++++++++++++++++++++++++++++++++++++++++++++++
unique_patients = df["patient_id"].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

train_mask = df["patient_id"].isin(train_patients)
test_mask = df["patient_id"].isin(test_patients)

X_train_raw = X[train_mask].copy()
X_test_raw = X[test_mask].copy()
y_train_raw = y[train_mask].copy()
y_test_raw = y[test_mask].copy()

# Target Encoding ++++++++++++++++++++++++++++++++++++++++++++++++++++++
encoders: dict[str, LabelEncoder] = {}
y_train_enc: dict[str, np.ndarray] = {}
y_test_enc: dict[str, np.ndarray] = {}

for col in TARGETS:
    le = LabelEncoder()
    le.fit(y[col].astype(str))
    y_train_enc[col] = le.transform(y_train_raw[col].astype(str))
    y_test_enc[col] = le.transform(y_test_raw[col].astype(str))
    encoders[col] = le
    print(f"{col}: {len(le.classes_)} classes --> {list(le.classes_)}")

# Feature Engineering ++++++++++++++++++++++++++++++++++++++++++++++++
X_train = pd.get_dummies(X_train_raw, dummy_na=False)
X_test = pd.get_dummies(X_test_raw, dummy_na=False)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print(f"\nFeature matrix shape: {X_train.shape}")

# Validation Split +++++++++++++++++++++++++++++++++++++++++++++++++++
train_sub_idx, val_idx = train_test_split(
    np.arange(len(X_train)),
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_enc["hpv_association_p16"],
)

X_tr = X_train[train_sub_idx]
X_val = X_train[val_idx]


def subset_labels(enc_dict, idx_array):
    return {
        "hpv_out": enc_dict["hpv_association_p16"][idx_array],
        "site_out": enc_dict["primary_tumor_site"][idx_array],
        "grade_out": enc_dict["grading"][idx_array],
    }


y_tr = subset_labels(y_train_enc, train_sub_idx)
y_val = subset_labels(y_train_enc, val_idx)
y_test = {
    "hpv_out": y_test_enc["hpv_association_p16"],
    "site_out": y_test_enc["primary_tumor_site"],
    "grade_out": y_test_enc["grading"],
}


# weights +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_sample_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


hpv_cw = get_sample_weights(y_tr["hpv_out"])
site_cw = get_sample_weights(y_tr["site_out"])
grade_cw = get_sample_weights(y_tr["grade_out"])

sw = (
    np.array([hpv_cw[l] for l in y_tr["hpv_out"]])
    + np.array([site_cw[l] for l in y_tr["site_out"]])
    + np.array([grade_cw[l] for l in y_tr["grade_out"]])
) / 3.0

print("\nClass distributions (train):")
for k, v in [
    ("HPV", y_tr["hpv_out"]),
    ("Site", y_tr["site_out"]),
    ("Grade", y_tr["grade_out"]),
]:
    print(f"  {k}: {np.bincount(v)}")


# Model +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def build_model(input_dim, encoders):
    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    hpv_out = tf.keras.layers.Dense(
        len(encoders["hpv_association_p16"].classes_),
        activation="softmax",
        name="hpv_out",
    )(x)
    site_out = tf.keras.layers.Dense(
        len(encoders["primary_tumor_site"].classes_),
        activation="softmax",
        name="site_out",
    )(x)
    grade_out = tf.keras.layers.Dense(
        len(encoders["grading"].classes_), activation="softmax", name="grade_out"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=[hpv_out, site_out, grade_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            k: "sparse_categorical_crossentropy"
            for k in ("hpv_out", "site_out", "grade_out")
        },
        loss_weights=LOSS_WEIGHTS,
        metrics={k: ["accuracy"] for k in ("hpv_out", "site_out", "grade_out")},
    )
    model.summary()
    return model


# Callbacks ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE // 3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# Training ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SEEDS = [0, 1, 2, 3, 4]
results = []
histories = []

for seed in SEEDS:
    print(f"\n{'='*50}\nRUNNING SEED {seed}\n{'='*50}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_model(X_tr.shape[1], encoders)

    history = model.fit(
        X_tr,
        [y_tr["hpv_out"], y_tr["site_out"], y_tr["grade_out"]],
        validation_data=(
            X_val,
            [y_val["hpv_out"], y_val["site_out"], y_val["grade_out"]],
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sw,
        callbacks=make_callbacks(),
        verbose=0,
    )
    histories.append(history)

    preds = model.predict(X_test, verbose=0)
    hpv_acc = np.mean(np.argmax(preds[0], axis=1) == y_test["hpv_out"])
    site_acc = np.mean(np.argmax(preds[1], axis=1) == y_test["site_out"])
    grade_acc = np.mean(np.argmax(preds[2], axis=1) == y_test["grade_out"])

    print(
        f"  Seed {seed} — HPV: {hpv_acc:.4f}  Site: {site_acc:.4f}  Grade: {grade_acc:.4f}"
    )
    results.append([hpv_acc, site_acc, grade_acc])


# Plotting Loss +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
min_epochs = min(len(h.history["loss"]) for h in histories)

all_train_loss = np.array([h.history["loss"][:min_epochs] for h in histories])
all_val_loss = np.array([h.history["val_loss"][:min_epochs] for h in histories])

mean_train_loss = all_train_loss.mean(axis=0)
std_train_loss = all_train_loss.std(axis=0)
mean_val_loss = all_val_loss.mean(axis=0)
std_val_loss = all_val_loss.std(axis=0)
ep = range(len(mean_train_loss))

plt.figure(figsize=(8, 5))
plt.plot(ep, mean_train_loss, label="Train Loss (mean)")
plt.fill_between(
    ep, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2
)
plt.plot(ep, mean_val_loss, label="Validation Loss (mean)")
plt.fill_between(
    ep, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (5 seeds)")
plt.legend()
plt.grid(True)
plt.savefig("Figures/text_loss.png", bbox_inches="tight")
plt.close()

# Per Task Accuracy +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for task in ["hpv_out", "site_out", "grade_out"]:
    train_key = f"{task}_accuracy"
    val_key = f"val_{task}_accuracy"
    min_ep_t = min(len(h.history[train_key]) for h in histories)

    all_tr = np.array([h.history[train_key][:min_ep_t] for h in histories])
    all_vl = np.array([h.history[val_key][:min_ep_t] for h in histories])

    mean_tr, std_tr = all_tr.mean(0), all_tr.std(0)
    mean_vl, std_vl = all_vl.mean(0), all_vl.std(0)
    ep_t = range(len(mean_tr))

    plt.figure(figsize=(8, 5))
    plt.plot(ep_t, mean_tr, label=f"{task} Train (mean)")
    plt.fill_between(ep_t, mean_tr - std_tr, mean_tr + std_tr, alpha=0.2)
    plt.plot(ep_t, mean_vl, label=f"{task} Val (mean)")
    plt.fill_between(ep_t, mean_vl - std_vl, mean_vl + std_vl, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{task} Accuracy (5 seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Figures/text_{task}_accuracies.png", bbox_inches="tight")
    plt.close()

# Confusion Matrix +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
preds = model.predict(X_test, verbose=0)

task_info = {
    "hpv_out": {"idx": 0, "encoder": "hpv_association_p16"},
    "site_out": {"idx": 1, "encoder": "primary_tumor_site"},
    "grade_out": {"idx": 2, "encoder": "grading"},
}

for task_name, info in task_info.items():
    y_pred = np.argmax(preds[info["idx"]], axis=1)
    y_true = y_test[task_name]
    labels = np.arange(len(encoders[info["encoder"]].classes_))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=encoders[info["encoder"]].classes_).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"{task_name} Confusion Matrix (final seed)")
    plt.xticks(rotation=45)
    plt.savefig(f"Figures/text_{task_name}_confusion.png", bbox_inches="tight")
    plt.close(fig)


# Evaluation ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def evaluate_model(model, X, y_true):
    res = model.evaluate(X, y_true, verbose=0)
    print("\nTest Results:")
    for name, val in zip(model.metrics_names, res):
        print(f"  {name}: {val:.4f}")
    return dict(zip(model.metrics_names, res))


test_results = evaluate_model(model, X_test, y_test)

# Final Stats ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
results = np.array(results)

print("\n=== Final Results ===")
for i, task in enumerate(["HPV", "Site", "Grade"]):
    mean = results[:, i].mean()
    std = results[:, i].std()
    ci = 1.96 * std / np.sqrt(len(SEEDS))
    print(f"{task}: {mean:.4f} ± {std:.4f} (95% CI ± {ci:.4f})")

# Baseline Comparison +++++++++++++++++++++++++++++++++++++++++++++++++++++++
baseline_results = {}
print("\n=== Majority Class Baselines ===")
for task_name, y_col in zip(
    ["HPV", "Site", "Grade"], ["hpv_out", "site_out", "grade_out"]
):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr[y_col])
    acc = dummy.score(X_test, y_test[y_col])
    baseline_results[task_name] = acc
    print(f"{task_name} Majority Class Baseline: {acc:.4f}")

print("\n=== Comparison to Baseline ===")
for i, task in enumerate(["HPV", "Site", "Grade"]):
    model_mean = results[:, i].mean()
    baseline = baseline_results[task]
    improvement = model_mean - baseline
    print(
        f"{task}: Model = {model_mean:.4f}, "
        f"Baseline = {baseline:.4f}, "
        f"Improvement = {improvement:+.4f}"
    )