import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

CLINICAL_PATH = "StructuredData/clinical_data.json"
PATHOLOGICAL_PATH = "StructuredData/pathological_data.json"
BLOOD_PATH = "StructuredData/blood_data.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15

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

LOSS_WEIGHTS = {"hpv_out": 2.0, "site_out": 0.8, "grade_out": 1.2}


# Data loading
def load_json_to_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.json_normalize(data)


clinical = load_json_to_df(CLINICAL_PATH)
pathological = load_json_to_df(PATHOLOGICAL_PATH)
blood_raw = load_json_to_df(BLOOD_PATH)

blood_raw.columns = blood_raw.columns.str.strip()

blood = blood_raw.pivot_table(
    index="patient_id", columns="analyte_name", values="value", aggfunc="mean"
).reset_index()

blood.columns.name = None

clinical.columns = clinical.columns.str.strip()
pathological.columns = pathological.columns.str.strip()

df = clinical.merge(pathological, on="patient_id", how="inner").merge(
    blood, on="patient_id", how="inner"
)
print("Merged dataset shape:", df.shape)

df = df.dropna(subset=TARGETS)

# Features
X = df.drop(columns=TARGETS + ["patient_id"] + LEAKY_COLS, errors="ignore")
y = df[TARGETS]

# Split
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

# Target Encoding
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

# Feature Engineering
X_train = pd.get_dummies(X_train_raw, dummy_na=True)
X_test = pd.get_dummies(X_test_raw, dummy_na=True)

# Align so test has same columns as train
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Validation Split
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

# Sample weights
hpv_classes = np.unique(y_tr["hpv_out"])
hpv_weights = compute_class_weight(
    class_weight="balanced", classes=hpv_classes, y=y_tr["hpv_out"]
)
hpv_class_weight = dict(zip(hpv_classes.tolist(), hpv_weights.tolist()))
sample_weights = np.array([hpv_class_weight[label] for label in y_tr["hpv_out"]])

# ─Model
inputs = tf.keras.Input(shape=(X_tr.shape[1],))

x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

hpv_out = tf.keras.layers.Dense(
    len(encoders["hpv_association_p16"].classes_), activation="softmax", name="hpv_out"
)(x)

site_out = tf.keras.layers.Dense(
    len(encoders["primary_tumor_site"].classes_), activation="softmax", name="site_out"
)(x)

grade_out = tf.keras.layers.Dense(
    len(encoders["grading"].classes_), activation="softmax", name="grade_out"
)(x)

model = tf.keras.Model(inputs=inputs, outputs=[hpv_out, site_out, grade_out])

model.compile(
    optimizer="adam",
    loss={
        k: "sparse_categorical_crossentropy"
        for k in ("hpv_out", "site_out", "grade_out")
    },
    loss_weights=LOSS_WEIGHTS,
    metrics={k: ["accuracy"] for k in ("hpv_out", "site_out", "grade_out")},
)
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=PATIENCE // 2, min_lr=1e-6, verbose=1
    ),
]

# Training
model.fit(
    X_tr,
    [y_tr["hpv_out"], y_tr["site_out"], y_tr["grade_out"]],
    validation_data=(X_val, [y_val["hpv_out"], y_val["site_out"], y_val["grade_out"]]),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    sample_weight=sample_weights,
    callbacks=callbacks,
    verbose=1,
)


# Evaluation
def evaluate_model(model, X, y_true):
    results = model.evaluate(X, y_true, verbose=0)
    print("\nTest Results:")
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")


evaluate_model(model, X_test, y_test)

model.save("multimodal_fusion_json.keras")
print("\nModel saved successfully.")