import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = "pathological_data_filtered.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


TARGETS = [
    "hpv_association_p16",
    "primary_tumor_site",
    "grading"
]

df = df.dropna(subset=TARGETS)

X = df.drop(columns=TARGETS + ["patient_id"])
y = df[TARGETS]

X = pd.get_dummies(X, dummy_na=True)

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoders = {}
y_encoded = {}

for col in TARGETS:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col].astype(str))
    encoders[col] = le

    print(f"{col}: {len(le.classes_)} classes")


indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded["hpv_association_p16"]
)

X_train, X_test = X[train_idx], X[test_idx]

y_train = {
    "hpv_out": y_encoded["hpv_association_p16"][train_idx],
    "site_out": y_encoded["primary_tumor_site"][train_idx],
    "grade_out": y_encoded["grading"][train_idx]
}

y_test = {
    "hpv_out": y_encoded["hpv_association_p16"][test_idx],
    "site_out": y_encoded["primary_tumor_site"][test_idx],
    "grade_out": y_encoded["grading"][test_idx]
}


inputs = tf.keras.Input(shape=(X_train.shape[1],))

x = tf.keras.layers.Dense(256, activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(64, activation="relu")(x)


hpv_out = tf.keras.layers.Dense(
    len(encoders["hpv_association_p16"].classes_),
    activation="softmax",
    name="hpv_out"
)(x)

site_out = tf.keras.layers.Dense(
    len(encoders["primary_tumor_site"].classes_),
    activation="softmax",
    name="site_out"
)(x)

grade_out = tf.keras.layers.Dense(
    len(encoders["grading"].classes_),
    activation="softmax",
    name="grade_out"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=[hpv_out, site_out, grade_out]
)


model.compile(
    optimizer="adam",
    loss={
        "hpv_out": "sparse_categorical_crossentropy",
        "site_out": "sparse_categorical_crossentropy",
        "grade_out": "sparse_categorical_crossentropy"
    },

    loss_weights={
        "hpv_out": 1.5,
        "site_out": 1.0,
        "grade_out": 1.0
    },

    metrics={
        "hpv_out": ["accuracy"],
        "site_out": ["accuracy"],
        "grade_out": ["accuracy"]
    }
)

model.summary()

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

results = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Results:")
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

model.save("pathology_unimodal_model.keras")

print("\nModel saved.")