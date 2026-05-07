# ─────────────────────────────────────────
# Soft Sensor Mini Project — v1
# Light Sensor → Liquid Level → Valve Signal
# ─────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

# Set a random seed so results are reproducible every run
np.random.seed(42)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

NUM_SAMPLES = 1000   # how many training examples we generate
WINDOW_SIZE = 32     # how many time-steps per sensor reading
                     # (your real system uses 128 — same idea, just smaller here)

# ─────────────────────────────────────────
# STEP 1 — Generate Fake Sensor Data
# ─────────────────────────────────────────

# Generate 1000 random "true" liquid levels between 0 and 1
# These are the ground truth labels the NN will try to learn
true_level = np.random.uniform(0, 1, NUM_SAMPLES)

X = []   # will hold light sensor windows (inputs to the NN)
Y = []   # will hold true liquid levels  (targets the NN learns to predict)

for level in true_level:

    # Physics model: more liquid → less light gets through
    # At level=0 (empty): base_light = 1.0 (full brightness)
    # At level=1 (full):  base_light = 0.3 (heavily attenuated)
    base_light = 1.0 - 0.7 * level

    # Real sensors are noisy — emulsions scatter light unpredictably
    # We add small random noise to each of the 32 time-steps in the window
    noise = np.random.normal(0, 0.05, WINDOW_SIZE)  # mean=0, std=0.05

    # Final signal = physics baseline + noise, shape: (32,)
    signal = base_light + noise

    X.append(signal)   # one row of input features
    Y.append(level)    # one target label

# Convert lists to numpy arrays for use with TensorFlow
X = np.array(X)   # shape: (1000, 32)
Y = np.array(Y)   # shape: (1000,)

# ─────────────────────────────────────────
# STEP 2 — Visualize What the NN Will See
# ─────────────────────────────────────────

# Plot the first 5 sensor windows so we can see what the data looks like
# Each line is a different liquid level — notice they cluster at different heights
# but overlap due to noise. One single reading can't tell you the level;
# the whole window pattern is what carries the information.
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.plot(X[i], label=f'Liquid={Y[i]:.2f}')
plt.title("Raw light sensor windows")
plt.xlabel("Time index")
plt.ylabel("Light intensity")
plt.legend()
plt.show()

# ─────────────────────────────────────────
# STEP 3 — Scale the Data
# ─────────────────────────────────────────

# Divide X by 1.2 to bring values into a tighter range around [0, 1]
# We DON'T subtract the mean — that would destroy the physical meaning
# (the absolute light level carries information about how much liquid is present)
# Y is already between 0 and 1, so no scaling needed
X_scaled = X / 1.2
Y_scaled = Y

# ─────────────────────────────────────────
# STEP 4 — Build the Neural Network
# ─────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sequential means layers are stacked one after another (no branches)
model = Sequential([

    # Layer 1: takes the 32-point sensor window as input
    # 32 neurons, ReLU activation (kills negative values, passes positive ones)
    # This layer learns basic features from the raw signal
    Dense(32, activation='relu', input_shape=(WINDOW_SIZE,)),

    # Layer 2: wider layer — learns more complex combinations of features
    # 64 neurons gives the model more capacity to find patterns
    Dense(64, activation='relu'),

    # Output layer: single neuron, no activation
    # Outputs one continuous number — the predicted liquid level
    Dense(1)
])

# Compile sets up how the model will learn:
#   Adam optimizer — adaptive learning rate, works well out of the box
#   MSE loss — penalizes large prediction errors more than small ones
#   Good choice for regression (predicting a continuous number)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mse"
)

# Print a summary of the model architecture and parameter count
model.summary()

# ─────────────────────────────────────────
# STEP 5 — Train the Model
# ─────────────────────────────────────────

# model.fit() runs the training loop:
#   epochs=40        — pass through the full dataset 40 times
#   batch_size=16    — update weights every 16 samples (not all at once)
#   validation_split — hold out 20% of data to check for overfitting
#   verbose=0        — suppress per-epoch output (keeps terminal clean)
history = model.fit(
    X_scaled, Y_scaled,
    epochs=40,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# Plot training vs validation loss over epochs
# If both curves drop and stay close together → model is learning, not overfitting
# If val loss rises while train loss drops → overfitting (memorizing, not generalizing)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Training loss")
plt.legend()
plt.show()

# ─────────────────────────────────────────
# STEP 6 — Test the Model (Truth vs Prediction)
# ─────────────────────────────────────────

# Run all 1000 inputs through the trained model to get predictions
# .flatten() converts shape (1000, 1) → (1000,) so it plots cleanly
pred = model.predict(X_scaled).flatten()

# Plot the true liquid levels vs what the NN predicted
# If the orange line tracks the blue line closely → soft sensor is working
plt.figure(figsize=(10, 4))
plt.plot(Y, label="True Liquid")
plt.plot(pred, label="Predicted (NN)", alpha=0.8)  # alpha=0.8 makes it slightly transparent
plt.legend()
plt.show()

# ─────────────────────────────────────────
# STEP 7 — Convert Prediction to Valve Command
# ─────────────────────────────────────────

def valve_opening(level):
    # Simple inverse control rule:
    #   level=0 (empty tank) → valve fully open  (1.0)
    #   level=1 (full tank)  → valve fully closed (0.0)
    # np.clip ensures the output never goes below 0 or above 1
    # (predictions can drift slightly outside [0,1])
    return np.clip(1.0 - level, 0, 1)

# Apply the valve rule to the NN's predicted liquid levels
valve_cmd = valve_opening(pred)

# Plot the resulting valve command signal over all 1000 samples
plt.figure(figsize=(10, 4))
plt.plot(valve_cmd, label="Valve Command")
plt.title("Valve opening signal")
plt.legend()
plt.show()