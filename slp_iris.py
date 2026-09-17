"""
Single Layer Perceptron - Binary Classification (Iris: Setosa dan Versicolor)

Mereplikasi proses training "SLP" di Google Sheets persis:
  - 100 sampel (50 Setosa baris 1-50, lalu 50 Versicolor baris 51-100).
  - Data training (80 sampel) = 40 Setosa pertama (baris 1-40) + 40
    Versicolor pertama (baris 51-90), urutannya sama kayak di sheet.
  - Data validasi/testing (20 sampel) = sisa 10 Setosa (baris 41-50)
    + sisa 10 Versicolor (baris 91-100) - sama seperti sheet
    Testing/testing1.
  - Bobot awal = 0.5 (bias, teta1..teta4), learning rate = 0.1
  - 5 epoch, gradient descent ONLINE (per sampel) - bobot di-update
    setiap satu sampel training selesai diproses, persis kayak tiap
    baris di spreadsheet.
  - Aktivasi sigmoid, loss pakai Sum-Square-Error.
  - Validasi tiap epoch pakai bobot hasil AKHIR epoch training itu
    (cuma forward pass, tanpa update) - sama kayak konvensi di sheet
    validation/testing.
"""

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

DATA_PATH = "iris_slp_data.csv"
LEARNING_RATE = 0.1
EPOCHS = 5


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data(path=DATA_PATH):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            x1, x2, x3, x4, label = row
            X.append([float(x1), float(x2), float(x3), float(x4)])
            y.append(0 if label.strip() == "Iris-setosa" else 1)  # Setosa=0, Versicolor=1
    return X, y


# ----------------------------------------------------------------------
# 2. Model helpers
# ----------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def forward(x, bias, w1, w2, w3, w4):
    z = bias + w1 * x[0] + w2 * x[1] + w3 * x[2] + w4 * x[3]
    o = sigmoid(z)
    pred = 1 if o > 0.5 else 0
    return o, pred


def evaluate(X_set, y_set, bias, w1, w2, w3, w4):
    """Forward pass only (no weight update) - used for validation/testing."""
    losses, correct = [], 0
    for x, target in zip(X_set, y_set):
        o, pred = forward(x, bias, w1, w2, w3, w4)
        error = o - target
        losses.append(error ** 2)
        if pred == target:
            correct += 1
    avg_loss = sum(losses) / len(losses)
    accuracy = correct / len(losses) * 100
    return avg_loss, accuracy


# ----------------------------------------------------------------------
# 3. Training loop (per-sample / online SGD, exactly like the spreadsheet)
# ----------------------------------------------------------------------
def split_data(X, y):
    """Same split used in the spreadsheet:
       train = rows 1-40 (Setosa) + rows 51-90 (Versicolor)   -> 80 samples
       test  = rows 41-50 (Setosa) + rows 91-100 (Versicolor) -> 20 samples
       (indices below are 0-based, so row 1 = index 0)
    """
    X_train = X[0:40] + X[50:90]
    y_train = y[0:40] + y[50:90]
    X_test = X[40:50] + X[90:100]
    y_test = y[40:50] + y[90:100]
    return X_train, y_train, X_test, y_test


def train():
    X, y = load_data()
    X_train, y_train, X_test, y_test = split_data(X, y)

    bias, w1, w2, w3, w4 = 0.5, 0.5, 0.5, 0.5, 0.5  # I5:M5 in the sheet

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        epoch_losses, epoch_correct = [], 0

        for x, target in zip(X_train, y_train):
            # ---- forward pass (uses the weights as they currently stand) ----
            o, pred = forward(x, bias, w1, w2, w3, w4)
            error = o - target
            epoch_losses.append(error ** 2)
            if pred == target:
                epoch_correct += 1

            # ---- backward pass (gradient of SSE w.r.t. sigmoid output) ----
            grad_common = 2 * error * (1 - o) * o
            d_bias = grad_common
            d_w1 = grad_common * x[0]
            d_w2 = grad_common * x[1]
            d_w3 = grad_common * x[2]
            d_w4 = grad_common * x[3]

            # ---- update weights immediately (per-sample SGD, like each row) ----
            bias -= LEARNING_RATE * d_bias
            w1 -= LEARNING_RATE * d_w1
            w2 -= LEARNING_RATE * d_w2
            w3 -= LEARNING_RATE * d_w3
            w4 -= LEARNING_RATE * d_w4

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_acc = epoch_correct / len(epoch_losses) * 100

        # ---- validation: forward pass only, weights at END of this epoch ----
        val_loss, val_acc = evaluate(X_test, y_test, bias, w1, w2, w3, w4)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.6f} train_acc={train_acc:.2f}%  |  "
              f"val_loss={val_loss:.6f} val_acc={val_acc:.2f}%")

    print("\nFinal trained weights:")
    print(f"bias={bias:.6f}, teta1={w1:.6f}, teta2={w2:.6f}, "
          f"teta3={w3:.6f}, teta4={w4:.6f}")

    return history


RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)


def _smooth_curve(epochs, values, n_points=300):
    """Return (x, y) for a smooth spline through (epochs, values)."""
    x_smooth = np.linspace(min(epochs), max(epochs), n_points)
    spline = make_interp_spline(epochs, values, k=3)  # cubic spline
    return x_smooth, spline(x_smooth)


def _plot_pair(epochs, train_vals, val_vals, title, ylabel, filename_prefix):
    # ---- straight-line version ----
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_vals, marker="o", label=f"Training {ylabel}")
    plt.plot(epochs, val_vals, marker="o", label=f"Validation {ylabel}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename_prefix}_straight.png"), dpi=150)
    plt.close()

    # ---- smooth-curve version (Excel-like) ----
    x_train, y_train = _smooth_curve(epochs, train_vals)
    x_val, y_val = _smooth_curve(epochs, val_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(x_train, y_train, color="#1f77b4", label=f"Training {ylabel}")
    plt.plot(x_val, y_val, color="#ff7f0e", label=f"Validation {ylabel}")
    plt.plot(epochs, train_vals, "o", color="#1f77b4")
    plt.plot(epochs, val_vals, "o", color="#ff7f0e")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{filename_prefix}_smooth.png"), dpi=150)
    plt.close()


def plot_charts(history):
    epochs = history["epoch"]

    _plot_pair(epochs, history["train_acc"], history["val_acc"],
               "Accuracy per Epoch", "Accuracy (%)", "accuracy_chart")

    _plot_pair(epochs, history["train_loss"], history["val_loss"],
               "Loss (Avg Sum Square Error) per Epoch", "Avg Sum Square Error", "loss_chart")

    print(f"\nSaved to '{RESULT_DIR}/': accuracy_chart_straight.png, accuracy_chart_smooth.png, "
          "loss_chart_straight.png, loss_chart_smooth.png")


if __name__ == "__main__":
    hist = train()
    plot_charts(hist)