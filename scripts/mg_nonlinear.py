r"""
Nonlinear-decoder control: is the strong-entangled code's information PRESENT (just not linearly
accessible)? We decode each factor from the RAW state input with (a) a linear probe and (b) a small
MLP, on both perceptual codes. If the MLP recovers the factors from the strong code while the
linear probe cannot---and the trained model's residual readability matches only the linear level
(0.58, from mg_probe)---then the model performs at most linear-level extraction: linear readability
is a gate, not an information limit.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_nonlinear.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from mg.multiworld import TwoObjectWorld
from mg.data import make_encoder
from mg.world import Query

world = TwoObjectWorld()
states = world.states
Q = [Query(s, "bind", 3, s[3], "") for s in states]

print(f"{'code (raw input)':22s} {'dim':>4s} {'factor':>8s} {'linear':>7s} {'MLP':>6s}")
print("-" * 55)
for name in ["state_perceptual", "state_perceptual_hard"]:
    enc = make_encoder(name.replace("state_", ""), world)
    X = np.array([enc.encode(q, world) for q in Q])
    lin_all, mlp_all = [], []
    for f, fname in [(0, "color1"), (1, "shape1"), (2, "color2"), (3, "shape2")]:
        y = np.array([s[f] for s in states])
        lin = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
        mlp = cross_val_score(MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000,
                                            random_state=0), X, y, cv=5).mean()
        lin_all.append(lin); mlp_all.append(mlp)
        print(f"{name:22s} {X.shape[1]:>4d} {fname:>8s} {lin:7.2f} {mlp:6.2f}")
    print(f"{'':22s} {'':>4s} {'MEAN':>8s} {np.mean(lin_all):7.2f} {np.mean(mlp_all):6.2f}")
    print("-" * 55)
print("\nRecall (mg_probe): the trained model's residual decodability of the answer under the strong")
print("code is ~0.59, i.e. it matches the LINEAR level of its input, not the MLP-recoverable level.")
