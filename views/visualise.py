"""The visualisers.

Some things in this subject are almost impossible to hold from a description
and obvious from thirty frames of animation. Gradient descent is one: "it takes
steps against the slope" means nothing until you watch a point walk down a
bowl, overshoot when the step is too big, and crawl when it is too small.

So each of these is a real computation, animated. Nothing here is a cartoon —
the descent really is descending on the real loss surface of the real bundled
data, and moving the learning rate really does change what happens, including
making it diverge. Divergence is left in on purpose: watching the loss climb
is how the number stops being arbitrary.

Everything is drawn with matplotlib in the sandbox, so the visualisers run
under exactly the same constraints as a lesson's code.
"""
from __future__ import annotations

import streamlit as st

from core import sandbox, ui

settings = ui.page("Visualise")

ui.hero("Watch it happen",
        "Each of these is a real computation on the bundled data, drawn frame "
        "by frame. Change the controls and watch what breaks.",
        eyebrow="Interactive", art="network")
st.write("")

tabs = st.tabs(["Gradient descent", "Decision boundary", "Bias and variance",
                "PCA", "Clustering"])


def show(code: str, key: str, caption: str = "") -> None:
    if st.button("Run it", type="primary", key=f"run-{key}",
                 use_container_width=True):
        with st.spinner("Computing…"):
            result = sandbox.run_notebook([code], timeout=90)
        if not result.ok:
            st.error(result.first_error)
            return
        cell = result.cells[0]
        for figure in cell.figures:
            ui.figure(figure)
        if cell.stdout.strip():
            st.code(cell.stdout.rstrip(), language="text")
        if caption:
            st.caption(caption)
    with st.expander("The code behind it"):
        st.code(code, language="python")


# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Gradient descent, descending")
    st.markdown(
        "A single-feature model fitted to the house data by hand. The left "
        "panel is the loss surface; the dots are the steps the algorithm "
        "actually took. The right panel is the loss over time."
    )
    rate = st.slider("Learning rate", 0.001, 1.2, 0.05, 0.001, format="%.3f",
                     help="Below about 0.01 it crawls. Above about 1.0 it "
                          "overshoots and the loss climbs.")
    steps = st.slider("Steps", 5, 300, 60, 5)

    show(f"""
import numpy as np, matplotlib.pyplot as plt
from core.datasets import load

df = load('house_prices')
x = df['area_sqft'].to_numpy(float)
x = (x - x.mean()) / x.std()
y = np.log(df['price'].to_numpy(float))
y = (y - y.mean()) / y.std()

def loss(w, b):
    return float((((w * x + b) - y) ** 2).mean())

w = b = -2.0
rate = {rate}
path, losses = [(w, b)], [loss(w, b)]
for _ in range({steps}):
    error = (w * x + b) - y
    w -= rate * 2 * (error * x).mean()
    b -= rate * 2 * error.mean()
    if not np.isfinite(w) or abs(w) > 1e6:
        break
    path.append((w, b))
    losses.append(loss(w, b))

grid_w = np.linspace(-2.5, 2.5, 90)
grid_b = np.linspace(-2.5, 2.5, 90)
W, Bb = np.meshgrid(grid_w, grid_b)
surface = np.array([[loss(wi, bi) for wi in grid_w] for bi in grid_b])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].contourf(W, Bb, surface, levels=28, cmap='magma', alpha=.85)
axes[0].contour(W, Bb, surface, levels=14, colors='white',
                linewidths=.4, alpha=.3)
px, py = zip(*path)
axes[0].plot(px, py, '-o', color='#7ee0b8', markersize=3.2, linewidth=1.3)
axes[0].plot(px[0], py[0], 'o', color='#ff7a9c', markersize=9, label='start')
axes[0].plot(px[-1], py[-1], '*', color='#ffb86b', markersize=16, label='end')
axes[0].set_xlabel('slope'); axes[0].set_ylabel('intercept')
axes[0].set_title('the loss surface, and the path taken')
axes[0].legend(loc='upper right')

axes[1].plot(losses, color='#6ea8fe')
axes[1].set_xlabel('step'); axes[1].set_ylabel('mean squared error')
axes[1].set_title('loss over time')
if len(losses) > 1 and losses[-1] > losses[0]:
    axes[1].set_title('loss over time — it is going UP')
fig.tight_layout()

print('steps taken:', len(path) - 1)
print('start loss: ', round(losses[0], 4))
print('end loss:   ', round(losses[-1], 4))
if losses[-1] > losses[0]:
    print('DIVERGED — the step size is too large. Each step overshoots the')
    print('valley and lands further up the other side.')
elif len(losses) > 2 and abs(losses[-1] - losses[-2]) < 1e-6:
    print('Converged: the last step changed almost nothing.')
else:
    print('Still descending — it would keep improving with more steps.')
""", key="gd", caption="The path is the actual sequence of weights, not an "
                       "illustration of one.")


# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### A decision boundary forming")
    st.markdown("The same two features, four models. What separates them is "
                "not accuracy but the *shape* each is able to draw.")

    show("""
import numpy as np, matplotlib.pyplot as plt
from core.datasets import load
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = load('customer_churn')
X = df[['tenure_months', 'support_calls']].to_numpy(float)
X = StandardScaler().fit_transform(X)
y = df['churned'].to_numpy()

models = [('logistic regression', LogisticRegression()),
          ('k-NN, k=25', KNeighborsClassifier(25)),
          ('one tree, depth 4', DecisionTreeClassifier(max_depth=4, random_state=0)),
          ('forest of 200', RandomForestClassifier(n_estimators=200, random_state=0))]

xs = np.linspace(X[:, 0].min() - .4, X[:, 0].max() + .4, 220)
ys = np.linspace(X[:, 1].min() - .4, X[:, 1].max() + .4, 220)
XX, YY = np.meshgrid(xs, ys)
grid = np.c_[XX.ravel(), YY.ravel()]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
sample = np.random.default_rng(0).choice(len(X), 700, replace=False)
for ax, (name, model) in zip(axes.ravel(), models):
    model.fit(X, y)
    Z = model.predict_proba(grid)[:, 1].reshape(XX.shape)
    ax.contourf(XX, YY, Z, levels=20, cmap='RdYlBu_r', alpha=.7)
    ax.contour(XX, YY, Z, levels=[0.5], colors='white', linewidths=1.6)
    ax.scatter(X[sample, 0], X[sample, 1], c=y[sample], cmap='RdYlBu_r',
               s=5, edgecolors='none', alpha=.55)
    ax.set_title(name)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('the white line is where the model says 50/50', y=1.0)
fig.tight_layout()

print('Logistic regression can only draw a straight line.')
print('k-NN draws whatever the neighbours say, which is bumpy.')
print('One tree draws rectangles, because it only splits on one feature at a time.')
print('The forest averages 200 sets of rectangles, which smooths them out.')
""", key="db")


# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Bias against variance, on one picture")
    st.markdown("The same data fitted at several polynomial degrees. Watch "
                "the training error fall forever while the test error turns "
                "around.")
    show("""
import numpy as np, matplotlib.pyplot as plt
from core.datasets import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

df = load('house_prices').fillna(load('house_prices').median(numeric_only=True))
small = df.sample(260, random_state=0)
X = small[['area_sqft']].to_numpy(float)
y = np.log(small['price'].to_numpy(float))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

degrees = range(1, 16)
train_err, test_err = [], []
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(),
                          LinearRegression()).fit(X_tr, y_tr)
    train_err.append(mean_squared_error(y_tr, model.predict(X_tr)))
    test_err.append(mean_squared_error(y_te, model.predict(X_te)))

best = int(np.argmin(test_err)) + 1

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(list(degrees), train_err, '-o', color='#7ee0b8', label='training')
axes[0].plot(list(degrees), test_err, '-o', color='#ff7a9c', label='test')
axes[0].axvline(best, color='#ffb86b', linestyle='--',
                label='best test error (degree %d)' % best)
axes[0].set_xlabel('polynomial degree')
axes[0].set_ylabel('mean squared error (log scale)')
axes[0].set_yscale('log')
axes[0].set_title('training error never stops falling'); axes[0].legend()

order = np.argsort(X_te.ravel())
axes[1].scatter(X_te, y_te, s=8, color='#8b95a9', alpha=.6, label='test data')
for degree, colour in ((1, '#6ea8fe'), (best, '#7ee0b8'), (15, '#ff7a9c')):
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(),
                          LinearRegression()).fit(X_tr, y_tr)
    axes[1].plot(X_te.ravel()[order], model.predict(X_te)[order],
                 color=colour, linewidth=1.8, label='degree %d' % degree)
axes[1].set_xlabel('area (sq ft)'); axes[1].set_ylabel('log price')
axes[1].set_title('too stiff, about right, and far too wobbly')
axes[1].legend()
fig.tight_layout()

print('best degree by test error:', best)
print('training error at degree 1: ', round(train_err[0], 4))
print('training error at degree 15:', round(train_err[-1], 4))
print('test error at degree 15:    ', round(test_err[-1], 4))
print()
print('The training error at degree 15 is the lowest it ever gets, and the')
print('model is the worst one here by a factor of about a billion — which is')
print('why the left axis has to be logarithmic. That gap is overfitting.')
""", key="bv")


# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### PCA, turning the data to face you")
    st.markdown("Thirty correlated measurements, projected down to two "
                "directions chosen to keep as much spread as possible.")
    show("""
import numpy as np, matplotlib.pyplot as plt
from core.datasets import load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = load('breast_cancer')
y = df['diagnosis']
X = StandardScaler().fit_transform(
    df.drop(columns=['diagnosis']).select_dtypes('number'))

pca = PCA().fit(X)
coords = pca.transform(X)
shares = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
labels = np.asarray(y)
for value, colour in zip(np.unique(labels), ('#6ea8fe', '#ff7a9c')):
    mask = labels == value
    axes[0].scatter(coords[mask, 0], coords[mask, 1], s=11, alpha=.65,
                    color=colour, label=str(value))
axes[0].set_xlabel('first component (%.0f%% of the spread)' % (shares[0] * 100))
axes[0].set_ylabel('second component (%.0f%%)' % (shares[1] * 100))
axes[0].set_title('thirty dimensions, flattened to two')
axes[0].legend()

axes[1].bar(range(1, 16), shares[:15], color='#6ea8fe')
axes[1].plot(range(1, 16), np.cumsum(shares[:15]), '-o', color='#7ee0b8',
             markersize=4, label='running total')
axes[1].axhline(0.9, color='#ffb86b', linestyle='--', label='90%')
axes[1].set_xlabel('component'); axes[1].set_ylabel('share of variance')
axes[1].set_title('most of it is in the first few'); axes[1].legend()
fig.tight_layout()

needed = int(np.searchsorted(np.cumsum(shares), 0.90) + 1)
print('components for 90%% of the variance: %d of %d' % (needed, len(shares)))
print('The two classes separate visibly, and PCA never saw the labels.')
""", key="pca")


# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### k-means, and why choosing k is the hard part")
    k = st.slider("Number of clusters", 2, 9, 4)
    show(f"""
import numpy as np, matplotlib.pyplot as plt
from core.datasets import load
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = load('customer_churn')
features = ['tenure_months', 'monthly_charges', 'support_calls', 'age']
X = df[features].replace(9999, np.nan)
X = X.fillna(X.median())
X = StandardScaler().fit_transform(X)
flat = PCA(n_components=2).fit_transform(X)

model = KMeans(n_clusters={k}, n_init=10, random_state=0).fit(X)
centres = PCA(n_components=2).fit(X).transform(model.cluster_centers_)

inertias, silhouettes = [], []
for candidate in range(2, 10):
    fitted = KMeans(n_clusters=candidate, n_init=10, random_state=0).fit(X)
    inertias.append(fitted.inertia_)
    silhouettes.append(silhouette_score(X, fitted.labels_, sample_size=1500,
                                        random_state=0))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].scatter(flat[:, 0], flat[:, 1], c=model.labels_, cmap='viridis',
                s=6, alpha=.6)
axes[0].scatter(centres[:, 0], centres[:, 1], marker='X', s=180,
                color='#ffb86b', edgecolors='black', linewidths=.8)
axes[0].set_title('{k} clusters, drawn on the first two components')
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].plot(range(2, 10), inertias, '-o', color='#6ea8fe', label='inertia')
axes[1].set_xlabel('k'); axes[1].set_ylabel('inertia', color='#6ea8fe')
twin = axes[1].twinx()
twin.plot(range(2, 10), silhouettes, '-o', color='#7ee0b8',
          label='silhouette')
twin.set_ylabel('silhouette', color='#7ee0b8')
axes[1].set_title('inertia always falls; silhouette does not')
fig.tight_layout()

print('inertia at k=2: ', round(inertias[0], 1))
print('inertia at k=9: ', round(inertias[-1], 1))
print('best silhouette at k =', int(np.argmax(silhouettes)) + 2)
print()
print('Inertia falls at every k, so minimising it would always choose the')
print('largest k you tried. The silhouette turns around, which is why it can')
print('actually answer the question.')
""", key="km")
