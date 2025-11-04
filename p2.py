# %% [markdown]
# ### Cancer classification achieving 97% F1 score


# %%
import sklearn.datasets as ds
dataset = ds.load_breast_cancer(as_frame=True)


# %%
#convert to dataframe
df = dataset.frame


# %% [markdown]
# ## General idea of dataset

# %%
print(df.head())
print(df.info())
print(df.describe())

# %%
print(df.target.value_counts()) #No it's not balanced but it is managable

# %%


# %%
import matplotlib.pyplot as plt
import pandas as pd

# cols_to_plot = ['mean radius', 'mean texture', 'mean smoothness', 'mean compactness', 'mean perimeter']
cols_to_plot = list(df.columns.values)

fig, ax = plt.subplots(nrows=16, ncols=2, figsize=(12, 32))
axes = ax.flatten()  # makes it easy to loop

for i, col_name in enumerate(cols_to_plot):
    df[col_name].plot(
        kind='density',
        bw_method='scott',
        color='blue',
        linestyle='-',
        linewidth=2,
        ax=axes[i]
    )
    axes[i].set_title(col_name)

plt.tight_layout()
plt.show()


# %%
import seaborn as sns
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# %%
#most correlated pairs
import numpy as np
corr_matrix = df.corr().abs()

#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))
# print(sol[:5])

#least correlated features
print(sol[:(len(sol)- 1)])

# %%
from sklearn.model_selection import train_test_split
X = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(x_train))
print(len(x_test))


# %%
#fit scaler on training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = model.predict(x_train)
print(classification_report(y_train, y_pred_train))

# %%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = model.predict(x_train)
print(classification_report(y_train, y_pred_train))

# %%
from sklearn.naive_bayes import BernoulliNB
#AWFUL RESULTS!
model = BernoulliNB()
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = model.predict(x_train)
print(classification_report(y_train, y_pred_train))

# %%
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
estimators = [
    ("randomForest", RandomForestClassifier()),
    ("logisticRegression", LogisticRegression(max_iter=5000)),
    ("XGBclassifier", HistGradientBoostingClassifier()),
    ("KNNClassifier", KNeighborsClassifier(n_neighbors=3)),
    ("AdaBoostClassifier", AdaBoostClassifier())
]
model = StackingClassifier(estimators, final_estimator=LogisticRegression(max_iter=5000))
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = model.predict(x_train)
print(classification_report(y_train, y_pred_train))

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, random_state=42
)

lr = make_pipeline(
    StandardScaler(with_mean=False),  # set with_mean=False if x is sparse; else True
    LogisticRegression(max_iter=5000, C=1.0, class_weight=None, random_state=42)
)

hgb = HistGradientBoostingClassifier(
    max_depth=None, learning_rate=0.07, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=42
)

knn = make_pipeline(
    StandardScaler(with_mean=False),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)

adb = AdaBoostClassifier(
    n_estimators=300, learning_rate=0.05, random_state=42
)

estimators = [
    ("rf", rf),
    ("lr", lr),
    ("hgb", hgb),
    ("knn", knn),
    ("adb", adb),
]

meta = LogisticRegression(
    max_iter=5000, C=0.5, penalty="l2", random_state=42
)

model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    passthrough=True,
    stack_method="auto",
    cv=5,            # out-of-fold predictions → better generalization
    n_jobs=-1
)

model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = model.predict(x_train)
print(classification_report(y_train, y_pred_train))


# %%
dup_train = pd.DataFrame(x_train).duplicated().sum()
overlap = pd.merge(pd.DataFrame(x_train).assign(_t=1),
                   pd.DataFrame(x_test).assign(_t=2), how="inner").shape[0]
print("Duplicate rows in train:", dup_train)
print("Exact overlaps train↔test:", overlap)

# %%
from sklearn.experimental import enable_hist_gradient_boosting  # no-op if recent sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report
import numpy as np

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

param_grid = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [10, 20, 50],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0]
}
hgb = HistGradientBoostingClassifier(random_state=42)
gs = GridSearchCV(hgb, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=0)
gs.fit(x_train, y_train)
best = gs.best_estimator_
print("Best HGB:", gs.best_params_, "CV f1:", gs.best_score_)


# %%
# model.fit(x_train, y_train)
y_pred_test = best.predict(x_test)
print("TEST RESULTS")
print(classification_report(y_test, y_pred_test))
print("TRAIN RESULTS")
y_pred_train = best.predict(x_train)
print(classification_report(y_train, y_pred_train))


