import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
pd.options.display.float_format = '{:,.4f}'.format

customers = pd.read_excel('/Users/thoitruong/Documents/Portfolio/customers.xlsx')
transactions = pd.read_excel('/Users/thoitruong/Documents/Portfolio/transactions.xlsx')


SNAPSHOT_DATE = transactions["transaction_date"].max() + pd.Timedelta(days=1)
print("snapshot_date: ", SNAPSHOT_DATE)

rfm = (
    transactions
    .groupby("customer_id")
    .agg({
        "transaction_date": lambda x: (SNAPSHOT_DATE - x.max()).days,
        "customer_id": "count",
        "amount": "sum"
    })
    .rename(columns={
        "transaction_date": "recency",
        "customer_id": "frequency",
        "amount": "monetary"
    })
    .reset_index()
)

rfm_q = rfm.copy()

# Recency: nhỏ tốt → score cao
rfm_q["R_score"] = pd.qcut(
    rfm_q["recency"],
    q=5,
    labels=[5, 4, 3, 2, 1]
)

# Frequency: lớn tốt
rfm_q["F_score"] = pd.qcut(
    rfm_q["frequency"],
    q=5,
    labels=[1, 2, 3, 4, 5]
)

# Monetary: lớn tốt
rfm_q["M_score"] = pd.qcut(
    rfm_q["monetary"],
    q=5,
    labels=[1, 2, 3, 4, 5]
)

rfm_q[["customer_id", "R_score", "F_score", "M_score"]].head()

rfm_q["RFM_score"] = (
    rfm_q["R_score"].astype(str) +
    rfm_q["F_score"].astype(str) +
    rfm_q["M_score"].astype(str)
)

rfm_q[["customer_id", "RFM_score"]].head()

def rfm_segment(row):
    r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    if r >= 4 and f >= 3:
        return "Loyal Customers"

    if r >= 4 and f <= 2:
        return "New Customers"

    if r <= 2 and f >= 3:
        return "At Risk"

    if r <= 2 and f <= 2:
        return "Hibernating"

    return "Others"

rfm_q["rfm_segment"] = rfm_q.apply(rfm_segment, axis=1)
rfm_q[["customer_id", "RFM_score", "rfm_segment"]].head()

# Tính số lượng và tỷ lệ phần trăm
counts = rfm_q["rfm_segment"].value_counts()
percentages = rfm_q["rfm_segment"].value_counts(normalize=True) * 100

# Gộp thành một bảng tổng hợp
summary_table = pd.concat([counts, percentages], axis=1)
summary_table.columns = ['Count', 'Percentage (%)']

segment_order = [
    "Champions", 
    "Loyal Customers", 
    "New Customers", 
    "At Risk", 
    "Hibernating", 
    "Others"
]

segment_profile_q = (
    rfm_q
    .groupby("rfm_segment")[["recency", "frequency", "monetary"]]
    .mean()
    .sort_values("monetary", ascending=False)
)

transactions['gap_days'] = (transactions['transaction_date'] - transactions.groupby('customer_id')['transaction_date'].shift(1)).dt.days

q98 = transactions['gap_days'].quantile(0.98)
thresholds = [30, 60, 90]

idx = min(np.searchsorted(thresholds, q98, side='right'), len(thresholds) - 1)
INACTIVITY_WINDOW = thresholds[idx]

rfm_q['churn_label'] = np.where(rfm_q['recency'] > INACTIVITY_WINDOW,1,0)

pd.crosstab(
    rfm_q["churn_label"],          # KMeans segment
    rfm_q["rfm_segment"]     # RFM rule-based
)

# CUTOFF_DATE = pd.Timestamp("2025-12-31")
HORIZON_DAY = 30


df = transactions.copy()
def label_churn(df, snapshot_date, horizon_day, inactivity_window):
    df_labels = []
    for i in range(5):
        cutoff_date = (snapshot_date - max(timedelta(horizon_day),timedelta(inactivity_window))).replace(day=1) - pd.DateOffset(months=i) - pd.DateOffset(days=1)
        future = df[
            (df["transaction_date"] > cutoff_date) &
            (df["transaction_date"] <= cutoff_date + timedelta(days=inactivity_window))
        ]
        active_customers = future["customer_id"].unique()

        labels = (
            df[df["transaction_date"] <= cutoff_date]
            [["customer_id"]]
            .drop_duplicates()
        )
        labels["churn"] = ~labels["customer_id"].isin(active_customers)
        labels["cutoff_date"] = cutoff_date
        df_labels.append(labels)
    return pd.concat(df_labels, ignore_index=True)

labels = label_churn(df, SNAPSHOT_DATE, HORIZON_DAY, INACTIVITY_WINDOW)


# Đếm số lượng khách hàng theo từng cutoff_date
counts = labels.groupby('cutoff_date')[['customer_id']].count()

# Thêm cột phần trăm
counts['pct'] = (counts['customer_id'] / counts['customer_id'].sum()) * 100

def build_features(df, snapshot_date, horizon_day, inactivity_window):
    df_agg = []
    for i in range(5):
        cutoff_date = (snapshot_date - max(timedelta(horizon_day),timedelta(inactivity_window))).replace(day=1) - pd.DateOffset(months=i) - pd.DateOffset(days=1)
        hist = df[df["transaction_date"] <= cutoff_date]

        agg = hist.groupby("customer_id").agg(
            recency_days=("transaction_date", lambda x: (cutoff_date - x.max()).days),
            freq_30d=("transaction_date", lambda x: (x >= cutoff_date - timedelta(days=30)).sum()),
            freq_90d=("transaction_date", lambda x: (x >= cutoff_date - timedelta(days=90)).sum()),
            monetary_90d=("amount", lambda x: x[x.index >= x.index.max() - 90].sum())
        )

        agg["freq_ratio"] = agg["freq_30d"] / (agg["freq_90d"] + 1e-6)
        agg["cutoff_date"] = cutoff_date
        df_agg.append(agg)
    return pd.concat(df_agg).reset_index()

features = build_features(df, SNAPSHOT_DATE, HORIZON_DAY, INACTIVITY_WINDOW)

data = features.merge(labels, on=["customer_id","cutoff_date"])

import pandas as pd
import numpy as np

from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from lightgbm import LGBMClassifier

import shap
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


# In[37]:


# Temporal train / val / test split

test = data[data["cutoff_date"]==data["cutoff_date"].max()]
train = data[~data["cutoff_date"].isin(test["cutoff_date"])]

X_train = train.drop(columns=["customer_id", "churn", "cutoff_date"])
y_train = train["churn"].astype(int)

X_test = test.drop(columns=["customer_id", "churn", "cutoff_date"])
y_test = test["churn"].astype(int)


# In[46]:


model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)


# In[47]:


imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
imp


# In[40]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_reg.fit(X_train, y_train)


# In[41]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)

tree.fit(X_train, y_train)


# In[42]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,          # number of trees
    max_depth=5,               # same depth as your tree
    min_samples_leaf=50,       # same regularization
    max_features='sqrt',       # feature subsampling (important)
    random_state=42,
    n_jobs=-1                  # use all cores
)

rf.fit(X_train, y_train)


# In[43]:


from sklearn.metrics import precision_score, recall_score, classification_report
def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
    }


eval_table = pd.DataFrame.from_dict(
    {"Logistic Regression": evaluate(log_reg, X_test, y_test),
     "Decision Trees": evaluate(tree, X_test, y_test),
     "Random Forest": evaluate(rf, X_test, y_test),
     "LightBoost": evaluate(model, X_test, y_test),
    },
    orient="index"
).reset_index().rename(columns={"index": "model"})


def plot_confusion_mix(model, X, y):
    # 1. Get binary predictions (0 or 1)
    y_pred = model.predict(X)

    # 2. Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # 3. Plot using Seaborn for a clean look
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

from lifetimes.utils import summary_data_from_transaction_data

summary = summary_data_from_transaction_data(
    df,
    customer_id_col="customer_id",
    datetime_col="transaction_date",
    monetary_value_col="amount",
    observation_period_end="2025-12-31"
)

summary.head()

summary.reset_index().shape

from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(
    summary["frequency"],
    summary["recency"],
    summary["T"]
)

summary['p_alive'] = bgf.conditional_probability_alive(
    summary['frequency'],
    summary['recency'],
    summary['T']
)


summary['exp_txn_30d'] = bgf.predict(
    30,
    summary['frequency'],
    summary['recency'],
    summary['T']
)

churn_compare = pd.merge(rfm_q[['customer_id', 'churn_label']], summary.reset_index()[['customer_id', 'p_alive']], on='customer_id')

data = rfm_q.copy()

data['avg_amount'] = data['monetary']/data['frequency']

df1 = transactions.groupby('customer_id').agg(latest_txn_date=("transaction_date", max)).reset_index().merge(data,on='customer_id').merge(customers,on='customer_id')

df1['tenure'] = (
    (df1['latest_txn_date'] + pd.to_timedelta(INACTIVITY_WINDOW, unit='D'))
        .clip(upper=pd.Timestamp("2025-12-31"))
    - df1['signup_date']
).dt.days

from lifelines import CoxPHFitter

features = ["frequency", "avg_amount"]

cph = CoxPHFitter()
cph.fit(
    df1[features + ["tenure", "churn_label"]],
    duration_col="tenure",
    event_col="churn_label"
)

from lifelines.utils import concordance_index

risk_pred = cph.predict_partial_hazard(df1[features])

c_index = concordance_index(
    df1["tenure"],
    -risk_pred,     # negative because higher risk = earlier event
    df1["churn_label"]
)

from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

# Survival predictions
surv_funcs = cph.predict_survival_function(df1[features])

# VALID time grid (CRITICAL FIX)
t_min = df1.loc[df1["churn_label"] == 1, "tenure"].min()
t_max = df1["tenure"].max()

times = np.linspace(t_min, t_max * 0.999, 100)

# Convert survival funcs
surv_preds = np.asarray([
    np.interp(times, surv_funcs.index.values, surv_funcs.iloc[:, i].values)
    for i in range(surv_funcs.shape[1])
])

# IBS computation
y = Surv.from_arrays(
    event=df1["churn_label"].astype(bool),
    time=df1["tenure"]
)

ibs = integrated_brier_score(
    y,      # train
    y,      # test
    surv_preds,
    times
)

print(f"Integrated Brier Score: {ibs:.4f}")

import numpy as np

rows = []
horizons = [30, 60, 90]

for _, row in df1.iterrows():
    surv_fn = cph.predict_survival_function(
        row[features].to_frame().T
    )

    t = surv_fn.index.values
    s = surv_fn.values.flatten()

    row_out = {"customer_id": row["customer_id"]}

    for h in horizons:
        surv_h = np.interp(h, t, s)
        row_out[f"surv_{h}d"] = surv_h
        row_out[f"churn_{h}d"] = 1 - surv_h

    rows.append(row_out)

survival_df = pd.DataFrame(rows)
df1 = df1.merge(survival_df, on='customer_id')

from lifelines import WeibullAFTFitter

raft = WeibullAFTFitter()
raft.fit(
    df1[features + ["tenure", "churn_label"]],
    duration_col="tenure",
    event_col="churn_label"
)

def discounted_mean_residual_life(surv_col, t0, r):
    """
    surv_col: pd.Series, survival probabilities indexed by time
    t0: int, tenure at cutoff
    r: daily discount rate (e.g. 0.001)
    """

    # Ensure t0 is within survival horizon
    t0 = min(t0, surv_col.index.max())

    # Interpolate survival at t0
    S_t0 = np.interp(
        t0,
        surv_col.index.values,
        surv_col.values
    )

    # Future times
    mask = surv_col.index > t0
    t = surv_col.index[mask].values
    S = surv_col.loc[mask].values

    # Discounted area (time shifted to start at 0)
    discounted_area = np.trapz(
        S * np.exp(-r * (t - t0)),
        t
    )

    return discounted_area / S_t0

surv = raft.predict_survival_function(df1)

discounted_expected_remaining_days = []

for i, t0 in enumerate(df1["tenure"].values):
    surv_col = surv.iloc[:, i]
    discounted_expected_remaining_days.append(
        discounted_mean_residual_life(surv_col, t0, 0.01)
    )

df1["expected_remaining_days"] = discounted_expected_remaining_days

df1["expected_remaining_days"].describe()

from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0.01)

summary_gg = summary[summary["monetary_value"] > 0]

ggf.fit(
    summary_gg["frequency"],
    summary_gg["monetary_value"]
)

summary.loc[summary_gg.index, "expected_avg_order_value"] = (
    ggf.conditional_expected_average_profit(
        summary_gg["frequency"],
        summary_gg["monetary_value"]
    )
)

summary["CLV_3m"] = ggf.customer_lifetime_value(
    bgf,
    summary_gg["frequency"],
    summary_gg["recency"],
    summary_gg["T"],
    summary_gg["monetary_value"],
    time=3,               # 3 tháng
    discount_rate=0.01,   # discount rate
    freq="D"
)

summary["CLV_6m"] = ggf.customer_lifetime_value(
    bgf,
    summary_gg["frequency"],
    summary_gg["recency"],
    summary_gg["T"],
    summary_gg["monetary_value"],
    time=3,               # 6 tháng
    discount_rate=0.01,   # discount rate
    freq="D"
)

df1['CLV_3m'] = summary["CLV_3m"].values
df1['CLV_6m'] = summary["CLV_6m"].values

df1['expected_avg_transaction'] = summary['expected_avg_order_value'].values

df1['expected_remaining_lifetime_value'] = df1['expected_avg_transaction'] * df1['expected_remaining_days'] * df1['frequency']/df1['tenure']

df1['expected_remaining_lifetime_value'] = np.where(df1['churn_label'] == 1, 0, df1['expected_remaining_lifetime_value'])