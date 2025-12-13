import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 文件夹 ----------
FIG_DIR = "figure"
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- 读取数据 ----------
reviews = pd.read_json("review-Hawaii_10.json", lines=True)
meta = pd.read_json("meta-Hawaii.json", lines=True)
business = pd.read_csv("business_cleaned.csv")  # 如果需要

# ---------- 时间处理 ----------
reviews["time_dt"] = pd.to_datetime(reviews["time"], unit="ms", errors="coerce")
reviews = reviews.dropna(subset=["time_dt"])
reviews["year"] = reviews["time_dt"].dt.year

# 只保留 2017-2021 年
reviews_filtered = reviews[(reviews["year"] >= 2017) & (reviews["year"] <= 2021)].copy()

# 只保留合法评分 1~5
reviews_filtered = reviews_filtered[(reviews_filtered["rating"] >= 1) & (reviews_filtered["rating"] <= 5)]

# ---------- 历年评分和评论数 ----------
by_biz_year = (
    reviews_filtered.groupby(["gmap_id", "year"])["rating"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_rating_year", "count": "num_reviews_year"})
)

# pivot 表格
year_pivot = by_biz_year.pivot(index="gmap_id", columns="year", values="avg_rating_year")
year_pivot.columns = [f"avg_{int(c)}" for c in year_pivot.columns]
year_pivot = year_pivot.fillna(0).reset_index()  # NaN 填 0

review_counts = by_biz_year.pivot(index="gmap_id", columns="year", values="num_reviews_year")
review_counts.columns = [f"num_reviews_{int(c)}" for c in review_counts.columns]
review_counts = review_counts.fillna(0).reset_index()

# ---------- 合并评分和评论数 ----------
data = year_pivot.merge(review_counts, on="gmap_id", how="left")

# ---------- 地理特征 ----------
meta_geo = meta[["gmap_id", "latitude", "longitude"]].drop_duplicates()
data = data.merge(meta_geo, on="gmap_id", how="left")
data = data[data["latitude"].notnull() & data["longitude"].notnull()].copy()

# ---------- KMeans 区域聚类 ----------
coords = data[["latitude", "longitude"]].to_numpy()
K = 20
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
data["region_cluster"] = kmeans.fit_predict(coords)

# 区域平均评论数
data["regional_avg_reviews"] = data.groupby("region_cluster")["num_reviews_2017"].transform("mean")

# ---------- 类别特征 ----------
meta_small = meta[["gmap_id", "category", "num_of_reviews"]].copy()

def get_primary_category(cat_list):
    if isinstance(cat_list, list) and len(cat_list) > 0:
        return cat_list[0]
    return "Unknown"

meta_small["primary_category"] = meta_small["category"].apply(get_primary_category)
meta_small.drop(columns=["category"], inplace=True)
meta_small["num_of_reviews"] = meta_small["num_of_reviews"].fillna(0)

data = data.merge(meta_small, on="gmap_id", how="left")

# 类别平均评分
category_avg = data.groupby("primary_category")[["avg_2017","avg_2018","avg_2019","avg_2020"]].mean().mean(axis=1).reset_index()
category_avg.columns = ["primary_category","category_avg_rating"]
data = data.merge(category_avg, on="primary_category", how="left")

# ---------- 衍生特征 ----------
# 历年评分标准差
data["rating_std"] = data[["avg_2017","avg_2018","avg_2019","avg_2020"]].std(axis=1)
# 历年评分变化趋势
data["rating_slope"] = data["avg_2020"] - data["avg_2017"]
# 历年评论占比
total_reviews = data[["num_reviews_2017","num_reviews_2018","num_reviews_2019","num_reviews_2020"]].sum(axis=1)
data["reviews_prop_2017_2020"] = np.where(
    total_reviews > 0,
    data["num_reviews_2017"] / total_reviews,
    0
)

# ---------- 筛选活跃商家 ----------
active_mask = total_reviews >= 20
data = data[active_mask].copy()

# ---------- 标签 ----------
y = data["avg_2021"].values

data["total_reviews_2017_2020"] = (
    data["num_reviews_2017"] +
    data["num_reviews_2018"] +
    data["num_reviews_2019"] +
    data["num_reviews_2020"]
)
# ---------- 数值特征 ----------
numeric_features = [
    "avg_2017","avg_2018","avg_2019","avg_2020",
    "regional_avg_reviews","category_avg_rating","rating_std","rating_slope",
    "reviews_prop_2017_2020","num_of_reviews",
    "latitude","longitude",
    "total_reviews_2017_2020"
]

X_num = data[numeric_features].copy()
X_num = X_num.fillna(X_num.mean())
from sklearn.preprocessing import MinMaxScaler
# 标准化
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
# ---------- 保存权重信息（原始值，不标准化）----------
weights_df = pd.DataFrame({
    'total_reviews_2017_2020': data["total_reviews_2017_2020"],
    'num_of_reviews': data["num_of_reviews"],
    'rating_std': data["rating_std"]
})
# ---------- 保存 X 和 y ----------
X_scaled.to_csv(os.path.join(DATA_DIR, "X_numeric.csv"), index=False)
pd.Series(y, name="avg_2021").to_csv(os.path.join(DATA_DIR, "y_label.csv"), index=False)
weights_df.to_csv(os.path.join(DATA_DIR, "sample_weights.csv"), index=False)  # 新增

joblib.dump(X_scaled, os.path.join(DATA_DIR, "X_numeric.joblib"))
joblib.dump(y, os.path.join(DATA_DIR, "y_label.joblib"))

# ---------- 相关性矩阵 ----------
corr_matrix = data[numeric_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_matrix_numeric.png"), dpi=300)
plt.show()

print("Feature matrix shape:", X_scaled.shape)
print("Label shape:", y.shape)
