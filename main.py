import gzip
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "figure"
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(FIG_DIR, name+'.png')}")


reviews = pd.read_json("review-Hawaii_10.json", lines=True)
meta = pd.read_json("meta-Hawaii.json", lines=True)

plt.rcParams["figure.figsize"] = (8, 6)
business = pd.read_csv("business_cleaned.csv")
print("Loaded business_cleaned.csv")

import matplotlib.pyplot as plt

num_businesses = business.shape[0]
num_users = reviews["user_id"].nunique()
business_counts = reviews.groupby("gmap_id").size()



plt.figure(figsize=(8,4))
plt.hist(
    business_counts,
    bins=50,
    color="#add8e6",          # 淡蓝
    edgecolor='black',
    alpha=0.8,
    linewidth=0.7
)
plt.gca().patch.set_alpha(0)  # 透明背景
plt.xlabel("Number of Reviews per Business")
plt.ylabel("Number of Businesses")
plt.title("Business Review Count Distribution")
plt.yscale('log')
plt.grid(True, alpha=0.3)
save_fig("business_review_count_distribution")
plt.show()

# ---------- 4. Business average rating distribution ----------
business_avg_rating = business["avg_rating"].dropna()

plt.figure(figsize=(8,4))
plt.hist(
    business_avg_rating,
    bins=20,
    color="#f4a7b9",          # 淡红
    edgecolor='black',
    alpha=0.8,
    linewidth=0.7
)
plt.gca().patch.set_alpha(0)
plt.xlabel("Average Rating")
plt.ylabel("Number of Businesses")
plt.title("Business Average Rating Distribution")
plt.grid(True, alpha=0.3)
save_fig("business_avg_rating_distribution")
plt.show()

# # ---------- Summary ----------
# print("Business review count stats:\n", business_counts.describe())
# print("Business average rating stats:\n", business_avg_rating.describe())


import seaborn as sns

plt.figure(figsize=(8,4))

sns.histplot(
    business_avg_rating,
    bins=20,
    kde=True,
    color="#f4a7b9",             # 淡红
    edgecolor=(0, 0, 0, 0.5),     # 黑色，透明度 0.4
    linewidth=1,
)

# 强制修改 KDE 线的颜色（深红）
plt.gca().lines[-1].set_color("#b3003c")   # 深红
plt.gca().lines[-1].set_linewidth(2)

plt.xlabel("Average Rating")
plt.ylabel("Count")
plt.title("Business Average Rating Distribution (Histogram + KDE)")
plt.grid(True, alpha=0.3)

save_fig("business_avg_rating_hist_kde_red")
plt.show()

reviews["time_dt"] = pd.to_datetime(reviews["time"], unit="ms", errors="coerce")
reviews = reviews.dropna(subset=["time_dt"])
reviews["year_month"] = reviews["time_dt"].dt.to_period("M")
# ---------- 创建 figure 文件夹 ----------
FIG_DIR = "figure"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Filter reviews between 2017Q1 and 2020Q4 ----------
reviews_filtered = reviews[(reviews["time_dt"] >= "2017-01-01") & (reviews["time_dt"] <= "2020-12-31")].copy()

# ---------- Extract year-quarter ----------
reviews_filtered["quarter"] = reviews_filtered["time_dt"].dt.to_period("Q")

# ---------- Compute quarterly average rating ----------
quarterly_avg = reviews_filtered.groupby("quarter")["rating"].mean().reset_index()
quarterly_avg["quarter_dt"] = quarterly_avg["quarter"].dt.to_timestamp()

# ---------- Compute quarterly review counts ----------
quarterly_counts = reviews_filtered.groupby("quarter").size().reset_index(name="review_count")
quarterly_counts["quarter_dt"] = quarterly_counts["quarter"].dt.to_timestamp()

# ---------- Plot ----------
fig, ax1 = plt.subplots(figsize=(12,5), facecolor='white')  # White background

# Line 1: Average rating
ax1.plot(
    quarterly_avg["quarter_dt"],
    quarterly_avg["rating"],
    marker="o",
    color="#add8e6",  # 淡蓝折线
    markerfacecolor="#003f7f",  # 深蓝点
    linewidth=2,
    label="Average Rating"
)
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Average Rating", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.grid(alpha=0.3)

# Line 2: Review count on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(
    quarterly_counts["quarter_dt"],
    quarterly_counts["review_count"],
    marker="o",
    color="#f4a7b9",  # 淡红折线
    markerfacecolor="#b3003c",  # 深红点
    linewidth=2,
    label="Review Count"
)
ax2.set_ylabel("Review Count", color="orange")
ax2.tick_params(axis='y', labelcolor="orange")

# X-axis: quarterly labels
ax1.set_xticks(quarterly_avg["quarter_dt"])
ax1.set_xticklabels([str(q) for q in quarterly_avg["quarter"].astype(str)], rotation=45)

# Title
plt.title("Quarterly Average Rating and Review Count (2017Q1 – 2020Q4)")

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()

# ---------- Save figure ----------
plt.savefig(os.path.join(FIG_DIR, "quarterly_avg_rating_review_count.png"), dpi=300)

plt.show()

business_reviews = reviews_filtered.groupby("gmap_id").agg(
    num_reviews=("rating", "count"),
    avg_rating=("rating", "mean")
).reset_index()

# ---------- Plot scatter ----------
plt.figure(figsize=(10,6))
plt.scatter(
    business_reviews["num_reviews"],
    business_reviews["avg_rating"],
    alpha=0.6,
    s=20,
    color='#003f7f'
)
plt.xlabel("Number of Reviews")
plt.ylabel("Average Rating")
plt.title("Business Average Rating vs Number of Reviews")
plt.xscale('log')  # log scale for better visualization
plt.ylim(0,5)
plt.grid(True, alpha=0.3)
save_fig("Business Average Rating vs Number of Reviews")
plt.show()
