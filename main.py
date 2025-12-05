import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# ---- 配置 ----
DATA_FILE = "Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States.csv"
OUT_DIR = "figures"
PLOT_DPI = 150

# ---- 保存图片函数 ----
def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close(fig)
    print("Saved:", path)

# ---- 数据清洗函数 ----
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("原始数据形状:", df.shape)

    # 删除缺失值
    df = df.dropna(subset=["ESTIMATE", "UNIT", "YEAR", "AGE"])

    # 删除 INDICATOR 列，如果存在
    if "INDICATOR" in df.columns:
        df = df.drop(columns=["INDICATOR"])

    # 转换数据类型
    df["ESTIMATE"] = pd.to_numeric(df["ESTIMATE"].astype(str).str.replace(",", ""), errors="coerce")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype('Int64')

    # 清洗后统计
    print("清洗后数据形状:", df.shape)
    return df

# ---- 绘制按年龄层折线图 ----
def plot_age_adjusted_with_range(df, save_name):
    # 筛选数据
    df_crude = df[df["UNIT"] == "Deaths per 100,000 resident population, crude"]
    df_ageadj = df[df["UNIT"] == "Deaths per 100,000 resident population, age-adjusted"]

    # 按年份计算粗死亡率各年龄层的最小值和最大值
    crude_stats = df_crude.groupby("YEAR")["ESTIMATE"].agg(["min", "max"])
    # 按年份计算 age-adjusted 平均值
    ageadj_mean = df_ageadj.groupby("YEAR")["ESTIMATE"].mean()

    # 绘图
    fig, ax = plt.subplots(figsize=(12,6))

    # 阴影区域表示当年不同年龄层粗死亡率的范围（淡红色）
    ax.fill_between(crude_stats.index, crude_stats["min"], crude_stats["max"],
                    color="lightcoral", alpha=0.3, label="Crude rate range (all ages)")

    # 折线表示 age-adjusted
    ax.plot(ageadj_mean.index, ageadj_mean.values, color="blue", marker="o", label="Age-adjusted rate")

    # 标注 age-adjusted 折线的最大值
    max_year = ageadj_mean.idxmax()
    max_value = ageadj_mean.max()
    ax.scatter(max_year, max_value, color="blue", s=80, zorder=5)
    ax.text(max_year, max_value + 0.5, f"Max value = {max_value:.1f}", color="blue", fontsize=10, ha="center")

    ax.set_title("Drug Overdose Death Rate (1999-2019)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Deaths per 100,000 population")
    ax.grid(True)
    ax.legend(loc="upper left")

    # x 坐标显示整数年份
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    save_fig(fig, save_name)

# ---- 绘制年龄 × 年份热图 ----
def plot_age_year_heatmap(df, unit_name, title, save_name):
    df_unit = df[df["UNIT"] == unit_name]

    # pivot table: AGE × YEAR
    pivot = df_unit.pivot_table(index="AGE", columns="YEAR", values="ESTIMATE", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.3*pivot.shape[0])))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".1f", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Age Group")

    save_fig(fig, save_name)

# ---- 主函数 ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读取并清洗数据
    df = load_and_clean_data(DATA_FILE)

    plot_age_adjusted_with_range(df, "age_adjusted_with_range.png")
    print("\nVisualization saved to", OUT_DIR)

# ---- 程序入口 ----
if __name__ == "__main__":
    main()
