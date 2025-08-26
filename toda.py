import matplotlib.pyplot as plt
import seaborn as sns
import rfm 
import os
# Optional: Set style
sns.set(style="whitegrid")
output_dir = r"C:\Users\JustMe\commerce\processed"

# === Create folder for charts ===
charts_dir = os.path.join(output_dir, "charts")
os.makedirs(charts_dir, exist_ok=True)

# === RFM Segment Distribution (Bar Chart) ===
plt.figure(figsize=(8, 5))
segment_counts = rfm["Segment"].value_counts().sort_index()
sns.barplot(x=segment_counts.index, y=segment_counts.values, palette="viridis")
plt.title("Customer Count by RFM Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "rfm_segment_distribution.png"))
plt.close()

# === Histograms for Recency, Frequency, Monetary ===
for metric in ["Recency", "Frequency", "Monetary"]:
    if metric in rfm.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(rfm[metric].dropna(), bins=30, kde=True, color="skyblue")
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{metric.lower()}_histogram.png"))
        plt.close()

# === Pie Chart: Segment Breakdown ===
plt.figure(figsize=(6, 6))
rfm["Segment"].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Customer Segment Breakdown")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "segment_pie_chart.png"))
plt.close()

# === Top 10 Customers by CLV (Bar Chart) ===
if "CLV" in rfm.columns:
    top_clv = rfm.sort_values("CLV", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_clv["CLV"], y=top_clv.index, palette="mako")
    plt.title("Top 10 Customers by CLV")
    plt.xlabel("Customer Lifetime Value (CLV)")
    plt.ylabel("Customer ID")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "top_10_clv_customers.png"))
    plt.close()

print(f"[SUCCESS] Charts saved in: {charts_dir}")
