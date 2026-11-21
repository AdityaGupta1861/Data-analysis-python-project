
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("rfm_retail_uncleaned_600.csv")

print("Original Data:", df.shape)

# 2. Data Cleaning

# Remove duplicates and extra spaces
df = df.drop_duplicates()

for col in ["InvoiceNo", "CustomerID", "ProductDescription",
            "Category", "Country", "PaymentMethod"]:
    df[col] = df[col].astype("string").str.strip()

# Replace invalid values
df = df.replace(
    ["?", "Missing Value", "missing value", "NA", "N/A", "null", ""],
    pd.NA
)

# Clean Customer IDs
df["CustomerID"] = (
    df["CustomerID"].astype("string")
    .str.extract(r"(\d+)", expand=False)
    .apply(lambda x: "CUST-" + x if pd.notna(x) else pd.NA)
)

# Convert date and numeric columns
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], format="mixed", errors="coerce"
)

for col in ["Quantity", "UnitPrice", "TotalAmount"]:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace(
            r"[^0-9.\-]", "", regex=True
        ),
        errors="coerce"
    )

# Fill missing TotalAmount
df["TotalAmount"] = df["TotalAmount"].fillna(
    df["Quantity"] * df["UnitPrice"]
)

# Standardize categories and countries
df["Category"] = df["Category"].str.title().fillna("Unknown")
df["Country"] = df["Country"].str.title().fillna("Unknown")

# Fill missing descriptions and payment methods
df["ProductDescription"] = df["ProductDescription"].fillna("Unknown")
df["PaymentMethod"] = df["PaymentMethod"].str.title().fillna("Unknown")

# Remove invalid records
df = df.dropna(subset=[
    "InvoiceNo", "CustomerID", "InvoiceDate", "TotalAmount"
])

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["TotalAmount"] > 0)
]

df = df.drop_duplicates()

print("Cleaned Data:", df.shape)

# Save cleaned dataset
df.to_csv("rfm_retail_cleaned.csv", index=False)


# 3. RFM Calculation

today = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (today - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]


# 4. RFM Scoring

rfm["R"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    4, labels=[4, 3, 2, 1]
).astype(int)

rfm["F"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    4, labels=[1, 2, 3, 4]
).astype(int)

rfm["M"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    4, labels=[1, 2, 3, 4]
).astype(int)


# 5. Customer Segmentation

def segment(row):
    if row["R"] >= 4 and row["F"] >= 4 and row["M"] >= 4:
        return "Champions"
    elif row["F"] >= 3 and row["M"] >= 3:
        return "Loyal Customers"
    elif row["R"] <= 2:
        return "At Risk"
    elif row["M"] >= 4:
        return "Big Spenders"
    else:
        return "Regular Customers"

rfm["Segment"] = rfm.apply(segment, axis=1)


# 6. Display Results

print("\nRFM Analysis:")
print(rfm)

print("\nCustomer Segments:")
print(rfm["Segment"].value_counts())


# 7. Graphs

# Bar Chart
rfm["Segment"].value_counts().plot(
    kind="bar", title="Customer Segmentation"
)
plt.xlabel("Customer Groups")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Pie Chart
rfm["Segment"].value_counts().plot(
    kind="pie", autopct="%1.1f%%",
    title="Customer Segment Distribution"
)
plt.ylabel("")
plt.tight_layout()
plt.show()

# Monetary Distribution
rfm["Monetary"].plot(
    kind="hist", bins=15,
    title="Customer Spending Distribution"
)
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# 8. Save Final Results

rfm.to_csv("customer_segmentation_result.csv")

print("\nProject Completed Successfully!")
