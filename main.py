import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Create directories for outputs
os.makedirs('images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("Starting E-Commerce Customer Segmentation Project...")

# 1. Load Data
print("\n--- Loading Data ---")
df = pd.read_csv('dataset/part_1_ecommerce_customer_segmentation.csv')
print(f"Initial Dataset Shape: {df.shape}")

# 2. Data Cleaning
print("\n--- Data Cleaning ---")
# Drop duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Dropped {initial_rows - len(df)} duplicate records.")

# Handle missing values
missing_customers = df['CustomerID'].isnull().sum()
missing_descriptions = df['Description'].isnull().sum()
print(f"Missing Customer IDs: {missing_customers}")
print(f"Missing Descriptions: {missing_descriptions}")
df = df.dropna(subset=['CustomerID', 'Description'])
print(f"Dataset Shape after dropping missing values: {df.shape}")

# Convert data types
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(str)

# Handle Negative/Zero Quantities and Unit Prices, and Cancelled Invoices
# Cancelled invoices usually start with 'C'
cancelled_invoices = df[df['InvoiceNo'].str.startswith('C', na=False)]
print(f"Found {len(cancelled_invoices)} cancelled invoices. Removing them.")
df = df[~df['InvoiceNo'].str.startswith('C', na=False)]

zero_neg_qty = df[df['Quantity'] <= 0]
print(f"Found {len(zero_neg_qty)} records with zero or negative quantity. Removing them.")
df = df[df['Quantity'] > 0]

zero_neg_price = df[df['UnitPrice'] <= 0]
print(f"Found {len(zero_neg_price)} records with zero or negative price. Removing them.")
df = df[df['UnitPrice'] > 0]

print(f"Cleaned Dataset Shape: {df.shape}")

# 3. Feature Engineering
print("\n--- Feature Engineering ---")
df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']

# Snapshot date for Recency calculation (1 day after the latest invoice)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by CustomerID
customer_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique', # Frequency (number of purchases/orders)
    'TotalRevenue': 'sum', # Monetary (total spend)
    'Quantity': 'sum', # Total quantity purchased
    'StockCode': 'nunique', # Number of unique products purchased
    'Country': 'first' # Country of customer
}).reset_index()

# Rename columns
customer_data.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalRevenue': 'Monetary'
}, inplace=True)

# Calculate Average Order Value (AOV)
customer_data['AverageOrderValue'] = customer_data['Monetary'] / customer_data['Frequency']
customer_data['TotalPurchases'] = customer_data['Frequency']

print(f"Customer Level Data Shape: {customer_data.shape}")

# 4. Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")

# Which countries generate the highest sales?
country_sales = df.groupby('Country')['TotalRevenue'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=country_sales.values, y=country_sales.index, palette='viridis')
plt.title('Top 10 Countries by Total Sales')
plt.xlabel('Total Revenue')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('images/top_countries_sales.png')
plt.close()

# Which products are sold the most? (Quantity)
product_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=product_qty.values, y=product_qty.index, palette='mako')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Total Quantity')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('images/top_products_quantity.png')
plt.close()

# Which products generate the highest revenue?
product_rev = df.groupby('Description')['TotalRevenue'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=product_rev.values, y=product_rev.index, palette='rocket')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('images/top_products_revenue.png')
plt.close()

# Distribution of Recency, Frequency, Monetary
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(customer_data['Recency'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Recency')
axes[0].set_xlabel('Recency (Days)')
axes[0].set_ylabel('Number of Customers')
sns.histplot(customer_data[customer_data['Frequency'] < 50]['Frequency'], bins=30, kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title('Distribution of Frequency (Zoomed in)')
axes[1].set_xlabel('Frequency (Number of Purchases)')
axes[1].set_ylabel('Number of Customers')
sns.histplot(customer_data[customer_data['Monetary'] < 20000]['Monetary'], bins=30, kde=True, ax=axes[2], color='salmon')
axes[2].set_title('Distribution of Monetary (Zoomed in)')
axes[2].set_xlabel('Monetary Value')
axes[2].set_ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('images/rfm_distributions.png')
plt.close()

# Distribution of Average Order Value
plt.figure(figsize=(8, 5))
sns.histplot(customer_data['AverageOrderValue'], bins=40, kde=True, color='purple')
plt.title('Distribution of Average Order Value')
plt.xlabel('Average Order Value')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('images/aov_distribution.png')
plt.close()

# 5. Customer Segmentation using K-Means
print("\n--- Customer Segmentation ---")

# We will use Recency, Frequency, and Monetary for clustering
features = ['Recency', 'Frequency', 'Monetary']
X = customer_data[features]

# Because RFM data is heavily skewed, we apply log transformation
# We add a small constant (1) to handle zero values
X_log = np.log1p(X)

# Normalize/Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Elbow Method to find the optimal number of clusters (K)
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('images/elbow_method.png')
plt.close()

# From elbow plot, let's select K=4 (a common choice for RFM)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"Assigned {optimal_k} clusters to the customers.")

# 6. Cluster Interpretation & Business Analysis
print("\n--- Cluster Interpretation & Business Analysis ---")

# Define meaningful cluster names based on RFM characteristics
# Note: These names are mapped based on the interpretation of the cluster centroids
cluster_map = {
    0: 'Potential Loyalists',
    1: 'At-Risk Customers',
    2: 'Inactive / Lost',
    3: 'Champions'
}
customer_data['Segment'] = customer_data['Cluster'].map(cluster_map)

# Calculate Business Metrics per Segment
segment_analysis = customer_data.groupby('Segment').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum']
}).reset_index()

# Flatten columns
segment_analysis.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Spending', 'Total_Revenue']

# Calculate percentages for business impact
total_revenue = segment_analysis['Total_Revenue'].sum()
total_customers = segment_analysis['Customer_Count'].sum()

segment_analysis['Revenue_Contribution_%'] = (segment_analysis['Total_Revenue'] / total_revenue * 100).round(2)
segment_analysis['Customer_Base_%'] = (segment_analysis['Customer_Count'] / total_customers * 100).round(2)

# Sort by Revenue Contribution for the BA
segment_analysis = segment_analysis.sort_values(by='Total_Revenue', ascending=False)

print("\n--- Segment Business Summary ---")
print(segment_analysis[['Segment', 'Customer_Count', 'Customer_Base_%', 'Total_Revenue', 'Revenue_Contribution_%']])

# 7. Visualizing Segments for Business Stakeholders
plt.figure(figsize=(12, 7))
sns.scatterplot(x='Recency', y='Monetary', hue='Segment', data=customer_data, 
                palette='viridis', s=100, alpha=0.7, style='Segment')
plt.title('Business Segments: Recency vs Monetary Value', fontsize=15)
plt.xlabel('Recency (Days since last purchase)', fontsize=12)
plt.ylabel('Total Spend (Monetary Value)', fontsize=12)
plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('images/business_segments_scatter.png')
plt.close()

# Revenue vs Customer Count comparison
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

sns.barplot(x='Segment', y='Total_Revenue', data=segment_analysis, ax=ax1, palette='Blues_d', alpha=0.7)
sns.lineplot(x='Segment', y='Customer_Count', data=segment_analysis, ax=ax2, color='red', marker='o', linewidth=2)

ax1.set_title('Revenue Contribution vs. Customer Count by Segment', fontsize=15)
ax1.set_ylabel('Total Revenue', color='blue', fontsize=12)
ax2.set_ylabel('Number of Customers', color='red', fontsize=12)
plt.tight_layout()
plt.savefig('images/revenue_vs_customers.png')
plt.close()

# 8. Save Final Business Outputs
# Save the full dataset with segment names
customer_data.to_csv('outputs/customer_segments_final.csv', index=False)

# Save the summary report for Excel/PowerPoint
segment_analysis.to_csv('outputs/business_summary_report.csv', index=False)

print("\nBusiness analysis complete.")
print("- Final segmented data: outputs/customer_segments_final.csv")
print("- Business summary report: outputs/business_summary_report.csv")
print("- New business visualizations saved to /images")
print("Project Execution Completed Successfully.")
