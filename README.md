# E-Commerce Customer Segmentation using RFM and Clustering

## Project Title
E-Commerce Customer Segmentation: Identifying High-Value Customers and Improving Retention using Machine Learning

## Business Problem
The e-commerce company wants to design targeted marketing campaigns, improve customer retention, and identify high-value customer groups to optimize marketing spend. Without understanding different customer personas, the company risks sending generic promotions that do not resonate with their audience, leading to lost revenue and potential customer churn. The goal is to use historical transaction data to segment customers based on their purchasing behavior.

## 1. Data Understanding
The dataset provides transactional information for an e-commerce platform. Here is a clear breakdown of the dataset characteristics:

- **What each column represents**:
  - `InvoiceNo`: Unique identifier for the transaction. If it starts with 'C', the order was cancelled or returned.
  - `StockCode`: Unique identifier for the specific product.
  - `Description`: The name or description of the product.
  - `Category`: The category the product belongs to (e.g., Grocery, Electronics).
  - `Quantity`: The number of items purchased in that specific transaction line.
  - `InvoiceDate`: The date and time the transaction occurred.
  - `UnitPrice`: The price per single unit of the product.
  - `CustomerID`: Unique identifier for the customer making the purchase.
  - `Country`: The country where the customer resides.

- **What a single row represents**:
  A single row represents a **single product line-item within a specific transaction**. This means if a customer buys 5 different products in one checkout, it will be recorded as 5 separate rows sharing the same `InvoiceNo` and `CustomerID`.

- **What type of business this dataset belongs to**:
  This dataset belongs to a **global e-commerce retail business** that sells a wide variety of consumer goods (like electronics, apparel, and home items) to customers across multiple countries.

- **What kind of customer or sales analysis can be done using this dataset**:
  - **RFM Analysis**: Segmenting customers based on Recency, Frequency, and Monetary value.
  - **Cohort Analysis**: Tracking customer retention over time.
  - **Market Basket Analysis**: Identifying which products are frequently bought together.
  - **Sales Trend Analysis**: Understanding seasonality and peak sales periods.

- **What business questions can be answered using this dataset**:
  - Who are our most valuable customers (VIPs) and who are at risk of churning?
  - Which countries generate the most revenue?
  - What are the top-selling products by quantity and revenue?
  - What is the average order value (AOV) per customer?

- **What business questions cannot be answered due to missing information**:
  - We cannot answer questions about **Customer Demographics** (e.g., age, gender, occupation) because that data is missing.
  - We cannot calculate **Profit Margins** because there is no data on the cost of goods sold (COGS).
  - We cannot determine **Marketing ROI** or **Customer Acquisition Cost** because we do not know which marketing channels brought the customers to the store.

## 2. Data Cleaning
Before diving into analysis, the dataset required rigorous cleaning to ensure data integrity. Here is a clear explanation of every cleaning step performed:

- **Missing customer IDs**: Dropped records where `CustomerID` was missing. Since the goal is customer segmentation, any transaction that cannot be tied to a specific customer is unusable.
- **Missing product descriptions**: Dropped rows with missing `Description`. These rows often correspond to system errors or unidentifiable items.
- **Negative or zero quantities**: Filtered out rows where `Quantity <= 0`. Negative quantities typically represent returns or errors, and zero quantities are invalid for sales analysis.
- **Zero or negative unit prices**: Filtered out rows where `UnitPrice <= 0`. These represent free items, adjustments, or errors, which distort revenue calculations.
- **Cancelled or returned invoices**: Removed all invoices where the `InvoiceNo` starts with the letter 'C'. These represent cancellations and should not be counted towards active sales or purchase frequency.
- **Duplicate records**: Removed exact duplicate rows to prevent artificially inflating a customer's purchase frequency or monetary value.
- **Incorrect data types**: Converted `InvoiceDate` from a string into a proper Datetime object for accurate Recency calculations. Converted `CustomerID` to a string since it is a categorical identifier, not a numeric metric.

## 3. Feature Engineering
To perform customer-level segmentation, the transaction data was aggregated from the invoice level to the customer level. We created the following specific features:

**Customer-Level Features:**
- **Total revenue per customer**: Calculated by summing `Quantity * UnitPrice` for all their purchases.
- **Total number of purchases**: The total count of transactions (invoices) associated with the customer.
- **Total quantity purchased**: The sum of all items bought by the customer across all orders.
- **Average order value (AOV)**: Total revenue divided by the total number of purchases.
- **Number of unique products purchased**: The count of distinct `StockCode`s the customer has bought.
- **Country of customer**: The country where the customer resides (using the first observed country per customer).

**RFM Table:**
We also created an RFM table to capture the core dimensions of customer behavior:
- **Recency**: How recently the customer made a purchase (calculated as the number of days between their last purchase and a "snapshot" date one day after the dataset's latest date).
- **Frequency**: How often the customer purchases (same as Total number of purchases).
- **Monetary**: How much the customer has spent (same as Total revenue per customer).

## 4. Exploratory Data Analysis
To deeply understand customer and sales behavior, we generated several visualizations (saved in the `/images` folder) to answer key business questions:

**1. Which countries generate the highest sales?**
![Top Countries Sales](images/top_countries_sales.png)
*Interpretation*: The United Kingdom is by far the most dominant market, generating the highest sales volume. It is followed by India, France, Germany, and the Netherlands, which represent strong secondary markets.

**2. Which products are sold the most?**
![Top Products by Quantity](images/top_products_quantity.png)
*Interpretation*: Items such as 'Organic Cotton T-Shirt' and 'Smart Watch Strap' are sold in the highest volumes, indicating strong consumer demand for everyday basics and tech accessories.

**3. Which products generate the highest revenue?**
![Top Products by Revenue](images/top_products_revenue.png)
*Interpretation*: While basics sell in high volume, high-ticket items like 'Running Shoes', 'Bluetooth Speakers', and 'Denim Jackets' drive the majority of top-line revenue.

**4. What is the distribution of customer purchase frequency & order value?**
![RFM Distributions](images/rfm_distributions.png)
![AOV Distribution](images/aov_distribution.png)
*Interpretation*: The distributions for Frequency, Monetary value, and Average Order Value are heavily right-skewed. The vast majority of customers purchase only 1-3 times and spend moderate amounts, whereas a small "long tail" of customers buys extremely often and spends vastly more.

**5. Are there outliers in quantity, price, or revenue?**
*Interpretation*: Yes, significant outliers exist. Prior to cleaning, there were extreme negative outliers (returns) and zero-priced items. Even after cleaning, the heavy right-skew in our distributions reveals positive outliers: a handful of customers (likely B2B wholesale buyers) who purchase items in massive quantities, generating exceptionally high revenue per order compared to the median shopper.

**6. Which customers appear to be high-value customers?**
*Interpretation*: High-value customers are easily identified in the extreme right tails of the Frequency and Monetary histograms. These are the VIP shoppers who place orders frequently and consistently have high Average Order Values, representing the core profitability of the business.

## 5. Customer Segmentation using K-Means
- Used the **K-Means algorithm** for clustering based on the RFM features.
- Applied **Log Transformation** to Recency, Frequency, and Monetary to normalize the heavily skewed distributions.
- **Standard Scaler** was used to bring all features to a common scale with mean=0 and variance=1.
- Determined the optimal number of clusters (K=4) using the **Elbow Method**.

## 6. Cluster Interpretation
1. **Cluster 0: Recent Average Buyers** (Avg Recency: ~18 days, Freq: 1.8, Spend: ~$10,500)
   - *Behavior*: Bought very recently but frequency is average.
   - *Value*: High potential to become loyal if engaged properly.
2. **Cluster 1: At-Risk Customers** (Avg Recency: ~168 days, Freq: 1.6, Spend: ~$12,270)
   - *Behavior*: Haven't purchased in a long time but have spent a moderate amount historically.
   - *Value*: Previous good spenders who are likely churning.
3. **Cluster 2: Inactive / Occasional Buyers** (Avg Recency: ~176 days, Freq: 1.1, Spend: ~$2,655)
   - *Behavior*: Longest time since last purchase, lowest frequency, and lowest total spend.
   - *Value*: Low value, likely one-off buyers who haven't returned.
4. **Cluster 3: High Value / Loyal Customers** (Avg Recency: ~78 days, Freq: 3.4, Spend: ~$26,013)
   - *Behavior*: Highest frequency and highest monetary spend.
   - *Value*: The most valuable segment for the business. They generate the most revenue and shop often.

## 7. Business Recommendations
- **For High Value / Loyal Customers (Cluster 3)**: Implement an exclusive VIP loyalty program. Offer early access to new products, personalized shopping experiences, and premium support.
- **For Recent Average Buyers (Cluster 0)**: Send targeted cross-sell and up-sell email campaigns to increase their frequency. Provide a small discount on their next purchase to turn them into frequent shoppers.
- **For At-Risk Customers (Cluster 1)**: Send aggressive "We miss you" win-back campaigns with high-value discounts or personalized product recommendations based on their past purchases.
- **For Inactive Buyers (Cluster 2)**: Do not spend heavy marketing budget here. Limit to generalized, low-cost marketing like generic newsletters or seasonal sale announcements.
- **General Strategy**: Since the UK dominates sales, allocate the majority of the marketing budget locally. Consider creating localized, country-specific campaigns to boost sales in growing markets like India and France.

## How to Run the Project
1. Clone this repository to your local machine.
2. Ensure you have the necessary dependencies installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script to clean data, perform EDA, and generate customer segments:
   ```bash
   python main.py
   ```
4. The output visualizations will be saved in the `images/` directory, and the final segmented dataset will be available in the `outputs/` directory.
