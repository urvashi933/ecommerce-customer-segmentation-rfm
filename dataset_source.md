# Dataset Source

The dataset used in this project is an E-Commerce transaction dataset. 

**Source Link:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1XC-00liRViTlyeFaig3mYTkQcBrheph6?usp=sharing)

## Dataset Details
This dataset contains transactional records of an e-commerce platform. It provides line-item details for individual purchases made by customers.

**Key Columns:**
- **InvoiceNo**: The unique invoice number for the transaction.
- **StockCode**: Unique identifier for the item/product.
- **Description**: The name or description of the product.
- **Category**: Product category (e.g., Grocery, Electronics, Apparel).
- **Quantity**: The number of units purchased in a single transaction line.
- **InvoiceDate**: The timestamp of when the purchase was made.
- **UnitPrice**: The price of a single unit of the product.
- **CustomerID**: The unique identifier for the customer making the purchase.
- **Country**: The country where the customer resides.

## Usage in this Project
This dataset is used to perform:
1. **Data Cleaning**: Handling missing values, cancellations, and data type conversions.
2. **Feature Engineering**: Creating Recency, Frequency, and Monetary (RFM) variables.
3. **Exploratory Data Analysis**: Uncovering insights into top-performing products and regional sales.
4. **Customer Segmentation**: Using K-Means clustering to identify different customer personas based on their purchasing behavior.
