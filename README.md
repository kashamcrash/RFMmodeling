# Build a RFM (Recency Frequency Monetary) Model for Retail Customers

## What does RFM Mean?
- Recency means the number of days since a customer made the last purchase. 
- Frequency is the number of purchase in a given period. It could be 3 months, 6 months or 1 year. Monetary is the total amount of money a customer spent in that given period. 
- Therefore, big spenders will be differentiated among other customers such as MVP (Minimum Viable Product) or VIP.

### Task in Hand
1. Perform cohort analysis (a cohort is a group of subjects that share a defining characteristic).
2. Create month cohorts and analyze active customers for each cohort.
3. Analyze the retention rate of customers.
4. Build RFM Segments. Give recency, frequency, and monetary scores individually by dividing them into quartiles.
5. Perform customer segmentation using RFM analysis and Kmeans.


#### What is the Dataset about?
* This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. 
* InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
* StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. 
* Description: Product (item) name. Nominal. 
* Quantity: The quantities of each product (item) per transaction. Numeric. 
* InvoiceDate: Invoice Date and time. Numeric, the day and time when each transaction was generated. 
* UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
* CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
* Country: Country name. Nominal, the name of the country where each customer resides.
