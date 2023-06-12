# CLTV Prediction with BG-NBD and Gamma-Gamma

## Business Problem

FLO wants to set a roadmap for sales and marketing activities.
In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

## Dataset Story

The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.


## Variables

- master_id: Unique customer number
- order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of first purchase made by the customer
- last_order_date : Date of last purchase made by the customer
- last_order_date_online : The date of the last purchase made by the customer on the online platform
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : The total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : The total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
- customer_value_total_ever_online : Total fee paid by the customer for online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


## TASKS

#### TASK 1: Preparing the Data
- 1- Read the flo_data_20K.csv data. Make a copy of the dataframe.
- 2- Define the outlier thresholds and replace with thresholds functions needed to outliers.
- Note: When calculating cltv, the frequency values must be integers. Therefore, round the lower and upper limits with round().
- 3- If the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" have outliers, replace them
- 4- Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spend.
- 5- Examine the variable types. Change the type of variables that express date to date.


#### TASK 2: Creating the CLTV Data Structure
- 1- Take 2 days after the date of the last purchase in the data set as the date of analysis.
- 2- Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
- 3- Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly terms.


#### TASK 3: Establishment of BG/NBD and Gamma-Gamma Models, Calculation of CLTV
- 1- Fit the BG/NBD model
    - a- Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
    - b- Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
- 2- Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.
- 3- Calculate 6 months CLTV and add it to the dataframe with the name cltv.
    - a- Observe the 20 people with the highest Cltv value.


#### TASK 4: Creating Segments by CLTV
- 1- Divide all your 6 month old customers into 4 groups (segments) and add the group names to the data set. Add it to the dataframe with the name cltv_segment.
- 2- Make short 6-month action suggestions to the management for 2 groups that you will choose from among 4 groups.

### BONUS: Functionalize the whole process.
