##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

###############################################################
# Business Problem
###############################################################
# FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases
# as OmniChannel (both online and offline shopper) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the most recent purchase was made
# first_order_date : Date of first purchase made by the customer
# last_order_date : Date of last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : The total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline : Total fee paid by the customer for offline purchases
# customer_value_total_ever_online : Total fee paid by the customer for online purchases
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months


###############################################################
# TASKS
###############################################################
# TASK 1: Preparing the Data
           # 1. Read the flo_data_20K.csv data. Make a copy of the dataframe.
           # 2. Define the outlier thresholds and replace with thresholds functions needed to outliers.
           # Note: When calculating cltv, the frequency values must be integers. Therefore, round the lower and upper limits with round().
           # 3. If the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" have outliers, replace them
           # 4. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spend.
           # 5. Examine the variable types. Change the type of variables that express date to date.

# TASK 2: Creating the CLTV Data Structure
           # 1.Take 2 days after the date of the last purchase in the data set as the date of analysis.
           # 2.Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
           # Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly terms.


# TASK 3: Establishment of BG/NBD and Gamma-Gamma Models, Calculation of CLTV
           # 1. Fit the BG/NBD model
                # a. Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.
                # b. Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
           # 2. Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.
           # 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
                # a. Observe the 20 people with the highest Cltv value.

# TASK 4: Creating Segments by CLTV
           # 1.Divide all your 6 month old customers into 4 groups (segments) and add the group names to the data set. Add it to the dataframe with the name cltv_segment.
           # 2.Make short 6-month action suggestions to the management for 2 groups that you will choose from among 4 groups.

# BONUS: Functionalize the whole process.


###############################################################
# TASK 1:  Preparing the Data
###############################################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

# 1. Read the flo_data_20K.csv data. Make a copy of the dataframe.
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)

check_df(df)

# 2.Define the outlier_thresholds and replace_with_thresholds functions needed to replace outliers
# Note: When calculating cltv, the frequency values must be integers. Therefore, round the lower and upper limits with round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. If the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" have outliers, replace them

df.describe()
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")

# 4. Omnichannel means that customers shop from both online and offline platforms. Create new variables for each customer's total purchases and spend.

df["customer_total_order"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
df["customer_total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

# 5.  Examine the variable types. Change the type of variables that express date to date.

df.dtypes
df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)
df["first_order_date"] = df["first_order_date"].apply(pd.to_datetime)
df["last_order_date_online"] = df["last_order_date_online"].apply(pd.to_datetime)
df["last_order_date_offline"] = df["last_order_date_offline"].apply(pd.to_datetime)


###############################################################
# TASK 2: Creating the CLTV Data Structure
###############################################################

# 1.Take 2 days after the date of the last purchase in the data set as the date of analysis.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 3)


# 2.Create a new cltv dataframe with master_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.

cltv_df = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: last_order_date,
                                       "first_order_date": lambda first_order_date: first_order_date,
                                       "customer_total_order": lambda customer_total_order: customer_total_order.nunique(),
                                       "customer_total_value": lambda customer_total_value: customer_total_value.sum()})

cltv_df.head()

cltv_df["recency_cltv_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days / 7
cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days / 7
cltv_df.head()
cltv_df = cltv_df.reset_index()


cltv_df.drop("last_order_date", axis=1, inplace=True)
cltv_df.drop("first_order_date", axis=1, inplace=True)

cltv_df.columns = ["master_id", "frequency", "monetary", "recency_cltv_weekly", "T_weekly"]
cltv_df.describe().T

cltv_df["monetary_cltv_avg"] = cltv_df["monetary"] / cltv_df["frequency"]


###############################################################
# TASK 3: Establishment of BG/NBD and Gamma-Gamma Models, Calculation of CLTV
###############################################################

# 1.  Fit the BG/NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])


# Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])


# Reviewing 10 people who will make the most purchases in the 3rd and 6th months

cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10).index
cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10).index


# 2. Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])


# 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

# Observe the 20 people with the highest Cltv value.

cltv_df.sort_values("cltv", ascending=False).head(20)


###############################################################
# TASK 4: Creating Segments by CLTV
###############################################################

# 1. Divide all your 6 month old customers into 4 groups (segments) and add the group names to the data set.
# Add it to the dataframe with the name cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2. Examine the recency, frequency and monetary averages of the segments.

cltv_df.groupby("cltv_segment").agg({"count", "sum", "mean"})


# BONUS: Functionalize the whole process.


def cltv_prediction(dataframe, month=3):

    # Preparing the data
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    df["customer_total_order"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
    df["customer_total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)
    df["first_order_date"] = df["first_order_date"].apply(pd.to_datetime)
    df["last_order_date_online"] = df["last_order_date_online"].apply(pd.to_datetime)
    df["last_order_date_offline"] = df["last_order_date_offline"].apply(pd.to_datetime)
    df["last_order_date"].max()
    today_date = dt.datetime(2021, 6, 3)

    cltv_df = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: last_order_date,
                                           "first_order_date": lambda first_order_date: first_order_date,
                                           "customer_total_order": lambda
                                               customer_total_order: customer_total_order.nunique(),
                                           "customer_total_value": lambda
                                               customer_total_value: customer_total_value.sum()})

    cltv_df["recency_cltv_weekly"] = (cltv_df["last_order_date"] - cltv_df["first_order_date"]).dt.days / 7
    cltv_df["T_weekly"] = (today_date - cltv_df["first_order_date"]).dt.days / 7
    cltv_df = cltv_df.reset_index()
    cltv_df.drop("last_order_date", axis=1, inplace=True)
    cltv_df.drop("first_order_date", axis=1, inplace=True)
    cltv_df.columns = ["master_id", "frequency", "monetary", "recency_cltv_weekly", "T_weekly"]
    cltv_df["monetary_cltv_avg"] = cltv_df["monetary"] / cltv_df["frequency"]

    # Fit the BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])

    cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency_cltv_weekly"],
                                                                                           cltv_df["T_weekly"])

    cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency_cltv_weekly"],
                                                                                           cltv_df["T_weekly"])

    # Fit the Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                           cltv_df["monetary_cltv_avg"])

    # Calculate 6 months CLTV with BG/NBD and Gamma-Gamma
    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency_cltv_weekly'],
                                                  cltv_df['T_weekly'],
                                                  cltv_df['monetary_cltv_avg'],
                                                  time=6,
                                                  freq="M",
                                                  discount_rate=0.01)

    # Creating Segments by CLTV
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df


df = df_.copy()

cltv_prediction(df)

