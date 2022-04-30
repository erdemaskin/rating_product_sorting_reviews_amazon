###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

################################################ #
# Business Problem
################################################ #

# One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
# The solution to this problem is to provide more customer satisfaction for the e-commerce site, to make the product stand out for the sellers and to buy
# means a seamless shopping experience for buyers. Another problem is the correct ordering of the comments given to the products.
# It appears as #. Financial loss due to the fact that misleading comments will directly affect the sale of the product.
# will cause loss of customers as well. In the solution of these 2 basic problems, while the e-commerce site and the sellers increase their sales, the customers
# will complete the purchasing journey without any problems.

################################################ #
# Dataset Story
################################################ #

# This data set, which includes Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.

# Variables:
# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since evaluation
# helpful_yes - The number of times the review was found helpful
# total_vote - Number of votes given to the review


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

################################################ #
# TASK 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
################################################ #

# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the given points by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.


#################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
#################################################

df = pd.read_csv("week_4/amazon_review.csv")
df["overall"].mean()
df.head()

#################################################
# Step 2: Calculate the Weighted Average of Score by Date.
#################################################

# To calculate weighted points according to dates:
# - declare reviewTime variable as date variable
# - accept the max value of reviewTime as current_date
# - create a new variable by expressing the difference of each point-comment date and current_date in days
# - and divide the variable expressed in days by 4 with the quantile function (giving 3 quarters, then 4 parts)
# - you need to weight based on values from quartiles.
# - for example, if q1 = 12, then average the comments made less than 12 days ago when weighting
# - like giving high weight.


# day_diff: how many days have passed since the comment
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime(str(df['reviewTime'].max()))
df["day_diff"] = (current_date - df['reviewTime']).dt.days

# determination of time-based average weights
def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w4 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)), "overall"].mean() * w5 / 100

time_based_weighted_average(df)
df["overall"].mean()
# Step 3: Compare and interpret the average of each time period in weighted scoring.

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.2), "overall"].mean()
# 4.6802

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.2)) & (df["day_diff"] <= df["day_diff"].quantile(0.4)), "overall"].mean()
# 4.6916

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.4)) & (df["day_diff"] <= df["day_diff"].quantile(0.6)), "overall"].mean()
# 4.5787

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.6)) & (df["day_diff"] <= df["day_diff"].quantile(0.8)), "overall"].mean()
# 4.5517

df.loc[df["day_diff"] > df["day_diff"].quantile(0.8), "overall"].mean()
# 4.4346



################################################ #
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
################################################ #


################################################ #
# Step 1. Generate the variable helpful_no
################################################ #

# Note:
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()
df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]
df.head()
################################################ #
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and Add to Data
################################################ #

def wilson_lower_bound(up, down, confidence=0.95):
    """
         Calculate Wilson Lower Bound Score

         - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
         - The score to be calculated is used for product ranking.
         - Note:
         If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
         This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

         parameters
         ----------
         up: int
             up count
         down: int
             down count
         confidence: float
             confidence

         Returns
         -------
         wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values(by="wilson_lower_bound", ascending=False)
df["wilson_lower_bound"].hist (bins=50)
plt.show();

# import numpy as np
# 0.85253 * (1/np.sqrt(495))
# z * karekök(sigma^2 / n)              z = 2 genelde, sigma = 0.5 genelde
# 0.81858 - 0.85253
#
# 0.06780 * (1/np.sqrt(118))
# 0.03475 - 0.06780
wilson_lower_bound (15, 15)
wilson_lower_bound (400, 400)
wilson_lower_bound(1000, 1000)
################################################
# Step 3. Identify 20 Interpretations and Interpret Results.
################################################ #

df.sort_values("wilson_lower_bound", ascending=False).head(20)