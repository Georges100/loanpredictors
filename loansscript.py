#load file
import pandas as pd
import csv
loans_2007=pd.read_csv("/users/jorge/documents/data/loans2007.csv", encoding ="ISO-8859-1")
print(loans_2007.head())

#clean na
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)

#replace and categorize the target variable
loans_2007 = loans_2007[(loans_2007["loan_status"] == "Fully Paid") | (loans_2007["loan_status"] == "Charged Off")]

status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0
    }
}

loans_2007=loans_2007.replace(status_replace)
#save first draft
loans_2007.to_csv('loans_2007.csv', index=False)
#count by grade
loans_2007["grade"].value_counts()
#grade A
loansA=loans_2007[loans_2007["grade"]=="A"]
#grade A or B
loansA=loans_2007[(loans_2007["grade"]=="A") & (loans_2007["grade"]=="B")]
#clean interest rate
loans_2007["int_rate"] = loans_2007["int_rate"].str.rstrip("%").astype("float")
#interest rate for A grade and total
loansA["int_rate"].mean()
loans_2007["int_rate"].mean()
#loans not A nor B 
loansnotAB=loans_2007[(loans_2007["grade"]!="A") & (loans_2007["grade"]!="B")]
#loans not A 
loansnotA=loans_2007[loans_2007["grade"]!="A"]
#drop variables from file
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)

#clean nulls
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)

#save filtered version
loans_2007.to_csv('filtered_loans_2007.csv', index=False)

#read file and clean nulls
loans=pd.read_csv("filtered_loans_2007.csv")
null_counts=loans.isnull().sum()
loans=loans.drop("pub_rec_bankruptcies",axis=1)

loans=loans.dropna(axis=0)
loans=loans.drop("pymnt_plan",axis=1)
#create num categories from objects
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for c in cols:
    print(loans[c].value_counts())
#employment length to numbers
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
loans = loans.replace(mapping_dict)

#add dummy categories
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_columns])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)

#save cleaned file
loans.to_csv('cleaned_loans_2007.csv', index=False)
#read final file
loans=pd.read_csv("cleaned_loans_2007.csv")

#forecasting models
#select columns and target
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]


import numpy
lr = LogisticRegression(class_weight="balanced")
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)

specificity= tn / (fp + tn) 

accuracy=(tp + tn) / loans.shape[0]
ppv=tp/ (tp + fp)
npv= tn/ (tn + fn)
print( specificity)
print(accuracy)
print( ppv)
print( npv)

#roc auc score
auc_score = roc_auc_score(loans["loan_status"], predictions)
print(auc_score)

#with penalty
penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight=penalty)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)

#training set

import math
import random
from numpy.random import permutation
random_indices=permutation(loans.index)
test_cutoff=math.floor(len(loans)/3)
test=loans.loc[random_indices[1:test_cutoff]]
train=loans.loc[random_indices[test_cutoff:]]

#columns used
cols=['loan_amnt', 'int_rate','installment', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc','pub_rec', 'revol_bal', 'revol_util', 'total_acc','home_ownership_MORTGAGE',  'home_ownership_OWN', 'home_ownership_RENT','verification_status_Not Verified','verification_status_Source Verified', 'verification_status_Verified','purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation','purpose_educational', 'purpose_home_improvement', 'purpose_house','purpose_major_purchase', 'purpose_medical', 'purpose_moving','purpose_other', 'purpose_renewable_energy', 'purpose_small_business','purpose_vacation', 'purpose_wedding', 'term_ 36 months','term_ 60 months']
#logistic regression with predict proba
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(train[cols], train["loan_status"])

probabilities = model.predict_proba(test[cols])[:,1]

test["predicted_label"]=probabilities

#set cut to determine success and failure
n=test.shape[0]
probo=[]
for i in range(0,n):
    if test["predicted_label"].iloc[i]>0.9:
        probo.append(1)
    else:
        probo.append(0)
test["final_pred"]=probo

#calculate predictive measures

true_positive_filter = (test["final_pred"] == 1) & (test["loan_status"] == 1)
true_positives = len(test[true_positive_filter])
false_negative_filter = (test["final_pred"] == 0) & (test["loan_status"] == 1)
false_negatives = len(test[false_negative_filter])
true_negative_filter = (test["final_pred"] == 0) & (test["loan_status"] == 0)
true_negatives = len(test[true_negative_filter])
false_positive_filter = (test["final_pred"] == 1) & (test["loan_status"] == 0)
false_positives = len(test[false_positive_filter])
sensitivity = true_positives / (true_positives + false_negatives)
specificity= true_negatives / (false_positives + true_negatives) 

accuracy=(true_positives + true_negatives) / test.shape[0]


fpr = false_positives / (false_positives + true_negatives)
ppv=true_positives/ (true_positives + false_positives)
npv= true_negatives/ (true_negatives + false_negatives)

print(sensitivity)
print(fpr)
print( specificity)
print(accuracy)
print( ppv)
print( npv)

from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test[cols])
auc_score = roc_auc_score(test["loan_status"], probabilities[:,1])
print(auc_score)

#entropy

import numpy

def calc_entropy(column):
    
    counts = numpy.bincount(column)
    
    probabilities = counts / len(column)
    
    
    entropy = 0
    
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy

#calculate top information gainers
def calc_information_gain(data, split_name, target_name):
    
    original_entropy = calc_entropy(data[target_name])
    
    
    column = data[split_name]
    median = column.median()
    
    
    left_split = data[column <= median]
    right_split = data[column > median]
    
    
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    
    return original_entropy - to_subtract


information_gains=[]
columns=loans.columns
for col in columns:
    information_gain = calc_information_gain(loans, col, "loan_status")
    information_gains.append(information_gain)

columnas=pd.DataFrame(columns)
columnas["entropy"]=information_gains
sorted=columnas.sort_values(by="entropy",ascending=False)
top10=sorted.head(10)

#my own algorithm

lonso=['int_rate','int_rate','emp_length',"dti",  "purpose_wedding", "verification_status_Verified"]
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(train[lonso], train["loan_status"])

probabilities = model.predict_proba(test[lonso])[:,1]

test["predicted_label"]=probabilities

#set cut
n=test.shape[0]
probo=[]
for i in range(0,n):
    if test["predicted_label"].iloc[i]>0.9:
        probo.append(1)
    else:
        probo.append(0)
test["final_pred"]=probo

true_positive_filter = (test["final_pred"] == 1) & (test["loan_status"] == 1)
true_positives = len(test[true_positive_filter])
false_negative_filter = (test["final_pred"] == 0) & (test["loan_status"] == 1)
false_negatives = len(test[false_negative_filter])
true_negative_filter = (test["final_pred"] == 0) & (test["loan_status"] == 0)
true_negatives = len(test[true_negative_filter])
false_positive_filter = (test["final_pred"] == 1) & (test["loan_status"] == 0)
false_positives = len(test[false_positive_filter])
sensitivity = true_positives / (true_positives + false_negatives)
specificity= true_negatives / (false_positives + true_negatives) 

accuracy=(true_positives + true_negatives) / test.shape[0]


fpr = false_positives / (false_positives + true_negatives)
ppv=true_positives/ (true_positives + false_positives)
npv= true_negatives/ (true_negatives + false_negatives)

print(sensitivity)
print(fpr)
print( specificity)
print(accuracy)
print( ppv)
print( npv)

#roc auc
from sklearn.metrics import roc_auc_score
probabilos = model.predict_proba(test[lonso])
auc_score = roc_auc_score(test["loan_status"], probabilos[:,1])
print(auc_score)




