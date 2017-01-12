# loanpredictors
Finding predictors for Lending Club loans from 2007/11 

Lending Club is a lending company in the US for short-term personal and business loans, the largest online credit marketplace, as per their description. I've analysed their online data for conceded loans in 2007-2011 and 2015, trying to find the best predictors for paid loans, in the search for the best algorithm. 

Full article: http://www.dartycs.com/2017/01/12/loan-algortihm-usa/




Methodology

During this experiment I'll consider 6 different ratio: Sensitivity or True Positive Rate (TPR), False Positive Rate (FPR), Accuracy, Specificity, Positive Predictive Value (PPV) and Negative Predictive Value (NPV). To add up, I'll use the ROC score to value the quality of the model.

Data:

My predictive target variable is the loan status: Fully Paid (1) or Charged Off (0).

The variables I'll use are a selection of those offered in the data : loan amount, interest rate, installment, employment length, DTI (the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income), delinq_2 years( the number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years), open_acc (the number of open credit lines in the borrower's credit file), home ownership (mortaage, own, rent or other), verification status, purpose of the credit (car, credit card, debt consolidation, educational, home improvement, house, major purchase, medical, moving, small business, renewable energy, vacation, wedding, other),  term (number of monthly payments on the loan, either 36 or 60).
