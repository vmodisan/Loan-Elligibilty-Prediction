import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/Users/vmodi/OneDrive/Documents/MSc Data Science and Artificial Intelligence/INDIVIDUAL CAPSTONE PROJECT/Loan_Data.csv')
print(df)

#check for null values in each column
df.isnull().sum
print(df.isnull().sum())

df.isnull().count()
print(df.isnull().count())

#Check for Duplication
df.nunique()
print(df.nunique())

#Handle numerical missing values
#import the statistics module
import statistics
applicant_income_mode = df['Applicant Income'].mode().iat[0]
print(f"The mode of 'applicant income' is: {applicant_income_mode}")

# Fill null values with the mode
df['Applicant Income'] = df['Applicant Income'].fillna(df['Applicant Income'].mode()[0])

#check if null values have been replaced
count_non_null = df['Applicant Income'].notnull().sum()
print(f"Number of non-null values in 'Applicant Income': {count_non_null}")
print()

#check the data types of df
df.dtypes
print()

#inspect the top  five rows
df.head()
print(df.head())

#Inspect the bottom five rows
df.tail()
print(df.tail())

#identify the structure and completeness of the data
df.info()
print(df.info())


#Read Train and Test Dataset
train=pd.read_csv("/Users/vmodi/OneDrive/Documents/MSc Data Science and Artificial Intelligence/INDIVIDUAL CAPSTONE PROJECT/Loan_Data.csv")

train["Loan_Status"].count()
train['Loan_Status'].value_counts()
print(train['Loan_Status'].value_counts())
train["Loan_Status"].value_counts(normalize=True).plot.bar(title = 'Loan_Status')
plt.show()

train["Married"].count()
train["Married"].value_counts()
print(train["Married"].value_counts())
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
plt.show()

train["Gender"].count()
train["Gender"].value_counts()
print(train["Gender"].count())
train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')
plt.show()

train["Self_Employed"].count()
train["Self_Employed"].value_counts()
print(train["Self_Employed"].value_counts())
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')
plt.show()

train["Credit_History"].count()
train["Credit_History"].value_counts()
print(train["Credit_History"].count())
print(train["Credit_History"].value_counts())
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')
plt.show()

train["Dependents"].count()
train["Dependents"].value_counts()
print(train["Dependents"].count())
print(train["Dependents"].value_counts())
train['Dependents'].value_counts(normalize=True).plot.bar(title= 'Dependents')
plt.show()

train["Education"].count()
train["Education"].value_counts()
print(train["Education"].count())
print(train["Education"].value_counts())
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')
plt.show()

train["Residential_Area"].count()
train["Residential_Area"].value_counts()
print(train["Residential_Area"].count())
print(train["Residential_Area"].value_counts())
train['Residential_Area'].value_counts(normalize=True).plot.bar(title= 'Residential_Area')
plt.show()


#Applicant Income Distribution
import seaborn as sns
plt.figure(1)
plt.subplot(121)
sns.distplot(train["Applicant Income"]);

plt.subplot(122)
train["Applicant Income"].plot.box(figsize=(16,5))
plt.show()

#Applicant income measured by Education
train.boxplot(column='Applicant Income',by="Education" )
plt.suptitle(" ")
plt.show()

#Coapplicant Income distribution
plt.figure(1)
plt.subplot(121)
sns.distplot(train["Coapplicant Income"]);

plt.subplot(122)
train["Coapplicant Income"].plot.box(figsize=(16,5))
plt.show()

#distribution on loan amount variable
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['Loan Amount']);

plt.subplot(122)
train['Loan Amount'].plot.box(figsize=(16,5))
plt.show()

#Number of people who take loan as group by loan amount
# Set a custom color palette (optional)
custom_palette = ['#432371', '#FAAE7B']  

# Create the countplot
plt.figure(figsize=(8, 6))
sns.countplot(x='Loan Amount', data=df, palette=custom_palette)

# Customize the plot (add labels, title, etc.)
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.title('Distribution of Loan Amounts')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# Show the plot
plt.tight_layout()
plt.show()

#Loan amount term analysis
df = train.dropna()
# Set a custom color palette (yellow)
custom_palette = ["#FFFF00"]  # Yellow color
# Create a figure with two subplots
plt.figure(figsize=(16, 5))
# First subplot: Distribution plot (histogram and KDE)
plt.subplot(121)
sns.histplot(df["Loan_Amount_Term"], kde=True, color=custom_palette[0])
plt.title("Loan Amount Term Distribution")
plt.xlabel("Loan Amount Term")
plt.ylabel("Density")

# Second subplot: Box plot
plt.subplot(122)
sns.boxplot(x=df["Loan_Amount_Term"], color=custom_palette[0])
plt.title("Loan Amount Term Box Plot")
plt.xlabel("Loan Amount Term")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


#Relation between "Loan_Status" and "Married"
Married = pd.crosstab(train["Married"], train["Loan_Status"])

# Calculate percentage within each marital status
Married_percentage = Married.div(Married.sum(1).astype(float), axis=0)

# Create a grouped bar chart
plt.figure(figsize=(6, 4))
Married_percentage.plot(kind="bar", width=0.6)
plt.xlabel("Marital Status")
plt.ylabel("Percentage")
plt.title("Loan Approval by Marital Status")
plt.xticks(rotation=0)  # Rotate x-axis labels for readability
plt.legend(title="Loan Status", loc="upper right")
plt.show()


# Filter married applicants
married_applicants = train[train["Married"] == "Yes"]

# Calculate the number of married applicants with approved and disapproved loans
m_approved = married_applicants[train["Loan_Status"] == "Y"].shape[0]
m_disapproved = married_applicants[train["Loan_Status"] == "N"].shape[0]

# Filter unmarried applicants
unmarried_applicants = train[train["Married"] == "No"]

# Calculate the number of unmarried applicants with approved and disapproved loans
u_approved = unmarried_applicants[train["Loan_Status"] == "Y"].shape[0]
u_disapproved = unmarried_applicants[train["Loan_Status"] == "N"].shape[0]

# Print the results
print(f"Married Applicants - Approved: {m_approved}, Disapproved: {m_disapproved}")
print(f"Unmarried Applicants - Approved: {u_approved}, Disapproved: {u_disapproved}")

#relationship between loan status and gender
print(pd.crosstab(train["Gender"],train["Loan_Status"]))
Gender = pd.crosstab(train["Gender"],train["Loan_Status"])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()

# Calculate the number of male or female applicants with approved and disapproved loans
# Filter male applicants
male_applicants = train[train["Gender"] == "Male"]

# Calculate the number of male applicants with approved and disapproved loans
m_approved = male_applicants[train["Loan_Status"] == "Y"].shape[0]
m_disapproved = male_applicants[train["Loan_Status"] == "N"].shape[0]

# Filter female applicants
female_applicants = train[train["Gender"] == "Female"]

# Calculate the number of female applicants with approved and disapproved loans
f_approved = female_applicants[train["Loan_Status"] == "Y"].shape[0]
f_disapproved = female_applicants[train["Loan_Status"] == "N"].shape[0]

# Print the results
print(f"Male Applicants - Approved: {m_approved}, Disapproved: {m_disapproved}")
print(f"Female Applicants - Approved: {f_approved}, Disapproved: {f_disapproved}")

#Relationship between loan status and Dependents
print(pd.crosstab(train['Dependents'],train["Loan_Status"]))
Dependents = pd.crosstab(train['Dependents'],train["Loan_Status"])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.show()

#Relationship between education and loan status
Education = pd.crosstab(train["Education"], train["Loan_Status"])

# Calculate the percentage within each education category
Education_percentage = Education.div(Education.sum(1).astype(float), axis=0)

# Create a stacked bar chart
plt.figure(figsize=(4, 4))
Education_percentage.plot(kind="bar", stacked=True)
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.title("Loan Approval by Education")
plt.legend(title="Loan Status", loc="upper right")
plt.show()

#compare the numbers
# Filter data for graduates
graduates = train[train["Education"] == "Graduate"]

# Calculate the number of approved and disapproved loans for graduates
grad_approved = graduates[train["Loan_Status"] == "Y"].shape[0]
grad_disapproved = graduates[train["Loan_Status"] == "N"].shape[0]

# Filter data for non-graduates
non_graduates = train[train["Education"] == "Not Graduate"]

# Calculate the number of approved and disapproved loans for non-graduates
non_grad_approved = non_graduates[train["Loan_Status"] == "Y"].shape[0]
non_grad_disapproved = non_graduates[train["Loan_Status"] == "N"].shape[0]

# Print the results
print(f"Graduates - Approved: {grad_approved}, Disapproved: {grad_disapproved}")
print(f"Non-graduates - Approved: {non_grad_approved}, Disapproved: {non_grad_disapproved}")

#relationship between employment and loan status
print(pd.crosstab(train["Self_Employed"],train["Loan_Status"]))
Employed = pd.crosstab(train["Self_Employed"],train["Loan_Status"])
Employed.div(Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Self_Employed")
plt.ylabel("Percentage")
plt.show()

# Filling null values in Loan amount term with the mode

# Display the value counts before filling null values
print("Value counts before filling null values:")
print(df["Loan_Amount_Term"].value_counts())

# Fill null values with the mode (most frequent value)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

# Display the value counts after filling null values
print("\nValue counts after filling null values:")
print(df["Loan_Amount_Term"].value_counts())

train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
train["Married"].fillna(train["Married"].mode()[0],inplace=True)
train['Dependents'].fillna(train["Dependents"].mode()[0],inplace=True)
train["Self_Employed"].fillna(train["Self_Employed"].mode()[0],inplace=True)
train["Credit_History"].fillna(train["Credit_History"].mode()[0],inplace=True)

#Dividing data into train and test_df
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)\

import pandas as pd
from sklearn.model_selection import train_test_split


# Create new features for train df: Total Income, EMI, and Balance Income
train_df["TotalIncome"] = train_df["Applicant Income"] + train_df["Coapplicant Income"]
train_df["EMI"] = train_df["Loan Amount"] / train_df["Loan_Amount_Term"]
train_df["BalanceIncome"] = train_df["TotalIncome"] - (train_df["EMI"] * 12) 

# Print the first few rows to verify the new features
print(train_df[["TotalIncome", "EMI", "BalanceIncome"]].head())
print(train_df[["TotalIncome"]].head())

# Create new features for test df: Total Income, EMI, and Balance Income
test_df["TotalIncome"] = test_df["Applicant Income"] + test_df["Coapplicant Income"]
test_df["EMI"] = test_df["Loan Amount"] / test_df["Loan_Amount_Term"]
test_df["BalanceIncome"] = test_df["TotalIncome"] - (test_df["EMI"] * 12)  

# Print the first few rows to verify the new features
print(test_df[["TotalIncome", "EMI", "BalanceIncome"]].head())
print(test_df[["TotalIncome"]].head())

#distribution of train dataset Total Income.
sns.distplot(train_df["TotalIncome"])
plt.show()

# Calculate the natural logarithm of TotalIncome
train_df["TotalIncome_log"] = np.log(train_df["TotalIncome"])

# Plot the distribution of TotalIncome_log
plt.figure(figsize=(8, 6))
plt.title("Distribution of TotalIncome_log")
plt.xlabel("TotalIncome_log")
plt.ylabel("Density")
sns.distplot(train_df["TotalIncome_log"], color="skyblue", hist_kws={"edgecolor": "black"})
plt.show()

#distribution of test dataset Total Income.
sns.distplot(test_df["TotalIncome"])
plt.show()

# Calculate the natural logarithm of TotalIncome
test_df["TotalIncome_log"] = np.log(test_df["TotalIncome"])
# Plot the distribution of TotalIncome_log
plt.figure(figsize=(8, 6))
plt.title("Distribution of TotalIncome_log")
plt.xlabel("TotalIncome_log")
plt.ylabel("Density")
sns.distplot(train_df["TotalIncome_log"], color="skyblue", hist_kws={"edgecolor": "black"})
plt.show()

#Create EMI feature
train_df["EMI"]=train_df["Loan Amount"]/train_df["Loan_Amount_Term"]
test_df["EMI"]=test_df["Loan Amount"]/test_df["Loan_Amount_Term"]

#Creation of Balance Income
train_df["BalanceIncome"] = train_df["TotalIncome"]-train_df["EMI"]*1000 # To make the units equal we multiply with 1000
test_df["BalanceIncome"] = test_df["TotalIncome"]-test_df["EMI"]

# Drop the specified columns from train_df
columns_to_drop = ["Applicant Income", "Coapplicant Income", "Loan Amount", "Loan_Amount_Term"]
train_df.drop(columns=columns_to_drop, inplace=True)

# Print the updated DataFrame to verify the dropped columns
print(train_df.head())

# Drop the specified columns from test_df
columns_to_drop = ["Applicant Income", "Coapplicant Income", "Loan Amount", "Loan_Amount_Term"]
test_df.drop(columns=columns_to_drop, inplace=True)

# Print the updated DataFrame to verify the dropped columns
print(test_df.head())

# Drop the "Loan_ID" column from train_df and test_df
train_df.drop(columns=["Loan_ID"], inplace=True)
test_df.drop(columns=["Loan_ID"], inplace=True)

# Print the first 5 rows of both DataFrames
print("Train DataFrame:")
print(train_df.head(5))

print("\nTest DataFrame:")
print(test_df.head(5))

# Drop the "Loan_Status" column from Train DataFrame
train_df.drop(columns=["Loan_Status"], inplace=True)
# Print the first 5 rows of train DataFrame to verify the dropped column
print("Train DataFrame:")
print(train_df.head(5))

print("\nTest DataFrame:")

# Create a new DataFrame with only the "Loan_Status" column
loan_status_df = pd.DataFrame({"Loan_Status": df["Loan_Status"]})

# Save the Loan_Status DataFrame to a CSV file
loan_status_df.to_csv("loan_status.csv", index=False)

# Print a success message
print("Loan_Status saved to loan_status.csv")

# List of categorical columns (adjust as needed)
categorical_columns = ["Gender", "Married", "Education", "Self_Employed", "Residential_Area"]

# Create dummy variables for train_df
train_df_dummies = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)

# Create dummy variables for test_df
test_df_dummies = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)

# Print the first few rows of the updated DataFrames
print("Train DataFrame with dummy variables:")
print(train_df_dummies.head())

print("\nTest DataFrame with dummy variables:")
print(test_df_dummies.head())

#import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define features (X) and target (y)
X_train = train_df_dummies.drop(columns=["Loan_Status"])  # Adjust column names as needed
y_train = train_df_dummies["Loan_Status"]  # Adjust column name for target variable

# Create and train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Print a success message
print("Logistic regression model trained successfully!")