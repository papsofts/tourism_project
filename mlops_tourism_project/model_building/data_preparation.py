# Libraries for data manipulation
import pandas as pd
import sklearn

# Library for creating a folder
import os

# Library for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Library for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/papsofts/tourism-project/tourism.csv"
tour_dataset = pd.read_csv(DATASET_PATH)
print("Tourism dataset loaded successfully.")

# Define the target variable
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features= [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]


# List of categorical features in the dataset
categorical_features= [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Define independant variables in X using selected numeric and categorical features
X = tour_dataset[numeric_features + categorical_features]

# Define target variable in y
y = tour_dataset[target]

# Split dataset into train and test with 70/30 split ratio
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Save Xtrain, Xtest, ytrain and ytest data into its respective csv files in the project directory
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# list the split data files
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# upload the split data files into hugging face space
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="papsofts/tourism-project",
        repo_type="dataset",
    )
