import pandas as pd 
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    df = pd.get_dummies(df, drop_first=True)

    x = df.drop("charges", axis=1)
    y = df["charges"]
    return train_test_split(x,y, test_size=0.2, random_state=42)