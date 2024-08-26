import pandas as pd
import sklearn
import os

# Creating file paths for dataset and user input
data_path = os.path.abspath("miami-housing.csv")
user_path = os.path.abspath("miamiuser.csv")

house_data = pd.read_csv(data_path)
user_data = pd.read_csv(user_path)

y = house_data.SALE_PRC  # Target value named to be predicted

house_features = ['LATITUDE', 'LONGITUDE', 'LND_SQFOOT', 'age']  # Variables used to make prediction
X = house_data[house_features]

# Using DecisionTreeRegressor to fit X & y variables, and make predictions
from sklearn.tree import DecisionTreeRegressor

house_model = DecisionTreeRegressor(random_state=1)
house_model.fit(X, y)

print("Predictions: ")
raw_output = house_model.predict(user_data)

# Output results after formatting
i = 0
while i < len(raw_output):
    print("House " + str(i+1) + ": ${:0,.2f}".format(float(raw_output[i])))
    i += 1




