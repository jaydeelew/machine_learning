import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Read the data into a dataframe
melbourne_data = pd.read_csv("melb_data.csv")
# Filter out the rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target data
y = filtered_melbourne_data.Price
# Choose features (columns) from the dataframe
melbourne_features = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
]
# Filter out all other columns besides melbourne_features in the dataframe
X = filtered_melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)

# Create the model object
forest_model = RandomForestRegressor(random_state=1)
# Fit/Train the model with the training data
forest_model.fit(train_X, train_y)
# Using the validation date of the filtered dataframe, record the predictions
melbourne_predictions = forest_model.predict(val_X)
# See what the mae is when the predictions based on the set-apart validation data frame
# is compared to the set-apart target column
print(mean_absolute_error(val_y, melbourne_predictions))
