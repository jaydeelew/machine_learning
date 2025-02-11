import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


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
# Filter out all other columns in the dataframe
X = filtered_melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)

# Compare MAE with differing values of max_leaf_nodes
maxNodes_Mae = {}  # node:mae
for max_leaf_nodes in [5, 50, 100, 250, 350, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    maxNodes_Mae[max_leaf_nodes] = my_mae
    print(
        "Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae)
    )
print(min(maxNodes_Mae, key=maxNodes_Mae.get))
