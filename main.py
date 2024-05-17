# %%
#======Libraries======
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# %%
#======Function Definitions======
def dot_aligned(seq):
    """
    Aligns floating point values in a list to the decimal point.
    """
    snums = [str(n) for n in seq]
    dots = [s.find('.') for s in snums]
    m = max(dots)
    return [' '*(m - d) + s for s, d in zip(snums, dots)]

# %%
#======Initialization======
# Sets a random seed for NumPy and PyTorch
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Sets precision to 20 digits
torch.set_printoptions(precision = 20)
np.set_printoptions(precision = 20)

# Sets device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
#======Data Loading======
df = pd.read_csv("input/data_fahrenheit_celsius.csv") # Loads data
display(df.head(10))

# %%
# Visualizes the data
plt.figure(figsize=(6, 6), dpi=100)
sns.scatterplot(data=df, x="Fahrenheit", y="Celsius", s=15, color="#E67070")
plt.title("Temperature")
plt.show()

# %%
#======Data Cleaning======
# Defines the features and labels
features = df["Fahrenheit"].values.reshape(-1, 1)
labels = df["Celsius"].values.reshape(-1, 1)

# Splits the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Creates a separate dataframe for the testing set to graph later (actual)
test_df_actual = pd.DataFrame({"Fahrenheit": X_test.flatten(), "Celsius": y_test.flatten()})

# %%
#======Training======
# Hyperparameters
lr = 1e-4           # Learning rate
n_epochs = 30000    # Number of epochs

# Creates a model and sends it to the device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines a MSE loss function (ridge or L2 norm)
loss_fn = nn.MSELoss()

# Defines a stochastic gradient descent (SGD) optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr)

# Sets the model to training mode
model.train()

# Sends the training set to PyTorch
x_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)

# Loops through the number of epochs
for epoch in range(n_epochs):
    # Computes the predicted output
    yhat = model(x_train)

    # Computes the loss using MSE
    loss = loss_fn(yhat, y_train)
    loss.backward()

    # Punishes the model based off the training set
    optimizer.step()
    optimizer.zero_grad()

# Prints the weight and bias after training
states = dot_aligned(["{:.20f}".format(value.cpu().detach().numpy().item()) for value in model.state_dict().values()])
print(f"Weight:\t{states[0]}")  # Should be   0.5556
print(f"Bias:\t{states[1]}")    # Should be -17.7778

# %%
#======Testing======
# Sets the model to evaluation mode
model.eval()

# Sends the validation set to PyTorch
x_test = torch.from_numpy(X_test).float().to(device)

# Computes the predicted output
yhat = model(x_test)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_test).flatten()

# Creates a dataframe for the testing set to graph later (predicted)
test_df_predicted = pd.DataFrame({"Fahrenheit": test_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("Testing set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-1000, 1000, (100, 1))
y_new = (X_new - 32) / 1.8

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new.flatten(), "Celsius": y_new.flatten()})

# Sends the new set to PyTorch
x_new = torch.from_numpy(X_new).float().to(device)

# Computes the predicted output
yhat = model(x_new)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_new).flatten()

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("New set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Visualizes the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=test_df_actual, x="Fahrenheit", y="Celsius", s=25, color = "#E67070", ax=ax1)  # Testing points (actual)
sns.lineplot(data=test_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax1)          # Testing line (predicted)
ax1.set_title("Testing dataset (20 points)")
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax2)     # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax2)           # New line (predicted)
ax2.set_title("Newly generated dataset outside the range of training\n(100 points)")
plt.show()

# %%
#======Data Cleaning======
# Fits the data to the standard scaler
scaler = StandardScaler(with_mean=False)
scaler.fit(features)

# Defines the features and labels
features_scaled = scaler.transform(features)
labels_scaled = scaler.transform(labels)

# Splits the data into training and test sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(features_scaled, labels_scaled, test_size=0.2)

# Creates a separate dataframe for the testing set to graph later (actual)
test_df_actual = pd.DataFrame({"Fahrenheit": X_test_scaled.flatten(), "Celsius": y_test_scaled.flatten()})

# %%
#======Training======
# Sets the model to training mode
model.train()

# Sends the training set to PyTorch
X_train_scaled = torch.from_numpy(X_train_scaled).float().to(device)
y_train_scaled = torch.from_numpy(y_train_scaled).float().to(device)

# Loops through the number of epochs
for epoch in range(n_epochs):
    # Computes the predicted output
    yhat = model(X_train_scaled)

    # Computes the loss using MSE
    loss = loss_fn(yhat, y_train_scaled)
    loss.backward()

    # Punishes the model based off the training set
    optimizer.step()
    optimizer.zero_grad()

# Prints the weight and bias after training
states = dot_aligned(["{:.20f}".format(value.cpu().detach().numpy().item()) for value in model.state_dict().values()])
print(f"Weight:\t{states[0]}")  # Should be   0.5556
print(f"Bias:\t{states[1]}")    # Should be -17.7778

# %%
#======Testing======
# Sets the model to evaluation mode
model.eval()

# Sends the validation set to PyTorch
X_test_scaled = torch.from_numpy(X_test_scaled).float().to(device)

# Computes the predicted output
yhat = model(X_test_scaled)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_test_scaled).flatten()

# Creates a dataframe for the testing set to graph later (predicted)
test_df_predicted = pd.DataFrame({"Fahrenheit": test_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("Testing set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-1000, 1000, (100, 1))
y_new = (X_new - 32) / 1.8

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new.flatten(), "Celsius": y_new.flatten()})

# Sends the new set to PyTorch
X_new = torch.from_numpy(X_new).float().to(device)

# Computes the predicted output
yhat = model(X_new)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_new).flatten()

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("New set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Visualizes the data
fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=test_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax1)    # Testing points (actual)
sns.lineplot(data=test_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax1)          # Testing line (predicted)
ax1.set_title("Testing dataset (20 points)")
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax2)     # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax2)           # New line (predicted)
ax2.set_title("Newly generated dataset outside the range of training\n(100 points)")
plt.show()

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-1000, 1000, (100, 1))
y_new = (X_new - 32) / 1.8

# Scales the data to the standard scaler
X_new_scaled = scaler.transform(X_new)
y_new_scaled = scaler.transform(y_new)

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new_scaled.flatten(), "Celsius": y_new_scaled.flatten()})

# Sends the new set to PyTorch
X_new = torch.from_numpy(X_new_scaled).float().to(device)

# Computes the predicted output
yhat = model(X_new)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_new_scaled).flatten()

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Visualizes the data
plt.figure(figsize=(6, 6), dpi=100)
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070") # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF")       # New line (predicted)
plt.title("Newly generated and scaled dataset outside the range of training\n(100 points)")
plt.show()

# %%
#======Data Loading======
df_large = pd.read_csv("input/data_fahrenheit_celsius_large.csv") # Loads data
print(f"Size: {df_large.shape}")

# %%
# Visualizes the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=df_large, x="Fahrenheit", y="Celsius", s=15, color="#E67070", ax=ax1)
ax1.set_title("Temperature")
sns.histplot(df_large['Fahrenheit'], ax=ax2)
ax2.set_title("Large temperature dataset frequency")
plt.show()

# %%
#======Data Cleaning======
# Defines the features and labels
features = df_large["Fahrenheit"].values.reshape(-1, 1)
labels = df_large["Celsius"].values.reshape(-1, 1)

# Fits the data to the standard scaler
scaler.fit(features)

# Defines the features and labels
features_scaled = scaler.transform(features)
labels_scaled = scaler.transform(labels)

# Splits the data into training and test sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(features_scaled, labels_scaled, test_size = 0.2)

# Creates a separate dataframe for the testing set to graph later (actual)
test_df_actual = pd.DataFrame({"Fahrenheit": X_test_scaled.flatten(), "Celsius": y_test_scaled.flatten()})

#======Training======
# Sets the model to training mode
model.train()

# Sends the training set to PyTorch
X_train_scaled = torch.from_numpy(X_train_scaled).float().to(device)
y_train_scaled = torch.from_numpy(y_train_scaled).float().to(device)

# Loops through the number of epochs
for epoch in range(n_epochs):
    # Computes the predicted output
    yhat = model(X_train_scaled)

    # Computes the loss using MSE
    loss = loss_fn(yhat, y_train_scaled)
    loss.backward()

    # Punishes the model based off the training set
    optimizer.step()
    optimizer.zero_grad()
    
#======Testing======
# Sets the model to evaluation mode
model.eval()

# Sends the validation set to PyTorch
X_test_scaled = torch.from_numpy(X_test_scaled).float().to(device)

# Computes the predicted output
yhat = model(X_test_scaled)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_test_scaled).flatten()

# Creates a dataframe for the testing set to graph later (predicted)
test_df_predicted = pd.DataFrame({"Fahrenheit": test_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("Testing set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-10000, 10000, (1000, 1))
y_new = (X_new - 32) / 1.8

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new.flatten(), "Celsius": y_new.flatten()})

# Sends the new set to PyTorch
X_new = torch.from_numpy(X_new).float().to(device)

# Computes the predicted output
yhat = model(X_new)

# Gets the predicted output and compares it to the actual output
y_predicted = yhat.cpu().detach().numpy()
comparison = abs(y_predicted - y_new).flatten()

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("New set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Visualizes the data
fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=test_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax1)    # Testing points (actual)
sns.lineplot(data=test_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax1)          # Testing line (predicted)
ax1.set_title("Testing dataset (200 points)")
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax2)     # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax2)           # New line (predicted)
ax2.set_title("Newly generated dataset outside the range of training\n(1000 points)")
plt.show()

# %%
# Calculate metrics
mae = mean_absolute_error(new_df_actual['Celsius'], new_df_predicted['Celsius'])
mse = mean_squared_error(new_df_actual['Celsius'], new_df_predicted['Celsius'])
r2 = r2_score(new_df_actual['Celsius'], new_df_predicted['Celsius'])

# Prints metrics
print(f"Mean Absolute Error:\t{mae}")
print(f"Mean Squared Error:\t{mse}")
print(f"R-squared:\t\t{r2}")

# %%
#======Data Loading======
df = pd.read_csv("input/data_fahrenheit_celsius.csv") # Loads data

#======Data Cleaning======
# Defines the features and labels
features = df["Fahrenheit"].values.reshape(-1, 1)
labels = df["Celsius"].values.reshape(-1, 1)

# Splits the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Creates a separate dataframe for the testing set to graph later (actual)
test_df_actual = pd.DataFrame({"Fahrenheit": X_test.flatten(), "Celsius": y_test.flatten()})

#======Training======
model = LinearRegression()
model.fit(X_train, y_train)

#======Testing======
y_predicted = model.predict(X_train)
comparison = abs(y_predicted - y_train)

# Prints statistics
print("Training set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}\n")

y_predicted = model.predict(X_test)
comparison = abs(y_predicted - y_test)

# Creates a dataframe for the testing set to graph later (predicted)
test_df_predicted = pd.DataFrame({"Fahrenheit": test_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("Testing set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-10000, 10000, (20, 1))
y_new = (X_new - 32) / 1.8

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new.flatten(), "Celsius": y_new.flatten()})

# Computes the predicted output
y_predicted = model.predict(X_new)
comparison = abs(y_predicted - y_new)

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("New set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Visualizes the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=test_df_actual, x="Fahrenheit", y="Celsius", s=25, color = "#E67070", ax=ax1)  # Testing points (actual)
sns.lineplot(data=test_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax1)          # Testing line (predicted)
ax1.set_title("Testing dataset (20 points)")
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax2)     # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax2)           # New line (predicted)
ax2.set_title("Newly generated dataset outside the range of training\n(100 points)")
plt.show()

# %%
#======Data Cleaning======
# Fits the data to the standard scaler
scaler.fit(features)

# Defines the features and labels
features_scaled = scaler.transform(features)
labels_scaled = scaler.transform(labels)

# Splits the data into training and test sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(features_scaled, labels_scaled, test_size=0.2)

# Creates a separate dataframe for the testing set to graph later (actual)
test_df_actual = pd.DataFrame({"Fahrenheit": X_test_scaled.flatten(), "Celsius": y_test_scaled.flatten()})

#======Training======
model.fit(X_train_scaled, y_train_scaled)

#======Testing======
y_predicted = model.predict(X_train_scaled)
comparison = abs(y_predicted - y_train_scaled)

# Prints statistics
print("Training set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}\n")

y_predicted = model.predict(X_test_scaled)
comparison = abs(y_predicted - y_test_scaled)

# Creates a dataframe for the testing set to graph later (predicted)
test_df_predicted = pd.DataFrame({"Fahrenheit": test_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("Testing set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Evaluates numbers outside of the training set
X_new = np.random.uniform(-10000, 10000, (20, 1))
y_new = (X_new - 32) / 1.8

# Scales the data to the standard scaler
X_new_scaled = scaler.transform(X_new)
y_new_scaled = scaler.transform(y_new)

# Creates a dataframe for the new testing set to graph later (actual)
new_df_actual = pd.DataFrame({"Fahrenheit": X_new_scaled.flatten(), "Celsius": y_new_scaled.flatten()})

# Computes the predicted output
y_predicted = model.predict(X_new_scaled)
comparison = abs(y_predicted - y_new_scaled)

# Creates a dataframe for the new testing set to graph later (predicted)
new_df_predicted = pd.DataFrame({"Fahrenheit": new_df_actual["Fahrenheit"], "Celsius": y_predicted.flatten()})

# Prints statistics
print("New set statistics")
print(f"Median:\t{np.median(comparison)}")
print(f"Mean:\t{np.mean(comparison)}")
print(f"STD:\t{np.std(comparison)}")

# %%
# Visualizes the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
sns.scatterplot(data=test_df_actual, x="Fahrenheit", y="Celsius", s=25, color = "#E67070", ax=ax1)  # Testing points (actual)
sns.lineplot(data=test_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax1)          # Testing line (predicted)
ax1.set_title("Testing dataset (20 points)")
sns.scatterplot(data=new_df_actual, x="Fahrenheit", y="Celsius", s=25, color="#E67070", ax=ax2)     # New points (actual)
sns.lineplot(data=new_df_predicted, x="Fahrenheit", y="Celsius", color="#3075FF", ax=ax2)           # New line (predicted)
ax2.set_title("Newly generated dataset outside the range of training\n(100 points)")
plt.show()

# %%
# Calculate metrics
mae = mean_absolute_error(new_df_actual['Celsius'], new_df_predicted['Celsius'])
mse = mean_squared_error(new_df_actual['Celsius'], new_df_predicted['Celsius'])
r2 = r2_score(new_df_actual['Celsius'], new_df_predicted['Celsius'])

# Prints metrics
print(f"Mean Absolute Error:\t{mae}")
print(f"Mean Squared Error:\t{mse}")
print(f"R-squared:\t\t{r2}")