# Energy-consumption-prediction
# Energy Consumption Prediction
# Python + Machine Learning

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------------
# Historical energy consumption (kWh)
# ----------------------------------

energy = np.array([
    12, 14, 13, 15, 17, 18, 20,
    19, 21, 23, 22, 24, 26, 25,
    27, 29, 28, 30, 32, 31
])

# Time index
time = np.arange(1, len(energy) + 1).reshape(-1, 1)

# ----------------------------------
# Train Machine Learning Model
# ----------------------------------

model = LinearRegression()
model.fit(time, energy)

# Predict next hour
next_hour = np.array([[len(energy) + 1]])

prediction = model.predict(next_hour)

print("--------------------------------")
print(" ENERGY CONSUMPTION PREDICTION")
print("--------------------------------")

print(f"Current Consumption : {energy[-1]:.2f} kWh")
print(f"Predicted Next Hour : {prediction[0]:.2f} kWh")

# ----------------------------------
# Predict next 5 hours
# ----------------------------------

future_time = np.arange(
    len(energy) + 1,
    len(energy) + 6
).reshape(-1, 1)

future_prediction = model.predict(future_time)

print("\nFuture Predictions:")

for i, value in enumerate(future_prediction, 1):
    print(f"Hour +{i}: {value:.2f} kWh")

# ----------------------------------
# Plot results
# ----------------------------------

plt.plot(
    time,
    energy,
    marker="o",
    label="Actual Consumption"
)

plt.plot(
    future_time,
    future_prediction,
    marker="x",
    linestyle="--",
    label="Predicted Consumption"
)

plt.xlabel("Hour")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Energy Consumption Prediction")
plt.legend()
plt.grid(True)

plt.show()
