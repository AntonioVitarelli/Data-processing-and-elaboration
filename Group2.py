import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

calibration_board = pd.read_csv('calibrazione_board1.txt', sep='\t')
watermark_list = calibration_board['Watermark'].tolist()
resistance_list = calibration_board['Resistance'].tolist()

watermark_list = [float(i) for i in watermark_list]
resistance_list = [float(i) for i in resistance_list]

# Transform watermark_list values to 1/watermark_list[i]
transformed_watermark_list = [1/i for i in watermark_list]

# Compute linear regression
slope_calibration, intercept_calibration, r_value, p_value, std_err = linregress(transformed_watermark_list, resistance_list)
print("slope_calibration_calibration: ", slope_calibration)
print("intercept_calibration_calibration: ", intercept_calibration)

# Verify the linear regression
x_watermark_calibration = np.array(watermark_list)
y_resistance_calibration = slope_calibration * (1/x_watermark_calibration) + intercept_calibration

# Read the CSV file
Datasheet = pd.read_csv('Cal_Watermark_Datasheet.csv')

# Convert the 'pressure' and 'resistance' columns to lists
pressure_list_datasheet = Datasheet['Pressure'].tolist()
resistance_list_datasheet = Datasheet['Resistance'].tolist()


# Perform linear regression Datasheet
slope_calibration_datasheet, intercept_calibration_datasheet, r_value, p_value, std_err = linregress(resistance_list_datasheet, pressure_list_datasheet)

# Print the results
print(f"slope_calibration_datasheet: {slope_calibration_datasheet}")
print(f"intercept_calibration_datasheet: {intercept_calibration_datasheet}")

# Generate x values
x_resistance_datasheet = np.linspace(min(resistance_list_datasheet), max(resistance_list_datasheet), 100)

# Calculate the corresponding y values
y_pressure_datasheet = slope_calibration_datasheet * x_resistance_datasheet + intercept_calibration_datasheet

# Filter the DataFrame for the plant with id 13
#pianta = int(input("Enter plant id: "))
df = pd.read_csv('experiment.csv')
df_filtered = df.loc[df['id'] == 13]

# Collect all 'watermark' values in a list
watermark_values = df_filtered['Watermark'].tolist()

# Use the linear regression model to make predictions
resistance_experiment = []
watermark_experiment = []
for i in range(len(watermark_values)-1):
    watermark_experiment.append(1/float(watermark_values[i]))
for value in watermark_experiment:
    resistance_experiment.append(slope_calibration / value + intercept_calibration)
    
mat_pot = []
for r in resistance_experiment:
    mat_pot.append(slope_calibration_datasheet * r + intercept_calibration_datasheet)

# Create a list of timestamps (crono)
lenght = len(mat_pot)
crono = []
for i in range(lenght):
    crono.append(i)

# Plot the data points
plt.plot(crono, mat_pot, label='Plant 13')
plt.xlabel('Time')  # Sets the label for the x-axis
plt.ylabel('Watermark values')  # Sets the label for the y-axis
plt.title('Watermark values over time')  # Sets the title of the plot
plt.legend()  # Adds a legend to the plot
plt.grid(True)  # Adds a grid to the plot
plt.show()  # Displays the plot

# Plot the original data points od calibration board
plt.plot(watermark_list, resistance_list, label='Calibration function')
#Plot the linear regression line
plt.plot(x_watermark_calibration, y_resistance_calibration, 'r', label='Fitted function')
plt.xlabel('Watermark')  # Sets the label for the x-axis
plt.ylabel('Resistance')  # Sets the label for the y-axis
plt.title('Watermark vs Resistance')  # Sets the title of the plot
plt.legend()  # Adds a legend to the plot
plt.grid(True)  # Adds a grid to the plot
plt.show()  # Displays the plot

# Plot the original data points of datasheet
plt.plot(resistance_list_datasheet, pressure_list_datasheet, label='Datasheet function')
plt.plot(x_resistance_datasheet, y_pressure_datasheet, 'r', label='Fitted line')
plt.xlabel('Resistance')  # Sets the label for the x-axis
plt.ylabel('Pressure')  # Sets the label for the y-axis
plt.title('Resistance vs Pressure')  # Sets the title of the plot
plt.legend()  # Adds a legend to the plot
plt.grid(True)  # Adds a grid to the plot
plt.show()  # Displays the plot