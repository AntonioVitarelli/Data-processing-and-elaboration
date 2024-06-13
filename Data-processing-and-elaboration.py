import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from scipy.optimize import curve_fit
import datetime

# Read the file
calibration_board = pd.read_csv('calibrazione_board1.txt', sep='\t')
watermark_list = calibration_board['Watermark'].tolist()
resistance_list = calibration_board['Resistance'].tolist()
plt.plot(watermark_list, resistance_list, color='blue', label='Original data')

# Define the form of the hyperbola
def hyperbola(x, a, b):
    if np.any(x == 0):
        return np.nan 
    else:
        return a / (b * x)
# Define the form of the straight line
def straight_line(x, m, q):
    return m * x + q

# Perform the curve fitting
popt, pcov = curve_fit(hyperbola, watermark_list, resistance_list)

y_resistance_values = [hyperbola(w, *popt) for w in watermark_list]

# Print the optimal parameters a, b
print(f"The equation of the regression curve is: y = {popt[0]} / ({popt[1]} * x)")

# Plot the original data and the fitted curve
plt.plot(watermark_list, y_resistance_values, color='red', label='Fitted curve')
plt.xlabel('Watermark')
plt.ylabel('Resistance')
plt.title('Watermark vs Resistance')
plt.legend()
plt.grid(True)
plt.show()

# Read the CSV file
Datasheet = pd.read_csv('Cal_Watermark_Datasheet.csv')
Matric_potential_list_datasheet = Datasheet['Matric potential'].tolist()
Matric_potential_list_datasheet = [-Mat_pot for Mat_pot in Matric_potential_list_datasheet]
resistance_list_datasheet = Datasheet['Resistance'].tolist()

#Perform linear regression on Datasheet values 
slope_calibration_datasheet, intercept_calibration_datasheet, r_value, p_value, std_err = linregress(resistance_list_datasheet, Matric_potential_list_datasheet)
y_Matric_potential_datasheet = [straight_line(r, slope_calibration_datasheet, intercept_calibration_datasheet) for r in resistance_list_datasheet]

# Print the results
print(f"slope_calibration_datasheet: {slope_calibration_datasheet}")
print(f"intercept_calibration_datasheet: {intercept_calibration_datasheet}")

# Plot the original data points of datasheet
plt.scatter(resistance_list_datasheet, Matric_potential_list_datasheet, label='Datasheet data', s = 0.8)
plt.plot(resistance_list_datasheet, y_Matric_potential_datasheet, 'r', label='Fitted line')
plt.xlabel('Resistance')  # Sets the label for the x-axis
plt.ylabel('Matric_potential')  # Sets the label for the y-axis
plt.title('Resistance vs Matric Potential')  # Sets the title of the plot
plt.legend()  # Adds a legend to the plot
plt.grid(True)  # Adds a grid to the plot
plt.show()  # Displays the plot

# Read the CSV file into a DataFrame
df = pd.read_csv('experiment.csv')
# Order by date
df = df.sort_values('Date', ascending=True)
# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Define the start and end dates for the mask
start_date = pd.to_datetime('2024-03-01')
end_date = pd.to_datetime('2024-05-30')

# Create a mask that selects dates within the start and end dates
mask = (df['Date'] > start_date) & (df['Date'] <= end_date)

# Apply the mask to the DataFrame
df = df.loc[mask]

# Define the plant IDs to plot the matric potential 
plant_id = [ 2, 3, 4, 7, 8, 9, 5, 6, 10, 15]

# Loop over each plant ID
for id in plant_id:
    # Filter the DataFrame for the current plant ID
    df_filtered = df.loc[df['id'] == id]
    # Collect all 'watermark' values in a list
    watermark_values = df_filtered['Watermark'].tolist()
    # Transform all watermark values into resistances using the equation of the regression line
    resistance_values = [hyperbola(w, *popt) for w in watermark_values]
    # Transform all resistance values into Matric_potential using the equation of the regression line from the datasheet
    Matric_potential_values = [straight_line(r, slope_calibration_datasheet, intercept_calibration_datasheet) for r in resistance_values]
    # Filter the 'Date' array in the same way
    plt.plot(df_filtered['Date'], Matric_potential_values,  label='Plant ' + str(id))
plt.xlabel('Time')  # Sets the label for the x-axis
plt.ylabel('Matric potential values')  # Sets the label for the y-axis
plt.title('Matric potential values over time')  # Sets the title of the plot
plt.legend()  # Adds a legend to the plot
plt.grid(True)  # Adds a grid to the plot
plt.gcf().autofmt_xdate()  # Rotate and align the x labels
plt.show()  # Displays the plot
    
from scipy.stats import pearsonr
plant_id = [13, 14, 6, 8, 9, 10]
# Loop over each plant ID
for id in plant_id:
    # Filter the DataFrame for the current plant ID
    df_filtered = df.loc[df['id'] == id]
    #df_filtered = df_filtered.loc[df['Watermark'] != 0]
    # Collect all 'watermark' values in a list
    watermark_values = df_filtered['Watermark'].tolist()
    # Transform all watermark values into resistances using the equation of the regression line
    resistance_values = [hyperbola(w, *popt) for w in watermark_values]
    # Transform all resistance values into Matric_potential using the equation of the regression line from the datasheet
    Matric_potential_values = [straight_line(r, slope_calibration_datasheet, intercept_calibration_datasheet) for r in resistance_values]
    # Calculate the correlation
    # Assuming Matric_potential_values and df_filtered['Impedance'] are your two lists
    Matric_potential_values = np.array(Matric_potential_values)
    impedance_values = np.array(df_filtered['Impedance'])
    # Remove NaN and infinite values
    #Matric_potential_values = Matric_potential_values[np.isfinite(Matric_potential_values)]
    impedance_values = impedance_values[np.isfinite(impedance_values)]
    correlation, _ = pearsonr(df_filtered['Impedance'], Matric_potential_values)
    print(f"The correlation between impedence and matric potential of plant "+ str(id) + " is: " + str(correlation))
   
