# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:23:40 2024

@author: Kevin.Nebiolo
"""

# define some functions for Poisson kriging
def kriging_weights(distances, nugget, sill, range_param):
    """
    Calculate variogram weights for Poisson kriging with rates.

    Parameters:
    - distances: Array of distances between data points and cell centers.
    - nugget: Nugget effect of the variogram model.
    - sill: Sill of the variogram model.
    - range_param: Range parameter of the variogram model.

    Returns:
    - weights: Array of variogram weights for each distance.
    """
    # Calculate the variogram values based on the distances and variogram model
    gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * (distances / range_param) ** 2))

    # Convert gamma to weights (e.g., using inverse squared distances)
    weights = 1 / gamma

    return weights


def poisson_kriging(x, y, rates, cell_centers, offset, nugget, sill, range_param):
    """
    Perform Poisson kriging interpolation with rates and offset in 2D.
    """
    # Calculate distances between cell centers and data points
    distances = cdist(np.vstack([x, y]).T, cell_centers)
    
    # Calculate variogram weights
    weights = kriging_weights(np.min(distances, axis=1), nugget, sill, range_param)
    
    # Calculate cell offsets
    cell_offsets = np.exp(offset)
    
    # Calculate interpolated values for all cell centers simultaneously
    interpolated_values = np.sum(weights * (rates * cell_offsets) / np.exp(distances / range_param), axis=1)
    
    return interpolated_values

import numpy as np

# Example data (assuming you have x, y coordinates and the variable of interest)
x = summary.ts.X
y = summary.ts.Y
# variable = np.array(...)

# Create a spatial weights matrix
w = weights.DistanceBand.from_array(np.vstack((x, y)).T, threshold=10)

# Compute the semivariance
semivariance = esda.Variogram(np.vstack((x, y)).T, None, w, normalize=False)
#TODO - should that None be the count per cell?  

# Plot the variogram
fig, ax = plt.subplots(figsize=(8, 6))
semivariance.plot(ax=ax, hist=True, bins=10)
plt.xlabel('Distance Lag')
plt.ylabel('Semivariance')
plt.title('Spatial Variogram')
plt.show()

# Fit variogram model
variogram_model = spreg.OLS(semivariance.bins, np.ones((len(semivariance.bins), 1))).fit()

# Extract nugget, sill, and range parameters
nugget = variogram_model.params[0]
sill = variogram_model.params[0] + variogram_model.params[1]
range_param = semivariance.maxlag

# Variogram Interpretation Guidance:
# - Nugget: Represents variance at very small spatial scales, often due to measurement error or sampling variability.
# - Sill: Total variance of the variable being analyzed. It's the asymptotic value of the variogram model.
# - Range Parameter: Distance beyond which spatial autocorrelation is negligible. It's where the variogram plateaus.
# - Use variogram shape and parameters to guide interpolation method selection and model parameterization.


# Step 2: Calculate Distances, get Closest Cell and Get Unique Counts Per Cell
distances = cdist(np.vstack([x, y]).T, summary.melted_center)
closest_cells = np.argmin(distances, axis=1)
unique_counts = np.bincount(closest_cells)

# Step 3: Calculate Variogram Weights
weights = kriging_weights(distances, nugget, sill, range_param)

# Step 4: Calculate Cell Offsets

# Extract relevant columns from agent_movement
agent_ids, cell_ids, time_spent = agent_movement[:, 0], agent_movement[:, 1], agent_movement[:, 2]

# Initialize array to store cumulative exposure per cell
cumulative_exposure = np.zeros(n_cells)

# Use numpy bincount to aggregate exposure per cell
np.add.at(cumulative_exposure, cell_ids, time_spent)
cell_offsets = np.exp(offset)

# Step 5: Perform Interpolation
interpolated_values = poisson_kriging(x, y, rates, cell_centers, offset, nugget, sill, range_param)

