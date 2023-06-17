# Problem 2:
import cv2
import matplotlib
import numpy as np
import math
import matplotlib
from sympy import N
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import csv
from numpy import linalg as la
#------------------------Part 1 Covariance and EigenValues----------------------------%

def findVariance(mean_val, series_vals):
    # Find variance values for covariance matrix
    S = 0

    sub_series = [i - mean_val for i in series_vals]
    sub_series_T = [(i - mean_val).T for i in series_vals]

    num = sum([sub_series[i] * sub_series_T[i] for i in range(len(sub_series))])
    den = len(series_vals) - 1
    cov = num/den
    return cov

def findCOV(series_data_1, series_data_2):
    # Find COV value for covariance matrix
    mean_1 = series_data_1[0]
    series_1 = series_data_1[1]

    mean_2 = series_data_2[0]
    series_2 = series_data_2[1]

    sub_series_1 = [i - mean_1 for i in series_1]
    sub_series_2 = [i - mean_2 for i in series_2]
    num = sum([sub_series_1[i] * sub_series_2[i] for i in range(len(sub_series_1))])
    den = len(series_1) - 1
    COV = num/den
    return COV

    
with open('pc1.csv', 'r') as lidar_source:
    # Find covariance and normal vector to pc1.csv
    csv_reader = csv.reader(lidar_source)

    list_of_csv = list(csv_reader)
    x_vals = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
    y_vals = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
    z_vals = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]


    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    z_mean = np.mean(z_vals)

    list_of_means = [(x_mean, x_vals), (y_mean, y_vals), (z_mean, z_vals)]
    for i in list_of_means:
       variance = findVariance(i[0], i[1])

    COV = np.array([[findVariance(x_mean, x_vals), findCOV(list_of_means[1],list_of_means[0]), findCOV(list_of_means[2], list_of_means[0])], \
        [findCOV(list_of_means[0], list_of_means[1]), findVariance(y_mean, y_vals), findCOV(list_of_means[2], list_of_means[1])], \
            [findCOV(list_of_means[0], list_of_means[2]), findCOV(list_of_means[1], list_of_means[2]), findVariance(z_mean, z_vals)]])

    data = COV



    eigVals, eigVects = la.eig(data)
    print("Covariance Matrix:\n",data)
    print("EigVals and Vects of pc1.csv: \n", eigVals, "\n\n", eigVects)

    val_min = eigVals[0]
    min_count = 0
    for i in range(len(eigVals)):
        if eigVals[i] < val_min:
            val_min = eigVals[i]
            min_count = i
    eigVect_min = eigVects[min_count]
    eigVal_min = eigVals[min_count] 

# SHow normal direction and magnitude
    print("Normal direction (x,y,z): ", eigVect_min, "\nNormal magnitude: ", eigVal_min)
#------------------------------------------------------------------------%
#-----------------------Part 2 Total Least Squares:----------------------%

with open('pc1.csv', 'r') as lidar_source:
    csv_reader = csv.reader(lidar_source)

    list_of_csv = list(csv_reader)
    x_vals = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
    y_vals = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
    z_vals = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]


# Finding TLS:
x_mean = np.mean(x_vals)
y_mean = np.mean(y_vals)
z_mean = np.mean(z_vals)

n = len(x_vals)

# Create U matrix as shown in lecture
U = np.vstack(((np.sum(x_vals)-x_mean),(np.sum(y_vals)-y_mean),(np.sum(z_vals)-z_mean))).T

#Find U.transpose * U
UTU = np.dot(U.transpose(),U)
# print("UTU is shape: ", UTU.shape)

# Get eigens
eigVals, eigVects = la.eig(UTU)
# print("EigVals ", eigVals)
# print("eigvects ", eigVects)
minVal = np.argmin(eigVals)
# print("Min: ",minVal)
minVect = eigVects[minVal]
# We now have normal vector and magnitude

ax = plt.axes(projection='3d')
ax.scatter3D(x_vals, y_vals, z_vals, c = z_vals, cmap = 'Greens')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        # Use minimum values and origin point (means) to find plane fit and plot
        Z[r,c] = ((minVect[0] * X[r,c] + minVect[1] * Y[r,c] + minVect[2])-minVect[0] * x_mean-minVect[1] * y_mean)/z_mean
ax.plot_wireframe(X,Y,Z, color='lightblue')
plt.show()

with open('pc2.csv', 'r') as lidar_source:
    # Perform fit for pc2.csv
    csv_reader = csv.reader(lidar_source)

    list_of_csv = list(csv_reader)
    x_vals = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
    y_vals = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
    z_vals = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]

# Finding TLS:
x_mean = np.mean(x_vals)
y_mean = np.mean(y_vals)
z_mean = np.mean(z_vals)

n = len(x_vals)

U = np.vstack(((np.sum(x_vals)-x_mean),(np.sum(y_vals)-y_mean),(np.sum(z_vals)-z_mean))).T

UTU = np.dot(U.transpose(),U)
print("UTU is shape: ", UTU.shape)
eigVals, eigVects = la.eig(UTU)
print("EigVals ", eigVals)
print("eigvects ", eigVects)
minVal = np.argmin(eigVals)
print("Min: ",minVal)
minVect = eigVects[minVal]

ax = plt.axes(projection='3d')
ax.scatter3D(x_vals, y_vals, z_vals, c = z_vals, cmap = 'Greens')


xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = ((minVect[0] * X[r,c] + minVect[1] * Y[r,c] + minVect[2])-minVect[0] * x_mean-minVect[1] * y_mean)/z_mean
ax.plot_wireframe(X,Y,Z, color='lightblue')
plt.show()
