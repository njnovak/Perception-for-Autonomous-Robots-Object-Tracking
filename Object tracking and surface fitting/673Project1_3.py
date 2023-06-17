import random
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
def getPlaneHyperParams(name):
    tmpA = []
    tmpB = []
    with open(name, 'r') as lidar_source:
        csv_reader = csv.reader(lidar_source)

        list_of_csv = list(csv_reader)
        x = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
        y = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
        z = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]
    for i in range(len(x)):
        # Create A and B matrices
        tmpA.append([x[i],y[i],1])
        tmpB.append(z[i])
    
    b = np.matrix(tmpB).T
    A = np.matrix(tmpA)
    # FInd plance fit parameters using Ax = b equation
    fit = (A.T * A).I * A.T * b
    #Useful for confirmations
    errors = b-A * fit
    residual = la.norm(errors)

    print("Soln")
    print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    # print("errors (E):")
    # print(errors)
    # print("residual (R):")
    # print(residual)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x,y,z,color='green')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            # Create the fit using plane equation z = ax + by + c
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='lightblue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # PERFORM FOR PC1 AND PC2
getPlaneHyperParams('pc1.csv')
getPlaneHyperParams('pc2.csv')
