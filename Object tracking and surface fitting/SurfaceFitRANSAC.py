"""
Nicholas Novak
RANSAC SECTION
"""
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


with open('pc1.csv', 'r') as lidar_source:
    csv_reader = csv.reader(lidar_source)

    list_of_csv = list(csv_reader)
    x = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
    y = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
    z = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]
with open('pc2.csv', 'r') as lidar_source:
    csv_reader = csv.reader(lidar_source)

    list_of_csv2 = list(csv_reader)
    x2 = [float(list_of_csv[i][0]) for i in range(len(list_of_csv))]
    y2 = [float(list_of_csv[i][1]) for i in range(len(list_of_csv))]
    z2 = [float(list_of_csv[i][2]) for i in range(len(list_of_csv))]
def calcRansac(xyz,x,y,z, thresh = 5, iterations = 1000):
    
    inliers = []
    equation=[]
    n_points = len(xyz)
    i = 1

    while i < iterations:
        idx_samples = random.sample(range(n_points), 3)
        pts = [xyz[i] for i in idx_samples]
        p0 = np.array(pts[0]).astype('float64')
        p1 = np.array(pts[1]).astype('float64')
        p2 = np.array(pts[2]).astype('float64')
        
        V1 = p1-p0
        V2 = p2-p0

        normal = np.cross(np.array(V1), np.array(V2))
        
        # print(normal / la.norm(normal))
    
        a,b,c = normal
        d = np.dot(normal, p0)
        # print(a.dtype,xyz[0][0].dtype)
        idx_candidates = []
        # Find point distances to plane
        distance = [(a * x1 + b * y1 + c * z1 - d) / np.sqrt(a**2 + b**2 + c**2) for x1,y1,z1 in zip(x,y,z) if x1 not in [p0[0],p1[0],p2[0]] and y1 not in [p0[1],p1[1],p2[1]] and z1 not in [p0[2],p1[2],p2[2]]]

        # Determine what points' distances are within threshold
        idx_candidates = np.where(np.abs(distance) <= thresh)[0]

        if len(idx_candidates) > len(inliers)+1:
            # Cache maximum each loop
            equation = [a,b,c,d]
            inliers = idx_candidates
            print(len(idx_candidates))
        
        i+=1
    return equation, inliers
print("size", len(list_of_csv))

p = 0.99
e = .3
s = 3
iter_count = math.log(1-p)/math.log(1-((1-e)**s)) # Calculate iterations needed from lecture
print("Itercount = ",iter_count)
# iter_count = 44550

eq, inliers = calcRansac(list_of_csv,x,y,z,1.85,iter_count)
eq2, inliers2 = calcRansac(list_of_csv2,x2,y2,z2,1.85,iter_count)

print("Inlier sub: ", len(x)/len(inliers))
real_inliers = [list_of_csv[i] for i in inliers]
real_inliers2 = [list_of_csv2[i] for i in inliers2]


xp = np.linspace(-10, 10, 100)
yp = np.linspace(-10, 10, 100)

xpl, ypl = np.meshgrid(xp, yp)
eq_plot = eq[0] * xpl + eq[1] * ypl + eq[2]

ax = plt.axes(projection='3d')

ax.plot_surface(xpl,ypl,eq_plot)
ax.scatter3D(x, y, z, c = 'green')
print(np.shape(inliers))
ax.view_init(0,30)
plt.show()

xp = np.linspace(-10, 10, 100)
yp = np.linspace(-10, 10, 100)

xpl, ypl = np.meshgrid(xp, yp)
eq_plot = eq2[0] * xpl + eq2[1] * ypl + eq2[2]

ax = plt.axes(projection='3d')

ax.plot_surface(xpl,ypl,eq_plot)
ax.scatter3D(x, y, z, c = 'green')
print(np.shape(inliers))
ax.view_init(0,30)
plt.show()
