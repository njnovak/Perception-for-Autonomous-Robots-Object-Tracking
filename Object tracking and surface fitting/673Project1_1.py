import cv2
import matplotlib
import numpy as np
import math
import matplotlib
from sympy import N
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def findBallCoords(frame):

    # Find the coordinates of the center of the ball at a given frame
    blurred_mask = cv2.GaussianBlur(frame,(9,9),3,3)

    hsv_mask = cv2.cvtColor(blurred_mask, cv2.COLOR_BGR2HSV)

    bright_red_lower_bounds = (0, 100, 100)
    bright_red_upper_bounds = (10, 255, 255)
    bright_red_mask = cv2.inRange(hsv_mask, bright_red_lower_bounds, bright_red_upper_bounds)

    dark_red_lower_bounds = (160, 100, 100)
    dark_red_upper_bounds = (179, 255, 255)
    dark_red_mask = cv2.inRange(hsv_mask, dark_red_lower_bounds, dark_red_upper_bounds)


    weighted_mask = cv2.addWeighted(bright_red_mask, 1.0, dark_red_mask, 1.0, 0.0)

    
    # blurred_mask2 = cv2.GaussianBlur(weighted_mask, (9,9),3,3)
   
    img = np.copy(frame)
    # weighted_mask[0:80,:] = [0,0,0]

    resulting_frame = cv2.bitwise_and(img, img, mask = weighted_mask)
    resulting_frame[:,0:110] = [0,0,0]


    fx,fy,chn = resulting_frame.shape

    resulting_frame = cv2.resize(resulting_frame, (int(fx/5), int(fy/5)))

    fx,fy,chn = resulting_frame.shape

    num_pix = 0
    a = np.empty((0,2),int)
    b = np.copy(a)
    for col in range(fx):
        for row in range(fy):
            if resulting_frame[col,row,0] != 0:
                num_pix += 1
                if num_pix == 1:
                    col = fx - col
                    a = np.append(a,np.array([[col,row]]), axis=0)
                    # print(a)

    num_pix_rev = 0
    for col in reversed(range(fx)):
        for row in reversed(range(fy)):
            if resulting_frame[col,row,0]!=0:
                num_pix_rev += 1
                if num_pix_rev == 1:
                    col=fx-col
                    b=np.append(b,np.array([[col,row]]),axis=0)

    # print(a,b)
    col+=1
    cv2.waitKey(1)
    
    return a,b

def standardLeastSquares(vert_stack):

    # Find the standard least squares of a stack of points
    x_axis = vert_stack[:,0]
    y_axis = vert_stack[:,1]

    x_square = np.power(x_axis, 2)

    A = np.stack((x_square, x_axis, np.ones((len(x_axis)), dtype=int)), axis=1)

    A_trans = A.transpose()
    A_dot_A = A_trans.dot(A)
    A_dot_Y = A_trans.dot(y_axis)

    least_sqr_estimate = (np.linalg.inv(A_dot_A)).dot(A_dot_Y)
    B = least_sqr_estimate

    least_sqr_val = A.dot(least_sqr_estimate)

    return least_sqr_val

video_source = cv2.VideoCapture('ball.mov')

if (video_source.isOpened() == False):
    print("Error openeing video")

min_BGR1 = np.array([0,20,90])
max_BGR1 = np.array([73,83,150])

# Define mask bounds
min_HSV1 = np.array([0,120,70])
max_HSV1 = np.array([10,255,255])

min_HSV2 = np.array([170,120,70])
max_HSV2 = np.array([180,255,255])


reading, frame = video_source.read()
x_vals = []
y_vals = []
while reading:
    
    # Run through each frame of the video
    a,b = findBallCoords(frame=frame)
    for col in range(len(a)):
        x_vals = np.append(x_vals, ((a[col][1] + b[col][1]) / 2))

    for row in range(len(a)):
        y_vals = np.append(y_vals, ((a[row][0] + b[row][0]) / 2))
    reading, frame = video_source.read()

# U = np.array([x])

video_source.release()

cv2.destroyAllWindows()
stacked_vals = np.vstack((x_vals,y_vals)).T
y_vals_ls = standardLeastSquares(stacked_vals)

y_max = 0
x_max = 0

# Plot the results of the tracker and leeast squares curve

for i in range(len(y_vals_ls)):
    if y_vals_ls[i] > y_max:
        y_max = y_vals_ls[i]
        x_max = x_vals[i]
fig = plt.figure()
plt.title('Nicholas Novak Project 1 Question 1')
plt.subplot(111)
plt.xlabel('x-axis (pixels)')
plt.ylabel('y-axis (pixels)')
plt.scatter(x_vals*5, y_vals*5 - y_max * 5, c='red', label='plotted trajectory')
plt.plot(x_vals*5, y_vals_ls*5 - y_max * 5, c='green', label='least squares plot')
plt.title("Results of Video Plot")
print("Equation: ", eq)


# Find the equation of the parabola and plot it too to confirm it matches
def calc_parabola_equation(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

    return A,B,C

a,b,c = calc_parabola_equation(x_vals[0], y_vals_ls[0], x_max, y_max, x_vals[-1],y_vals[-1])

# Show equation of parabola
print("Equation of parabola: \n y=",a,"x^2 + ",b,"x + ",c)
y_pos = []
for i in range(len(x_vals)):
    x_v = x_vals[i]
    y_pos.append(((a*x_v**2)+(b*x_v)+c)*5 - y_max * 5)
plt.plot(x_vals*5, y_pos, c='blue', label='least squares equation')
plt.legend()
plt.show()

y_fin = y_vals[0] - 300

# Find final x position at the y postiion of the ground using the vertex and parabola equations
print("x-value at y = ",abs(y_fin*5)," pixels (ground): ")
x_fin = math.sqrt((y_fin/a) - y_max) + x_max
print(x_fin * 5)