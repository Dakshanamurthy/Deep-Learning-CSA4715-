import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])


mean_x = np.mean(x)
mean_y = np.mean(y)

cross_deviation = np.sum((x - mean_x) * (y - mean_y))
deviation_x = np.sum((x - mean_x)**2)


slope = cross_deviation / deviation_x
intercept = mean_y - slope * mean_x


def plot_regression_line(x, y, slope, intercept):
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, slope * x + intercept, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


print(f"Slope: {slope}, Intercept: {intercept}")


plot_regression_line(x, y, slope, intercept)
