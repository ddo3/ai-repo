import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import random

petal_length_all = []
petal_width_all = []


petal_length = []
petal_width = []
species_data = []

versicolor_length = []
versicolor_width = []

virginica_length = []
virginica_width = []

class_list = []

def sigmoid(z: float):
    return 1/(1 + np.exp(-1 * z))

def get_data_from_file() -> None: #DONE

    with open('irisdata.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #
            petal_width_all.append(float(row['petal_width']))
            petal_length_all.append(float(row['petal_length']))

            if(row['species'] != 'setosa'):
                petal_width.append(float(row['petal_width']))
                petal_length.append(float(row['petal_length']))
                species_data.append(row['species'])

        for i in range(0, len(species_data)):
            s = species_data[i]
            if(s == 'versicolor'):
                versicolor_length.append(petal_length[i])
                versicolor_width.append(petal_width[i])
                class_list.append(0)
            elif(s == 'virginica'):
                virginica_length.append(petal_length[i])
                virginica_width.append(petal_width[i])
                class_list.append(1)
################################################################################

def mean_square_error(x_data: list, y_data: list, weights: list, data_class: list) -> float:#DONE

    sum = 0

    for i in range(0, len(x_data)):
        x = x_data[i]
        y = x_data[i]

        z = np.dot(weights, [1, x, y])

        prediction = sigmoid(z)

        sum = sum + ( prediction - data_class[i] )**2

    return sum / len(x_data)

def gradient(start_index: int, end_index: int, epsilon: float, weight: list) -> List:

    sum_b = 0
    sum_x = 0
    sum_y = 0

    for i in range(start_index, end_index + 1):

        x = petal_length[i]
        y = petal_width[i]

        z = np.dot(weight, [1, x, y])

        prediction = sigmoid(z)

        term1 = 2 * (prediction - class_list[i])

        term2 = (prediction * (1 - prediction))

        sum_b = sum_b + (term1 * term2 * 1)

        sum_x = sum_x + (term1 * term2 * x)

        sum_y = sum_y + (term1 * term2 * y)

    n = (end_index - start_index + 1)



    b_change = (sum_b/n) * epsilon
    w0_change = (sum_x/n) * epsilon
    w1_change = (sum_y/n) * epsilon



    new_w = [weight[0] - b_change, weight[1] - w0_change , weight[2] - w1_change]

    return new_w

def plot(weight1: list, title: str) -> None:#, weight2: list):

    x = []
    y1 = []
    #y2 = []

    for x_data in petal_length:

        y_data1 = -1 *((weight1[1] * x_data) + weight1[0])/weight1[2]
        #y_data2 = -1 *((weight2[1] * x_data) + weight2[0])/weight2[2]

        y1.append(y_data1)
        #y2.append(y_data2)
        x.append(x_data)

    plt.figure(1)


    plt.plot(virginica_length, virginica_width,'o', versicolor_length, versicolor_width, 'x')
    plt.plot(x, y1, 'k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(title)
    plt.gca().legend(('virginica', 'versicolor', 'old weights1'))

    plt.show()

def mse_plot(x_list: list, y_list: list) -> None:
    plt.figure(1)
    plt.plot(x_list, y_list,'b')
    plt.xlabel('Number of Iterations')
    plt.ylabel('MSE Of Example')
    plt.show()

def p3partA(weights: list, epsilon: float, needs_middle_plot: bool ) -> None:

    plot(weights, "Before Learning")

    #take the inputs and epsilon
    old_weights = weights

    #get the mean squared error
    old_mse = mean_square_error(petal_length, petal_width, old_weights, class_list)

    #put them into the gradient decent function to get new weights
    new_weights = gradient(0, 99, epsilon, old_weights)

    #get the mean squared error
    new_mse = mean_square_error(petal_length, petal_width, new_weights, class_list)

    diff = np.abs(old_mse - new_mse)
    print(new_mse)
    count = 0

    x_list = []
    y_list = []

    #while the mean_square_errors are not within a certain bounds of each other
    while(diff > .000001):
    #   set new_weigths as old weights
        old_weights = new_weights

    #   use gradient decent fucntion to get new weights
        new_weights = gradient(0, 99, epsilon, old_weights)

    #   get the mean sqaure error for both
        old_mse = mean_square_error(petal_length, petal_width, old_weights, class_list)
        new_mse = mean_square_error(petal_length, petal_width, new_weights, class_list)

    #   get new diff value
        diff = np.abs(old_mse - new_mse)

        x_list.append(count)
        y_list.append(new_mse)

    #   increment a count
        count = count + 1

        if(count%100 == 0):
            print("Number of Iterations: " + str(count))
            #print("Old mse = " + str(old_mse))
            #print("New mse = " + str(new_mse))
            #print("Diff = " + str(diff))

        if count == 900 & needs_middle_plot:
            plot(new_weights, "Middle Of Learning")

    plot(new_weights, "After Learning")
    mse_plot(x_list, y_list)
    print("Converged!")

################################################################################

def p3partC(epsilon: float) -> None:
    #b(-.01, 10)
    #w0(-.01,5) .5
    #w1(.01, 1)

    #generate a random weight
    #int(str(number)[:2])
    b = float(str(random.uniform(-.01, 1))[:3])
    w0 = float(str(random.uniform(-.01, 1))[:3])
    w1 = float(str(random.uniform(.01, 1))[:3])

    w = [b,w0, w1]
    print("Random weight is " + str(w))
    #In your writeup, show the two output plots at the initial, middle, and ﬁnal locations of the decision boundary.
    p3partA(w, epsilon, True)

################################################################################
def main():
    get_data_from_file()

    cmd = sys.argv[1]
    w1 = [-2,.5, .9]

    if cmd == 'partA':
        print("Exercise 3: PartA")
        print("Uisng weights = " + str(w1))
        p3partA( w1, .1, False)

    elif cmd == 'partC':
        print("Exercise 3: PartC")
        p3partC(.1)

    else:
        print('That is an invalid command')

if __name__ == "__main__":
    main()
