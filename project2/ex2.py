import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

petal_length = []
petal_width = []
species_data = []

versicolor_length = []
versicolor_width = []

virginica_length = []
virginica_width = []

class_list = []

def get_data_from_file() -> None: #DONE

    with open('irisdata.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #
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

################################################

def sigmoid(z: float):
    return 1/(1 + np.exp(-1 * z))

def p2partA(x_data: list, y_data: list, weights: list, data_class: list) -> float:#DONE

    sum = 0

    for i in range(0, len(x_data)):
        x = x_data[i]
        y = x_data[i]

        z = np.dot(weights, [1, x, y])

        prediction = sigmoid(z)

        sum = sum + ( prediction - data_class[i] )**2

    return sum / len(x_data)

################################################

def plot_for_partB(weight1: list, weight2: list) -> None:
    plt.figure(1)

    plt.plot(virginica_length, virginica_width,'o', versicolor_length, versicolor_width, 'x')

    x = []
    y1 = []
    y2 = []

    for x_data in petal_length:

        y_data1 = -1 *((weight1[1] * x_data) + weight1[0])/weight1[2]
        y_data2 = -1 *((weight2[1] * x_data) + weight2[0])/weight2[2]

        y1.append(y_data1)
        y2.append(y_data2)
        x.append(x_data)

    plt.plot(x, y1, 'k')
    plt.plot(x, y2, 'b')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor', 'weights1', 'weights2'))
    plt.show()

def p2partB() -> None: #DONE

    weight1 = [-4,.5, .9]

    weight2 = [-3,.5, .9]

    print("Mean Squared Error for weight1: " + str(weight1))
    print(p2partA(petal_length, petal_width, weight1, class_list))

    print("Mean Squared Error for weight2: " +  str(weight2))
    print(p2partA(petal_length, petal_width, weight2, class_list))


    plot_for_partB(weight1, weight2)

################################################

def plot_for_partE(weight1: list, weight2: list) -> None:

    x = []
    y1 = []
    y2 = []

    for x_data in petal_length:

        y_data1 = -1 *((weight1[1] * x_data) + weight1[0])/weight1[2]
        y_data2 = -1 *((weight2[1] * x_data) + weight2[0])/weight2[2]

        y1.append(y_data1)
        y2.append(y_data2)
        x.append(x_data)

    plt.figure(1)

    plt.subplot(1,2,1)
    plt.plot(virginica_length, virginica_width,'o', versicolor_length, versicolor_width, 'x')
    plt.plot(x, y1, 'k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor', 'old weights1'))


    plt.subplot(1,2,2)
    plt.plot(virginica_length, virginica_width,'o', versicolor_length, versicolor_width, 'x')
    plt.plot(x, y2, 'k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor', 'new weights'))

    plt.show()

def p2partE(start_index: int, end_index: int, epsilon: float, weight: list) -> List:

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

################################################
def main():
    get_data_from_file()

    cmd = sys.argv[1]
    w1 = [-4,.5, .9]

    if cmd == 'partA':
        print("Exercise 2: PartA")
        print("Uisng weights = " + str(w1))
        val = p2partA(petal_length, petal_width, w1, class_list)
        print("Mean Squared Error = " + str(val))

    elif cmd == 'partB':
        print("Exercise 2: PartB")
        p2partB()

    elif cmd == 'partE':
        w = [-3,.5, .9]

        print("Exercise 2: PartE")
        print("The range is datapoint 0 to datapoint 99")
        print("The stepsize is .1")
        print("The weights used are " + str(w))


        new_w = p2partE(0,99, .1, w)
        print("Updated weights are " + str(new_w))
        plot_for_partE(w, new_w)

    else:
        print('That is an invalid command')

if __name__ == "__main__":
    main()
