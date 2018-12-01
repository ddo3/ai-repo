import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
For exercsies 2

1) inputs for the program are the data in paired for (x,y), the weights, and the pattern class


'''

petal_length = []
petal_width = []
species_data = []

versicolor_length = []
versicolor_width = []

virginica_length = []
virginica_width = []

def get_data_from_file() -> None: #DONE

    with open('irisdata.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #
            petal_width.append(float(row['petal_width']))
            petal_length.append(float(row['petal_length']))
            species_data.append(row['species'])


    for i in range(0, len(species_data)):
        s = species_data[i]
        if(s == 'versicolor'):
            versicolor_length.append(petal_length[i])
            versicolor_width.append(petal_width[i])
        elif(s == 'virginica'):
            virginica_length.append(petal_length[i])
            virginica_width.append(petal_width[i])


def h_function(w_list: list, x_list: list) -> float:
    bottom =  1 + np.exp(-1 * dot_prod(w_list, x_list))
    return 1/bottom

def p1partB(datapoint: tuple) -> int:
    x = datapoint[0]
    y = datapoint[1]

    weights = [.12, -.36]

    value = h_function( weights, [x,y])

    if(value < 0.5):
        return 1
    else:
        return 0

################################################
#y = -0.5* x + 4
#TODO WHY does it need the pattern classes????
def p2partA(x_data: list, y_data: list, weights: list, data_class: int) -> float:

    y_prime = []

    for x in x_data:
        y_p = weights[1] * x  + weights[0]
        y_prime.append(y_p)

    sum = 0
    for i in range(0, len(y_data)):
        y = y_data[i]
        y_p = y_prime[i]
        sum = sum + (y - y_p)**2

    return sum / len(x_data)

################################################

def plot_with_line(weights: list) -> None:
    plt.figure(1)

    plt.plot(virginica_length, virginica_width,'o', versicolor_length, versicolor_width, 'x')

    x = []
    y = []

    for x_data in petal_length:

        y_data = weights[1] * x_data + weights[0]

        x.append(x_data)
        y.append(y_data)

    plt.plot(x, y, 'k')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor'))
    plt.show()

def p2partB() -> None:

    weights1 = [4,-.5]

    weights2 = []

    print("Mean Squared Error for weights: "  + str(weight1))
    print(p2partA(petal_length, petal_width, weight1, 1))
    print("Mean Squared Error for weights: " +  str(weight2))
    print(p2partA(petal_length, petal_width, weight2, 1))



    plot_with_line(weight1)
    plot_with_line(weight2)

################################################

#def p2partE() -> None:

################################################

def get_best_fit_line() -> list:

    best_b = 2     #[intercept, slope]
    best_m = -10

    we = [best_b, best_m]

    min = p2partA(petal_length, petal_width, we, 1)
    for i in np.arange(2, 5, .1):
        for j in np.arange(-10,0, .1):
             w = [i,j]

             test = p2partA(petal_length, petal_width, w, 1)

             if(test < min):
                 #print('found bester weights')
                 best_b = i
                 best_m = j
                 min = test

    print('Best weights')
    print([best_b, best_m])

    return [best_b, best_m]



get_data_from_file()
w = [3.5,-.6]

#Also a good fit [3,-.25]

print(p2partA(petal_length, petal_width, w, 1))
plot_with_line(w)
#new_w = get_best_fit_line()
#plot_with_line(new_w)
#The smaller the means squared error, the closer you are to finding the line of best fit
