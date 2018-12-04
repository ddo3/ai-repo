import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
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
###############################################

def p1partA() -> None: #DONE

    plt.figure(1)
    plt.plot(versicolor_length, versicolor_width,'o', virginica_length, virginica_width, 'x')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('versicolor', 'virginica'))
    plt.show()

##############################################
def not_important() -> None:
    '''
    def sum_list(data: list) -> float:
        sum = 0
        for d in data:
            sum = sum + d

        return sum

    def mult_and_sum_lists(l1: list, l2: list) -> float:
        sum = 0

        for i in range(0, len(l1)):
            sum = ( l1[i] * l2[i] ) + sum

        return sum

    def get_w1(x_list: list, y_list: list) -> float:
        N = len(x_list)

        sum_x = sum_list(x_list)
        sum_y = sum_list(y_list)
        sum_x_and_y = mult_and_sum_lists(x_list, y_list)
        sum_x_squared = mult_and_sum_lists(x_list, x_list)

        top = (N * sum_x_and_y) - (sum_x * sum_y)

        bottom = (N * sum_x_squared ) - sum_x**2

        return top / bottom

    def get_w0(x_list: list, y_list: list, w1: float) -> float:
        N = len(x_list)

        sum_x = sum_list(x_list)
        sum_y = sum_list(y_list)

        return (sum_y - (w1 * sum_x))/N

    def get_linear_regression_list(x_list: list, y_list: list) -> list:
        datapoints = []

        w1 = get_w1(x_list, y_list)
        w0 = get_w0(x_list, y_list, w1)

        for x in x_list:
            y = (w1 * x) + w0
            datapoints.append( (x,y) )

        return datapoints

    def my_dumb_ass() -> None:
        datapoints = get_linear_regression_list(petal_length, petal_width)
        #print(datapoints)
        plt.figure(1)

        plt.plot(petal_length, petal_width,'o')

        x_list = []
        y_list = []

        for point in datapoints:
            x_list.append(point[0])
            y_list.append(point[1])

        plt.plot(x_list, y_list, 'k')

        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')

        plt.show()
    '''
#####################################################

def sigmoid(z: float):
    return 1/(1 + np.exp(-1 * z))

def h_function(w_list: list, x_list: list) -> float:
    z = np.dot(w_list, x_list)
    return sigmoid(z)

def test_partb() -> None:
    for i in range(0, len(virginica_length)):
        x = versicolor_length[i]
        y = versicolor_width[i]

        print( p1partB( (x,y) ))

def p1partB(datapoint: tuple) -> int: #DONE
    x = datapoint[0]
    y = datapoint[1]

    #This need to be 3 things [b, wo, w1]
    weights = [-4,.5, .9]

    value = h_function( weights, [1,x,y]) # this needs to be [1,x,y]

    if(value >= 0.5):
        return 1
    else:
        return 0

###############################################

def p1partC() -> None: #DONE

    plt.figure(1)

    weights = [-4,.5, .9] # bias, w0, w1
    #weights = [-3,.5, .9]

    plt.plot(versicolor_length, versicolor_width,'o', virginica_length, virginica_width, 'x')

    x = []
    y = []

    for i in range(0, len(petal_length)):
        x_d = petal_length[i]
        y_d = petal_width[i]

        #y_data = -0.5* x_d + 4
        y_data = -1 *((weights[1] * x_d) + weights[0])/weights[2]


        x.append(x_d)
        y.append(y_data)

    plt.plot(x, y, 'k')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('versicolor', 'virginica'))
    plt.show()

##################################################

def p1partD() -> None: #DONE

    x_data, y_data = np.meshgrid(petal_length, petal_width)

    weights = [-4,.5, .9]

    z_list = np.empty((100,100))

    for i in range(0,len(petal_length)):
        for j in range(0,len(petal_width)):

            x = x_data[i][j]
            y = y_data[i][j]

            z_data = h_function(weights, [1,x,y])
            z_list[i, j] = z_data

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_data, y_data = np.meshgrid(petal_length, petal_width)

    surf = ax.plot_surface(x_data, y_data, z_list, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

##################################################
def p1partE() -> None: #DONE
    #get a datapoint that is obviously in class 0
    class0_dp1 = (petal_length_all[79], petal_width_all[79]) #versicolor 2nd iris class
    class0_dp2 = (petal_length_all[88], petal_width_all[88])

    #get a datapoint that is obviously in class 1
    class1_dp1 = (petal_length_all[102], petal_width_all[102]) #virginica 3rd iris class
    class1_dp2 = (petal_length_all[109], petal_width_all[109])

    #get two datapoints that are abmbiguous
    amb_dp1 = (petal_length_all[70], petal_width_all[70])
    amb_dp2 = (petal_length_all[119], petal_width_all[119])


    print("Unambigiuosly class 0 : versicolor")
    print("     Point = "+ str(class0_dp1))
    print("     classification value: " + str(p1partB(class0_dp1)))
    print("     Point = "+ str(class0_dp2))
    print("     classification value: " + str(p1partB(class0_dp2)))

    print("Unambigiuosly class 1 : virginica")
    print("     Point = "+ str(class1_dp1))
    print("     classification value: " + str(p1partB(class1_dp1)))
    print("     Point = "+ str(class1_dp2))
    print("     classification value: " + str(p1partB(class1_dp2)))

    print("Close to boundary")
    print("     Point = "+ str(amb_dp1))
    print("     classification value: " + str(p1partB(amb_dp1)))
    print("     Point = "+ str(amb_dp2))
    print("     classification value: " + str(p1partB(amb_dp2)))


    plt.figure(1)

    plt.plot(petal_length_all[79], petal_width_all[79], 'bo')
    plt.plot(petal_length_all[88], petal_width_all[88], 'bo')

    plt.plot(petal_length_all[102], petal_width_all[102], 'rx')
    plt.plot(petal_length_all[109], petal_width_all[109], 'rx')

    plt.plot(petal_length_all[70], petal_width_all[70], 'kv')
    plt.plot(petal_length_all[119], petal_width_all[119], 'kv')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor'))
    plt.show()



get_data_from_file()

#p1partA()
#p1partB()
p1partC()
#p1partD()
#p1partE()
