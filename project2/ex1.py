import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
species_data = []



def get_data_from_file() -> None: #DONE

    with open('irisdata.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #
            sepal_width.append(float(row['sepal_width']))
            sepal_length.append(float(row['sepal_length']))
            petal_width.append(float(row['petal_width']))
            petal_length.append(float(row['petal_length']))
            species_data.append(row['species'])

###############################################

def p1partA() -> None: #DONE
    species1_x = []
    species1_y = []

    species2_x = []
    species2_y = []

    for index in range(0, len(species_data)):
        species = species_data[index]

        if(species == 'virginica'):
            species1_x.append(petal_length[index]);
            species1_y.append(petal_width[index]);

        elif(species == 'versicolor'):
            species2_x.append(petal_length[index]);
            species2_y.append(petal_width[index]);

    plt.figure(1)


    plt.plot(species1_x, species1_y,'o', species2_x, species2_y, 'x')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor'))
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

def dot_prod(l1: list, l2: list) -> float:
    sum = 0
    for i in range(0, len(l1)):
        sum = sum + (l1[i] * l2[i])

    return sum

def h_function(w_list: list, x_list: list) -> float:
    bottom =  1 + np.exp(-1 * dot_prod(w_list, x_list))
    return 1/bottom

def get_weight_list(w_list: list, x_list: list, y: float) -> List:
    new_w_list = []
    alpha = 1

    # TODO this may need to changed to either plus or minus
    for i in range(0, len(w_list)):
        data = w_list[i] + (alpha * h_function(w_list, x_list)) * (h_function(w_list, x_list) * (1 - h_function(w_list, x_list) ) * x_list[i])
        new_w_list.append(data)
    return new_w_list

def not_converged(c: list, y:float, w_list: list) -> bool:
    print("Hfucntion should be getting close to 0")
    print(h_function(w_list, c))

    if(y == 0):
        if( y == h_function(w_list, c)):
        #if( np.isclose(y, h_function(w_list, c))):
            return False
        else:
            return True

    else:

        if( y == h_function(w_list, c)):
            return False
        else:
            return True

def test_partB() -> None:
    species1_x = []
    species1_y = []

    species2_x = []
    species2_y = []


    for index in range(0, len(species_data)):
        species = species_data[index]

        if(species == 'virginica'): #3rd class
            species1_x.append(petal_length[index]);
            species1_y.append(petal_width[index]);

        elif(species == 'versicolor'): #2nd class
            species2_x.append(petal_length[index]);
            species2_y.append(petal_width[index]);

    for i in range(0, len(species2_x)): # versicolor
        x = species2_x[i]
        y = species2_y[i]

        print( p1partB((x,y)) )

def testing_weights() -> None:

    for x in x_list:
        y = (w1 * x) + w0
        datapoints.append( (x,y) )

def p1partB(datapoint: tuple) -> int: #DONE
    x = datapoint[0]
    y = datapoint[1]

    #weights are self defined
    #weights = [.015384, -.0492]
    #weights = [.35714, -1.142]
    #weights = [-0.09694660358659667, 0.3075453098834738]

    weights = [.12, -.36]

    value = h_function( weights, [x,y])

    if(value < 0.5):
        return 1
    else:
        return 0

###############################################

def p1partC() -> None: #DONE
    species1_x = []
    species1_y = []

    species2_x = []
    species2_y = []

    for index in range(0, len(species_data)):
        species = species_data[index]

        if(species == 'virginica'):
            species1_x.append(petal_length[index]);
            species1_y.append(petal_width[index]);

        elif(species == 'versicolor'):
            species2_x.append(petal_length[index]);
            species2_y.append(petal_width[index]);

    plt.figure(1)


    plt.plot(species1_x, species1_y,'o', species2_x, species2_y, 'x')

    x_val = []
    x_val.extend(species2_x)
    x_val.extend(species1_x)

    y_val = []
    y_val.extend(species2_y)
    y_val.extend(species1_y)

    x = []
    y = []

    for i in range(0, len(y_val)):
        x_d = x_val[i]
        y_d = y_val[i]

        y_data = -0.5* x_d + 4

        x.append(x_d)
        y.append(y_data)

    plt.plot(x, y, 'k')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor'))
    plt.show()

##################################################
def p1partD() -> None: #DONE

    x = []
    y = []
    z = []

    for index in range(0, len(species_data)):
        species = species_data[index]

        if not(species == 'setosa'):
            x.append(petal_length[index])
            y.append(petal_width[index])

            z_data = p1partB((petal_length[index], petal_width[index]))

            z.append(z_data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


##################################################
def p1partE() -> None: #DONE
    #get a datapoint that is obviously in class 0
    class0_dp1 = (petal_length[79], petal_width[79]) #versicolor 2nd iris class
    class0_dp2 = (petal_length[88], petal_width[88])

    #get a datapoint that is obviously in class 1
    class1_dp1 = (petal_length[102], petal_width[102]) #virginica 3rd iris class
    class1_dp2 = (petal_length[109], petal_width[109])
    #class1_dp2 = (petal_length[118], petal_width[118])
    #get two datapoints that are abmbiguous
    amb_dp1 = (petal_length[70], petal_width[70])
    amb_dp2 = (petal_length[119], petal_width[119])


    print("Unambigiuosly class 0")
    print("     Point = "+ str(class0_dp1))
    print("     classification value: " + str(p1partB(class0_dp1)))
    print("     Point = "+ str(class0_dp2))
    print("     classification value: " + str(p1partB(class0_dp2)))

    print("Unambigiuosly class 1")
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

    plt.plot(petal_length[79], petal_width[79], 'bo')
    plt.plot(petal_length[88], petal_width[88], 'bo')
    plt.plot(petal_length[102], petal_width[102], 'rx')
    plt.plot(petal_length[109], petal_width[109], 'rx')

    plt.plot(petal_length[70], petal_width[70], 'kv')
    plt.plot(petal_length[119], petal_width[119], 'kv')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    #plt.gca().legend(('virginica', 'versicolor'))
    plt.show()

get_data_from_file()

#print((petal_length[119], petal_width[119]))
# 118 in verginica

#p1partA()
#p1partB()
#p1partC()
#p1partD()
#p1partE()
