import csv
import matplotlib.pyplot as plt
import numpy as np

#'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
species_data = []


def get_data_from_file() -> None:

    with open('irisdata.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #
            sepal_width.append(row['sepal_width'])
            sepal_length.append(row['sepal_length'])
            petal_width.append(row['petal_width'])
            petal_length.append(row['petal_length'])
            species_data.append(row['species'])


def p1partA() -> None: #DONE
    species1_x = []
    species1_y = []

    species2_x = []
    species2_y = []

    species3_x = []
    species3_y = []

    for index in range(0, len(species_data)):
        species = species_data[index]

        if(species == 'virginica'):
            species1_x.append(petal_length[index]);
            species1_y.append(petal_width[index]);

        elif(species == 'versicolor'):
            species2_x.append(petal_length[index]);
            species2_y.append(petal_width[index]);
        #else:
            #species3_x.append(petal_length[index]);
            #species3_y.append(petal_width[index]);

    plt.figure(1)


    plt.plot(species1_x, species1_y,'o', species2_x, species2_y, 'x')#, species3_x, species3_y,'x')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().legend(('virginica', 'versicolor'))
    plt.show()

def p1partB() -> None:
    #look in book 18.6 pg 736!!

def p1partC() -> None:

def p1partD() -> None

def p1partE() -> None:


get_data_from_file()
#p1partA()
