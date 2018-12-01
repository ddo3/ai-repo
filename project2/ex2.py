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


#def p2partA() -> None:
#def p2partB() -> None:
#def p2partC() -> None:
#def p2partD() -> None:
#def p2partE() -> None:

get_data_from_file()
