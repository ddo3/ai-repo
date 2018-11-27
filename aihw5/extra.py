# Import necessary libraries
import random
import math
import matplotlib.pyplot as plt
import csv

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
            sepal_width.append(float(row['sepal_width']))
            sepal_length.append(float(row['sepal_length']))
            petal_width.append(float(row['petal_width']))
            petal_length.append(float(row['petal_length']))
            species_data.append(row['species'])


def distance(l0 : list, l1: list) -> float:
    d = 0.0
    for i in range(0,len(l0)):
        d += (l0[i] - l1[i])**2
    return math.sqrt(d)

def kmeans(k: int, datapoints: list):

    d = len(datapoints[0])
    max_iter = 1000
    i = 0

    cluster_assign = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)
    centers = []

    for i in range(0,k):
        new_cluster = []
        centers += [random.choice(datapoints)]
        recalculation = False

    plot_cluster(centers, datapoints, cluster_assign, 'Initial Means')
    count = 0
    while (cluster_assign != prev_cluster) or (i > max_iter) or (recalculation) :

        prev_cluster = list(cluster_assign)
        recalculation = False
        i += 1


        for p in range(0,len(datapoints)):
            min_dist = float("inf")


            for c in range(0,len(centers)):

                dist = distance(datapoints[p],centers[c])

                if (dist < min_dist):
                    min_dist = dist
                    cluster_assign[p] = c

        for k in range(0,len(centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster_assign[p] == k):
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1

            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)
                else:
                    new_center = random.choice(datapoints)
                    recalculation = True


            centers[k] = new_center

            if(k == 2):
                if(count == 4):
                    plot_cluster(centers, datapoints, cluster_assign, 'Middle Means')
            else:
                if(count == 5):
                    plot_cluster(centers, datapoints, cluster_assign, 'Middle Means')

        count = count + 1
    plot_cluster(centers, datapoints, cluster_assign, 'Final Means')

def plot_with_mean_points(mean_x: list, mean_y: list, title: str) -> None:

    #plot the data
    plt.plot(petal_length, petal_width, 'o')

    plt.plot(mean_x, mean_y, '*')

    plt.title(title)

    plt.show()

def plot_cluster(clusters: list, data_list: list, cluster_assign: list, title: str) -> None:

    plt.figure(1)

    for i in range(0, len(data_list)):
        marker = 'bo'
        if(cluster_assign[i] == 0):
            marker = 'rx'
        elif(cluster_assign[i] == 1):
            marker = "gv"

        data = data_list[i]
        x = data[0]
        y = data[1]

        if(i == 0):
            plt.plot(x , y, 'ms')
        else:
            plt.plot(x , y, marker)

    for data in clusters:
        x = data[0]
        y = data[1]
        plt.plot(x, y, "k*")

    plt.show()

def run():
    get_data_from_file()
    datapoints = []

    for i in range(0, len(petal_length)):
        x = petal_length[i]
        y = petal_width[i]
        tup = (x,y)
        datapoints.append(tup)

    kmeans(2,datapoints)

    kmeans(3,datapoints)


run()
