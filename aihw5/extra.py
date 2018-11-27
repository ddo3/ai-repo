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

#Euclidian Distance between two d-dimensional points
def eucldist(p0,p1):
    dist = 0.0
    for i in range(0,len(p0)):
        dist += (p0[i] - p1[i])**2
    return math.sqrt(dist)

#K-Means Algorithm
def kmeans(k,datapoints):

    # d - Dimensionality of Datapoints
    d = len(datapoints[0])

    #Limit our iterations
    Max_Iterations = 1000
    i = 0

    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)

    #Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0,k):
        new_cluster = []
        #for i in range(0,d):
        #    new_cluster += [random.randint(0,10)]
        cluster_centers += [random.choice(datapoints)]


        #Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        #In this particular implementation we want to force K exact clusters.
        #To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False

    print(cluster_centers)

    plot_cluster(cluster_centers, datapoints, cluster, 'Initial Means')
    count = 0
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        #Update Point's Cluster Alligiance
        for p in range(0,len(datapoints)):
            min_dist = float("inf")

            #Check min_distance against all centers
            for c in range(0,len(cluster_centers)):

                dist = eucldist(datapoints[p],cluster_centers[c])

                if (dist < min_dist):
                    min_dist = dist
                    cluster[p] = c   # Reassign Point to new Cluster


        #Update Cluster's Position
        for k in range(0,len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0,len(datapoints)):
                if (cluster[p] == k): #If this point belongs to the cluster
                    for j in range(0,d):
                        new_center[j] += datapoints[p][j]
                    members += 1

            for j in range(0,d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)

                #This means that our initial random assignment was poorly chosen
                #Change it to a new datapoint to actually force k clusters
                else:
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print("Forced Recalculation...")


            cluster_centers[k] = new_center

            if(k == 2):
                if(count == 4):
                    plot_cluster(cluster_centers, datapoints, cluster, 'Middle Means')
            else:
                if(count == 5):
                    plot_cluster(cluster_centers, datapoints, cluster, 'Middle Means')

        count = count + 1
    plot_cluster(cluster_centers, datapoints, cluster, 'Final Means')

    #print("======== Results ========")
    #print("Clusters", cluster_centers)
    #print("Iterations",i)
    #print("Assignments", cluster)

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

    #plt.plot(petal_length, petal_width)
    for data in clusters:
        x = data[0]
        y = data[1]
        plt.plot(x, y, "k*")

    plt.show()

#TESTING THE PROGRAM#
if __name__ == "__main__":
    #2D - Datapoints List of n d-dimensional vectors. (For this example I already set up 2D Tuples)
    #Feel free to change to whatever size tuples you want...
    get_data_from_file()
    datapoints = []

    for i in range(0, len(petal_length)):
        x = petal_length[i]
        y = petal_width[i]
        tup = (x,y)
        datapoints.append(tup)

    kmeans(2,datapoints)

    kmeans(3,datapoints)
