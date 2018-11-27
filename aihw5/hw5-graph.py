import matplotlib.pyplot as plt
import csv
from scipy.special import comb
import numpy as np
from random import shuffle
import random
from typing import List, Tuple
import math

# ai hw 5 problem 1 part b
def hw5_p1_partb(): #DONE
    y_values = np.linspace( 0, 5, 40 )
    #could also use np.arrange(0, 10, .5)
    likelihood = []

    theta = 3/4
    n = 4

    for y in y_values:
        n_choose_y = comb(n, y)
        sec_term = pow(theta, y)
        third_term = pow((1-theta), (n - y))

        likelihood.append(n_choose_y * sec_term * third_term)

    plt.plot(y_values, likelihood)
    plt.ylabel('P(Y| n = 4, theta = 3/4)')
    plt.xlabel('Y')
    plt.show()
#################################################################################################################################
def posterior_dist_theta(thetas: list, num_of_trials: int, num_of_heads: int) -> List:
    posterior = []

    for theta in thetas:
        n_choose_y = comb(num_of_trials, num_of_heads)
        sec_term = pow(theta, num_of_heads)
        third_term = pow((1-theta), (num_of_trials - num_of_heads))

        likelihood = n_choose_y * sec_term * third_term

        posterior.append(likelihood  * (1 + num_of_trials))

    return posterior

# ai hw 5 problem 1 part c
def hw5_p1_partc():#DONE
    plt.figure(1)

    #after head 1
    plt.subplot(2, 2, 1)
    plt.title("Heads = 1, Tails = 0")
    plt.xlabel("Theta")
    plt.ylabel("P(theta | y = 1, n = 1)")

    theta = np.arange(0, 1, .1)

    posterior1 = posterior_dist_theta(theta, 1, 1)
    plt.plot(theta, posterior1)

    #after head 2
    plt.subplot(2, 2, 2)
    plt.title("Heads = 2, Tails = 0")
    plt.xlabel("Theta")
    plt.ylabel("P(theta | y = 2, n = 2)")
    posterior2 = posterior_dist_theta(theta, 2, 2)
    plt.plot(theta, posterior1)

    #after tail 1
    plt.subplot(2, 2, 3)
    plt.title("Heads = 2, Tails = 1")
    plt.xlabel("Theta")
    plt.ylabel("P(theta | y = 2, n = 3)")
    posterior1 = posterior_dist_theta(theta, 3, 2)
    plt.plot(theta, posterior1)

    #after head 3
    plt.subplot(2, 2, 4)
    plt.title("Heads = 3, Tails = 1")
    plt.xlabel("Theta")
    plt.ylabel("P(theta | y = 3, n = 4)")
    posterior1 = posterior_dist_theta(theta, 4, 3)
    plt.plot(theta, posterior1)

    plt.show()
########################################################################################################################
def get_candy_list(lime: int, cherry: int) -> List:
    # lime = 0
    # cherry = 1

    final_list = []
    lime_list = [0] * lime
    cherry_list = [1] * cherry

    final_list.extend(lime_list)
    final_list.extend(cherry_list)

    shuffle(final_list)

    return final_list

def get_prob_data_given_hypoth(candy: int, hypothesis_num: int) -> float:
    if(hypothesis_num == 1): # all cherry
        if(candy == 0): #candy is lime
            return 0
        else:
            return 1
    elif(hypothesis_num == 2): # 75% cherry
         if(candy == 0): #candy is lime
             return 0.25
         else:
             return 0.75
    elif(hypothesis_num == 3): # 50/50
        return .5
    elif(hypothesis_num == 4): # 25% cherry
        if(candy == 1): #candy is cherry
            return 0.25
        else:
            return 0.75
    else: #all lime
        if(candy == 1): #candy is cherry
            return 0
        else:
            return 1

def likelihood(candy_list: List, hypothesis_num: int) -> float:
    prod = 1
    for candy in candy_list :
        candy_given_h = get_prob_data_given_hypoth(candy, hypothesis_num)
        prod = prod * candy_given_h

    return prod

def posterior_prob_of_hypothesis(hypothesis_num: int , candy_list: List) -> List: #work on this
    h_prod = [0.1, 0.2, 0.4, 0.2, 0.1]
    # we arre going to add up all the logs instead of multipy
    data_given_hypoth_sum = 0
    data_given_hypoth_list = []

    for i in range(0,len(candy_list)):
        temp_list = candy_list[0:i]
        likel = likelihood(temp_list, hypothesis_num)
        data_given_hypoth_list.append( likel * h_prod[hypothesis_num - 1])

    return data_given_hypoth_list #normalize(data_given_hypoth_list, h_prod[hypothesis_num - 1])

def make_plots_for_p2partA( candy_list: List, title: str) -> None:
    plt.figure(1)

    data1 = posterior_prob_of_hypothesis(1, candy_list)
    data2 = posterior_prob_of_hypothesis(2, candy_list)
    data3 = posterior_prob_of_hypothesis(3, candy_list)
    data4 = posterior_prob_of_hypothesis(4, candy_list)
    data5 = posterior_prob_of_hypothesis(5, candy_list)

    for i in range(0 , len(data1)):
        d1val = data1[i]
        d2val = data2[i]
        d3val = data3[i]
        d4val = data4[i]
        d5val = data5[i]

        sum = d1val + d2val + d3val + d4val + d5val

        data1[i] = d1val/sum
        data2[i] = d2val/sum
        data3[i] = d3val/sum
        data4[i] = d4val/sum
        data5[i] = d5val/sum

    x = range(0, len(candy_list))

    plt.ylim(0, 1)


    plt.subplot(1, 2, 1)
    plt.ylabel('Posterior probability of hypothesis')
    plt.xlabel('Number of observations in d')
    plt.plot(x, data1,'r',x,data2,'b',x,data3,'g',x,data4,'m',x,data5,'k')
    plt.gca().legend(('h1','h2', 'h3', 'h4', 'h5'))

    #do prediction calcualtion

    prediction_values = []

    prob_next_candy_is_lime_given_h1 = get_prob_data_given_hypoth(0, 1)
    prob_next_candy_is_lime_given_h2 = get_prob_data_given_hypoth(0, 2)
    prob_next_candy_is_lime_given_h3 = get_prob_data_given_hypoth(0, 3)
    prob_next_candy_is_lime_given_h4 = get_prob_data_given_hypoth(0, 4)
    prob_next_candy_is_lime_given_h5 = get_prob_data_given_hypoth(0, 5)

    for i in range(0, len(data1)):
        val1 = prob_next_candy_is_lime_given_h1 * data1[i]
        val2 = prob_next_candy_is_lime_given_h2 * data2[i]
        val3 = prob_next_candy_is_lime_given_h3 * data3[i]
        val4 = prob_next_candy_is_lime_given_h4 * data4[i]
        val5 = prob_next_candy_is_lime_given_h5 * data5[i]
        prediction_values.append(val1 + val2 + val3 + val4 + val5)


    plt.subplot(1,2,2)
    plt.ylabel('Probability that next candy is lime')
    plt.xlabel('Number of observations in d')

    plt.plot(x, prediction_values)

    plt.suptitle(title)
    plt.show()

# ai hw 5 problem 2 part a
def hw5_p2_parta():#DONE

    all_cherry = get_candy_list(0,100)
    all_lime = get_candy_list(100,0)
    half = get_candy_list(50,50)
    more_than_half_cherry = get_candy_list(25,75)
    less_than_half_cherry = get_candy_list(75,25)

    make_plots_for_p2partA(all_cherry, 'all_cherry')
    make_plots_for_p2partA(more_than_half_cherry, 'more_than_half_cherry')
    make_plots_for_p2partA(half, "Half and Half")
    make_plots_for_p2partA(less_than_half_cherry, 'less_than_half_cherry')

####################################################################################################################################

def make_plots_for_p2partC(candy_lists: tuple) -> None:

    x = range(0, len(candy_lists[0]))

    plt.figure(1)

    hypothesis = 1
    while (hypothesis != 6):

        data1 = posterior_prob_of_hypothesis(hypothesis, candy_lists[0])
        data2 = posterior_prob_of_hypothesis(hypothesis, candy_lists[1])
        data3 = posterior_prob_of_hypothesis(hypothesis, candy_lists[2])
        data4 = posterior_prob_of_hypothesis(hypothesis, candy_lists[3])
        data5 = posterior_prob_of_hypothesis(hypothesis, candy_lists[4])

        final_list = []

        for i in range(0, len(data1)):
            sum = data1[i] + data2[i] + data3[i] + data4[i] + data5[i]
            final_list.append( sum/5 )


        plt.plot(x, final_list)

        hypothesis = hypothesis + 1

    plt.gca().legend(('h1','h2', 'h3', 'h4', 'h5'))
    plt.show()

# ai hw 5 problem 2 part c
def hw5_p2_partc() -> None:

    all_cherry_list = get_candy_list(0, 100)
    most_cherry_list = get_candy_list(25, 75)
    half_cherry_list = get_candy_list(50, 50)
    less_cherry_list = get_candy_list(75, 25)
    all_lime_list = get_candy_list(100, 0)

    candy_lists = (all_cherry_list, most_cherry_list, half_cherry_list, less_cherry_list, all_lime_list)

    make_plots_for_p2partC(candy_lists)

#################################################################################################
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


def plot_cluster(cluster_x: list, cluster_y: list, cluster_assign: list, title: str) -> None:

    plt.figure(1)

    for i in range(0, len(petal_length)):
        marker = 'bo'
        if(cluster_assign[i] == 0):
            marker = 'rx'
        elif(cluster_assign[i] == 1):
            marker = "gv"

        x = petal_length[i]
        y = petal_width[i]
        if(i == 0):
            plt.plot(x , y, 'ms')
        else:
            plt.plot(x , y, marker)

    #plt.plot(petal_length, petal_width)

    plt.plot(cluster_x, cluster_y, "k*")

    plt.show()

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    term1 = np.power(x2 - x1, 2)
    term2 = np.power(y2 - y1, 2)

    addition  = term1 + term2

    dist = math.sqrt(  addition  )

    return np.absolute(dist)

def plot_with_mean_points(mean_x: list, mean_y: list, title: str) -> None:

    #plot the data
    plt.plot(petal_length, petal_width, 'o')

    plt.plot(mean_x, mean_y, '*')

    plt.title(title)

    plt.show()

def check_if_need_to_loop(cluster_assign_1: list, cluster_assign_2: list) -> bool:
    for i in range(0, len(cluster_assign_1)):
        if(cluster_assign_1[i] != cluster_assign_2[i]):
            return True

    return False


#SOMEHOW this method is failing
#or my placement is just wrong
def update_cluster_mean( mean_list: list, data_list: list, cluster_assign_list: list) -> List:
    print("***** CLUSTER ASSIGN LIST *****")
    print(cluster_assign_list)
    #print(len(cluster_assign_list))
    new_mean_list = []

    for i in range(0, len(mean_list)):
        count = 0
        sum = 0

        for j in range(0, len(data_list)):
            if( int(cluster_assign_list[j]) == i):
                count = count + 1
                sum = sum + float(data_list[j])

        new_mean_list.append( sum / count)


    return new_mean_list

def get_distances_from_clusters(cluster_x: list, cluster_y: list ) -> dict:
    distances_from_clusters = {}

    for i in range(0, len(cluster_x)): #0, 1, 2
        distance_list = []
        x2 = float(cluster_x[i])
        y2 = float(cluster_y[i])

        for j in range(0,len(petal_length)):
            x1 = float(petal_length[j])
            y1 = float(petal_width[j])

            dist = distance(x1, y1, x2 ,y2)
            distance_list.append(dist)

        distances_from_clusters[i] = distance_list

    return distances_from_clusters

def assign_data_to_cluster(distances_from_clusters: dict) -> List:
    cluster_assign = []
    for i in range(0, len(petal_length)):
        cluster_distances_for_i = []

        #get the distances for this point
        for key in distances_from_clusters:
            dist_list = distances_from_clusters[key]
            cluster_distances_for_i.append( dist_list[i] )

        index = 0
        min = cluster_distances_for_i[0]

        for j in range(0, len(cluster_distances_for_i)):
            if(i == 0):
                print("distance %f from cluster %d",cluster_distances_for_i[j], j)

            if cluster_distances_for_i[j] < min:
                min = cluster_distances_for_i[j]
                index = j


        cluster_assign.append((index)) #0, 1, 2

    return cluster_assign

def are_values_in_list_same(l1: list, l2:list ) -> bool:
    for i in range(0, len(l1)):
        if(l1[i] != l2[i]):
            return False

    return True

g_x = []
g_y = []

def kmeans_2(k : int) -> None:
    init_means_x = [] #length
    init_means_y = [] #width

    #select k points at random
    for i in range(0, k):
        rand = random.randint(1,len(petal_length))

        init_means_x.append(float(petal_length[rand]))
        init_means_y.append(float(petal_width[rand]))

    #PLOT These points
    print("####################  INITIAL  ###############################")
    print(init_means_x)
    print(init_means_y)
    plot_with_mean_points(init_means_x, init_means_y, "Initial Means")

    #Calculate the distance between each data point and cluster centers.
    distances_from_clusters = get_distances_from_clusters(init_means_x, init_means_y)

    #assign each point to a cluster based on distance from cluster
    cluster_assign = assign_data_to_cluster(distances_from_clusters)

    #recalcualte new clusters (update x mean and y_mean)
    updated_means_x = update_cluster_mean(init_means_x, petal_length, cluster_assign)
    updated_means_y = update_cluster_mean(init_means_y, petal_width, cluster_assign)

    #plot_with_mean_points(updated_means_x, updated_means_y, "Initial Means")

    g_x = init_means_x
    g_y = init_means_y
    plot_cluster(updated_means_x, updated_means_y, cluster_assign, "title")
    # while the updated mean and the global mean are not the same
    print("###################################################")
    while(not are_values_in_list_same(g_x, updated_means_x)):
        print("Updated Means")
        print(updated_means_x)
        print(updated_means_y)

        g_x = updated_means_x
        g_y = updated_means_y

        #Calculate the distance between each data point and cluster centers.
        distances_from_clusters = get_distances_from_clusters(updated_means_x, updated_means_x)

        #assign each point to a cluster based on distance from cluster
        cluster_assign = assign_data_to_cluster(distances_from_clusters)

        #recalcualte new clusters (update x mean and y_mean)
        updated_means_x = update_cluster_mean(updated_means_x, petal_length, cluster_assign)
        updated_means_y = update_cluster_mean(updated_means_x, petal_width, cluster_assign)


    plot_with_mean_points(updated_means_x, updated_means_y, "Final")


#TODO something is wrong the means are never changing!!!
def kmeans(k: int) -> None:
    init_means_x = [] #length
    init_means_y = [] #width

    #select k points at random
    for i in range(0, k):
        rand = random.randint(1,len(petal_length))

        init_means_x.append(petal_length[rand])
        init_means_y.append(petal_width[rand])

    #PLOT These points
    plot_with_mean_points(init_means_x, init_means_y, "Initial Means")

    #Calculate the distance between each data point and cluster centers.
    distances_from_clusters = {}

    for i in range(0, k):
        distance_list = []
        for j in range(0,len(petal_length)):
            x1 = float(petal_length[j])
            x2 = float(init_means_x[i])

            y1 = float(petal_width[j])
            y2 = float(init_means_y[i])

            distance_list.append(distance(x1, y1, x2 ,y2))

        distances_from_clusters[i] = distance_list

    #assign each point to a cluster based on distance from cluster
    cluster_assign = []

    for i in range(0, len(petal_length)):
        cluster_distances_for_i = []

        #get the distances for this point
        for key in distances_from_clusters:
            dist_list = distances_from_clusters[key]
            cluster_distances_for_i.append( dist_list[i] )

        index = 0
        min = cluster_distances_for_i[0]

        for j in range(0, len(cluster_distances_for_i)):
            if cluster_distances_for_i[j] < min:
                index = j


        cluster_assign.append((index)) #0, 1, 2

    ##############################################################################

    #Calculate a new cluster mean
    new_cluster_assign = []

    #recalcualte new clusters (update x mean and y_mean)
    new_means_x = update_cluster_mean(init_means_x, petal_length, cluster_assign)
    new_means_y = update_cluster_mean(init_means_y, petal_width, cluster_assign)

    dist_from_clusters = {}
    #calcualte distance from clusters all other points
    for i in range(0, k):
        distance_list = []
        for j in range(0,len(petal_length)):
            x1 = float(petal_length[j])
            x2 = float(new_means_x[i])

            y1 = float(petal_width[j])
            y2 = float(new_means_y[i])

            distance_list.append(distance(x1, y1, x2 ,y2))

        dist_from_clusters[i] = distance_list

    #assign data points to new clusters
    for i in range(0, len(petal_length)):
        cluster_distances_for_i = []

        #get the distances for this point
        for key in distances_from_clusters:
            dist_list = distances_from_clusters[key]
            cluster_distances_for_i.append( dist_list[i] )

        index = 0
        min = cluster_distances_for_i[0]

        for j in range(0, len(cluster_distances_for_i)):
            if cluster_distances_for_i[j] < min:
                index = j


        new_cluster_assign.append((index)) #0, 1, 2

    #
    count = 0


    old_cluster_assign = cluster_assign
    old_mean_x = init_means_x
    old_means_y = init_means_y



    while(check_if_need_to_loop(new_means_x, old_mean_x)):
    #while(check_if_need_to_loop(new_cluster_assign, old_cluster_assign)):
        old_cluster_assign = new_cluster_assign
        old_means_x = new_means_x
        old_means_y = new_means_y



        if(count % 100 == 0):
            plot_with_mean_points(new_means_x, new_means_y, "test Means")
        count = count + 1

        #recalcualte new clusters (update x mean and y_mean)
        new_means_x = update_cluster_mean(old_means_x, petal_length, new_cluster_assign)
        new_means_y = update_cluster_mean(old_means_y, petal_width, new_cluster_assign)

        #old_cluster_assign = new_cluster_assign

        #
        dist_from_clusters = {}

        #calcualte distance from clusters all other points
        for i in range(0, k):
            distance_list = []
            for j in range(0,len(petal_length)):
                x1 = float(petal_length[j])
                x2 = float(new_means_x[i])

                y1 = float(petal_width[j])
                y2 = float(new_means_y[i])

                distance_list.append(distance(x1, y1, x2 ,y2))

            dist_from_clusters[i] = distance_list

        #assign data points to new clusters
        for i in range(0, len(petal_length)):
            cluster_distances_for_i = []

            #get the distances for this point
            for key in distances_from_clusters:
                dist_list = distances_from_clusters[key]
                cluster_distances_for_i.append( dist_list[i] )

            index = 0
            min = cluster_distances_for_i[0]

            for j in range(0, len(cluster_distances_for_i)):
                if cluster_distances_for_i[j] < min:
                    index = j


            new_cluster_assign.append((index)) #0, 1, 2




    plot_with_mean_points(new_means_x, new_means_y, "Final Means")



def extra_credit() -> None:
    get_data_from_file()
    #kmeans_2(3)
    kmeans_2(2)
    #kmeans(3)

##################################################################################################################
#hw5_p1_partb()
#hw5_p1_partc()
#hw5_p2_parta()
#hw5_p2_partc()

#extra_credit()
