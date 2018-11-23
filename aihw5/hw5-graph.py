import matplotlib.pyplot as plt
import csv
from scipy.special import comb
import numpy as np
from random import shuffle
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
        n_choose_y = comb(num_of_trials, num_of_trials)
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
    #sum = 0.0
    prod = 1
    for candy in candy_list :
        candy_given_h = get_prob_data_given_hypoth(candy, hypothesis_num)
        #print(np.log(candy_given_h))
        #sum = sum + np.log(candy_given_h)
        prod = prod * candy_given_h
    #print("--------------------------------------")
    #print("list size = "+ str(len(candy_list)))
    #print(sum)

    #return np.exp(sum)
    return prod

def posterior_prob_of_hypothesis(hypothesis_num: int , candy_list: List) -> List: #work on this
    h_prod = [0.1, 0.2, 0.4, 0.2, 0.1]
    # we arre going to add up all the logs instead of multipy
    data_given_hypoth_sum = 0
    data_given_hypoth_list = []

    for i in range(0,len(candy_list)):
        temp_list = candy_list[0:i]
        likel = likelihood(temp_list, hypothesis_num)
        #print("likelyhood = "+ str(likel))
        #print(likel * h_prod[hypothesis_num - 1])

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
def make_ten_candy_lists(lime: int, cherry: int) -> dict:
    candy_lists = {}

    for i in range(0,10):
        candy_lists[i] = get_candy_list(lime, cherry)

    return candy_lists

#todo - instead of doing this, plot the difference of the values for the average ????
#then only do one graph???
def ave_values_at_index(index: int, hypothesis_num: int, list_of_data: List) -> float:
    divisor = len(list_of_data)

    sum = 0

    for t in list_of_data:
        hypoth_list = t[hypothesis_num -1]
        sum = sum + hypoth_list[index]

    return sum/divisor


def get_posterior_for_candy_lists(candy_lists: dict, num: int) -> Tuple:
    list_of_data = []
    #print(candy_lists[])

    for j in range(0, num):

        data1 = posterior_prob_of_hypothesis(1, candy_lists[j])
        data2 = posterior_prob_of_hypothesis(2, candy_lists[j])
        data3 = posterior_prob_of_hypothesis(3, candy_lists[j])
        data4 = posterior_prob_of_hypothesis(4, candy_lists[j])
        data5 = posterior_prob_of_hypothesis(5, candy_lists[j])

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

        list_of_data.append( (data1, data2, data3, data4, data5) )

    #now we have the normal data for just one graph
    #we need to do this for every candy list in the main List
    #and then average the results

    if(num == 1):
        return list_of_data[0]
    else:
        final_data1 = []
        final_data2 = []
        final_data3 = []
        final_data4 = []
        final_data5 = []

        example = list_of_data[0]
        ex2 = example[0]
        #print(typeof(example))

        for j in range(0, len(example[0])):
            final_data1.append(ave_values_at_index(j, 1, list_of_data))
            final_data2.append(ave_values_at_index(j, 2, list_of_data))
            final_data3.append(ave_values_at_index(j, 3, list_of_data))
            final_data4.append(ave_values_at_index(j, 4, list_of_data))
            final_data5.append(ave_values_at_index(j, 5, list_of_data))

        return (final_data1, final_data2, final_data3, final_data4, final_data5)


def make_plots_for_p2partC(candy_lists: dict, hypothesis_num: int, title: str) -> None:

    x = range(0, len(candy_lists[0]))

    plt.figure(1)


    data_tuple = get_posterior_for_candy_lists(candy_lists, hypothesis_num)
    plt.subplot(1, 3, 1)
    plt.title("One Data Set")
    plt.plot(x, data_tuple[0],'r',x,data_tuple[1],'b',x,data_tuple[2],'g',x,data_tuple[3],'m',x,data_tuple[4],'k')


    data_tuple5 = get_posterior_for_candy_lists(candy_lists, 5)
    plt.subplot(1, 3, 2)
    plt.title("Five Data Sets")
    plt.plot(x, data_tuple5[0],'r',x,data_tuple5[1],'b',x,data_tuple5[2],'g',x,data_tuple5[3],'m',x,data_tuple5[4],'k')


    data_tuple10 = get_posterior_for_candy_lists(candy_lists, 10)
    plt.subplot(1, 3, 3)
    plt.title("Ten Data Sets")
    plt.plot(x, data_tuple10[0],'r',x,data_tuple10[1],'b',x,data_tuple10[2],'g',x,data_tuple10[3],'m',x,data_tuple10[4],'k')

    plt.suptitle(title)
    plt.show()

# ai hw 5 problem 2 part c
def hw5_p2_partc() -> None:

    all_cherry_lists = make_ten_candy_lists(0, 100)
    most_cherry_lists = make_ten_candy_lists(25,75)
    half_cherry_lists = make_ten_candy_lists(50,50)
    less_cherry_lists = make_ten_candy_lists(75, 25)
    all_lime_lists = make_ten_candy_lists(100,0)

    make_plots_for_p2partC(all_cherry_lists, 1, "All Cherry Candies")
    make_plots_for_p2partC(most_cherry_lists, 2, "75% Cherry Candies")
    make_plots_for_p2partC(half_cherry_lists, 3, "Half and Half")
    make_plots_for_p2partC(less_cherry_lists, 4, "25% Cherry Candies")
    make_plots_for_p2partC(all_lime_lists, 5, "All Lime Candies")

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
            sepal_width.append(row['sepal_width'])
            sepal_length.append(row['sepal_length'])
            petal_width.append(row['petal_width'])
            petal_length.append(row['petal_length'])
            species_data.append(row['species'])


def plot_cluster(cluster_x: List, cluster_y: List, cluster_assign: List, title: str) -> None:
    plt.figure(1)

    for i in range(0, len(petal_length)):
        marker = 'o'
        if(cluster_assign[i] == 0):
            marker = 'x'
        elif(cluster_assign[i] == 1):
            marker = "v"

        x = petal_length[i]
        y = petal_width[i]

        plt.plot(x , y, marker = marker)


    plt.plot(cluster_x, cluster_y, marker = "*")

    plt.show()
    pass()

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dist = np.sqrt( np.power((x2 - x1), 2) + np.power((y2 - y1), 2) )
    return np.absolute(dist)


def kmeans(k: int) -> None:
    means_x = [] #length
    means_y = [] #width
    #select k points at random
    for i in range(0, k):
        rand = random.randint(1,len(petal_length))
        rand_y = random.randint(1,len(petal_length))

        means_x.append(petal_length[rand])
        means_y.append(petal_width[rand])

    #TODO PLOT These points

    #Calculate the distance between each data point and cluster centers.

    distances_from_clusters = {}

    for i in range(0, k):
        distance_list = []
        for j in range(0,len(petal_length)):
            x1 = petal_length[j]
            x2 = means_x[i]

            y1 = petal_width[j]
            y2 = means_y[i]

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


        cluster_assign.appned((index)) #0, 1, 2


    #all things assigned to cluster 1 will be in one calcualtion


    #estimate (update) means


    #repeat until converged

def extra_credit() -> None:
    kmeans(2)
    kmeans(3)

##################################################################################################################
#hw5_p1_partc()
#hw5_p2_parta()
#hw5_p2_partc()
get_data_from_file()
