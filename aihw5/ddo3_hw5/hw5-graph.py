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



#Uncomment these lines each probelm

#hw5_p1_partb()
#hw5_p1_partc()
#hw5_p2_parta()
#hw5_p2_partc()
