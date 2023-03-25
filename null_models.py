import networkx as nx
from networtest import G, N, E
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def caracGraph_ER(G, number_loops):
    '''
    G : graph that we want want to compare the properties

    function that generates random networks with N (number of nodes) and E (number of edges fixed) and compares it to a graph G
    it therefore returns various distributions and compares it to the properties of the graph G.
    '''
    # list of degrees of G
    b = list(G.degree())
    deglist = []
    for i in range(len(b)):
        deglist.append(b[i][1])

    # diameter of the graph of G
    diameter = nx.diameter(G)

    # number of nodes/edges of G
    N = nx.number_of_nodes(G)
    E = nx.number_of_edges(G)

    # average clustering of G
    clust = nx.clustering(G)
    dclust = list(clust.values())
    avclust = np.sum(dclust)/len(dclust)

    # shortest path of G
    shortest = nx.average_shortest_path_length(G)

    # list of the properies of our network G already defined [kmin,kmax,<c>,d,l]
    properties = [np.min(deglist), np.max(deglist), avclust, diameter,
                  shortest]
    k_min = []
    k_max = []
    clust = []
    diam = []
    l = []
    z_score = {}
    # loops of random network
    for n in range(number_loops):
        Grand = nx.gnm_random_graph(N, E, seed=np.random)
        # list of tuples where the first element is the node and the 2nd element its degree
        d = list(Grand.degree())
        degrand = []
        for i in range(N):
            degrand.append(d[i][1])

        # add degree min and max
        k_min.append(round(np.min(degrand), 1))
        k_max.append(round(np.max(degrand), 1))

        # add average clustering
        clustrand = nx.clustering(Grand)
        dclustrand = list(clustrand.values())
        clust.append(np.sum(dclustrand)/N)

        # remove the nodes of degree 0 , because otherwise we will have an error if we try to determive the average shortest path and the diameter d
        for i in range(N):
            if (d[i][1] == 0):
                Grand.remove_node(d[i][0])
        # add diameter and average shortest path
        diam.append(nx.diameter(Grand))
        l.append(nx.average_shortest_path_length(Grand))

    # we stock our various list in a 2D array
    distrib = np.array([*k_min, *k_max, *clust, *diam, *l]
                       ).reshape(len(properties), number_loops)

    # determination of the average and stantard deviations of our various distributions, we are going to use it the for z-score and p-value

    '''for j in range(len(properties)):
        sig = 0
        mu = round(np.sum(distrib[j, :])/number_loops, 4)
        for i in distrib[j, :]:
            sig = sig+(i-mu)**2
        sig = round(np.sqrt(sig/number_loops), 4)
        if sig == 0:
            z_score[(round(properties[j], 4), sig, mu)
                    ] = 'valeur de sigma nulle'
        else:
            z_score[(round(properties[j], 4), sig, mu)] = round(
                np.abs((properties[j]-mu)/sig), 4)

    # create a DataFrame containing the z_score for eaach property
    data_z_score = pd.DataFrame(z_score, index=['z_score'], columns=[
                                'k_min', 'k_max', 'average clustering', 'diameter', 'average shortest path'])'''

    # sub figures of my distributions----------------------------------------------------------------------------------------------------------
    figr, axr = plt.subplots(5, figsize=(12, 27))
    axr = axr.ravel()

    # distribution of k_min
    if (np.min(deglist) > np.max(k_min)):
        axr[0].hist(k_min, bins=int(np.max(k_min)), range=(
            0, np.min(deglist)), color='lightblue')
    else:
        axr[0].hist(k_min, bins=int(np.max(k_min)), range=(
            0, np.max(k_min)), color='lightblue')
    axr[0].set(xlabel='k_min ', ylabel='occurence')
    axr[0].set_title('Distribution of k_min')
    axr[0].vlines(np.min(deglist), 0, k_min.count(max(k_min, key=k_min.count)), colors='red',
                  label=f'k_min={properties[0]}')

    # distribution of k_max
    if (np.max(deglist) > np.max(k_max)):
        axr[1].hist(k_max, bins=int(np.max(k_max)), range=(
            0, np.max(deglist)), color='lightblue')
    else:
        axr[1].hist(k_max, bins=int(np.max(k_max)), range=(
            0, np.max(k_max)), color='lightblue')
    axr[1].set(xlabel='k_max', ylabel='occurence')
    axr[1].set_title('Distribution of k_max')
    axr[1].vlines(np.max(deglist), 0, k_max.count(max(k_max, key=k_max.count)), colors='red',
                  label=f'k_max={properties[1]}')

    # distribution of average clustering <c>
    if (avclust > (np.max(clust))):
        axr[2].hist(clust, bins=50, range=(
            0, avclust), color='lightblue')
    else:
        axr[2].hist(clust, bins=50, range=(
            0, np.max(clust)), color='lightblue')
    axr[2].set(xlabel='<c> ', ylabel='occurence')
    axr[2].set_title('Distribution of <c>')
    axr[2].vlines(avclust, 0, 80, colors='red',
                  label=f'<c>= {properties[2]:.4f}')

    # distribution of average shortest path l
    if (shortest > np.max(l)):
        axr[3].hist(l, bins=50, range=(
            0, shortest), color='lightblue')
    else:
        axr[3].hist(l, bins=50, range=(0, np.max(l)), color='lightblue')
    axr[3].set(xlabel='l', ylabel='occurence')
    axr[3].set_title('Distribution of average shortest path l')
    axr[3].vlines(shortest, 0, 80, colors='red',
                  label=f'l= {properties[3] :.4f}')

    # distribution of diameter d
    if (diameter > np.max(diam)):
        axr[4].hist(diam, bins=50, range=(
            0, diameter), color='lightblue')
    else:
        axr[4].hist(diam, bins=50, range=(
            0, np.max(diam)), color='lightblue')
    axr[4].set(xlabel='d', ylabel='occurence')
    axr[4].set_title('Distribution of diameter (longest shortest path)')
    axr[4].vlines(diameter, 0, diam.count(max(diam, key=diam.count)), colors='red',
                  label=f'd={properties[4]:.4f}')

    #figr.suptitle('Distribution of various parameters of our network, with N and E fixed')

    # Applying a legend and a grid to each subplot
    for i in range(5):
        axr[i].legend()
        axr[i].grid()

    # save all into a figure
    plt.savefig('Random_distributions_E_N_fixed')

    # DataFrmae of all parameters for each graph generated
    '''proprandom = pd.DataFrame
        {'k_min': k_min, 'k_max': k_max, 'clust': clust, 'diam': diam, 'l': l})'''

    # return "completed" to signal the end of the programm
    return ('completed')


if __name__ == "__main__":

    # running the function
    test = caracGraph_ER(G, 100)
    print(test)
