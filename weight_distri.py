import networkx as nx
import numpy as np
# datatri is the array in where we load the data concerning the interactions, using the "triinitial.py" script and saved in "tij_tri.dat" file
from networtest import datatri, G, N, E, infos
from commu import *


def weight_distri(G, weight_tab, number_loops):
    '''random instances of network G, with only a redistribution of the weights, wich are given by the list 'weight_tab'''
    avstrenght = []
    used_weight = []
    contactweightsexr = np.zeros((2, 2))
    contactclassweightr = {}
    for i in range(number_loops):
        if (len(used_weight) != 0):
            # making sure that we have the same list of weight used for each loop, by deleting it and in the next line fill it with the weight_tab in parameter
            del used_weight[:]
        used_weight = [i for i in weight_tab]
        np.random.shuffle(used_weight)
        for indice in range(len(used_weight)):
            G.add_edge(datatri[indice, 0], datatri[indice, 1],
                       interaction=used_weight[indice])
        # weighted contact matrix for each loop--------------------------------------------------------------------
        # sex
        weixrf = list(nx.get_edge_attributes(subF, 'interaction').values())
        contactweightsexr[0, 0] += np.sum(weixrf)/E_FF
        weixrm = list(nx.get_edge_attributes(subM, 'interaction').values())
        contactweightsexr[1, 1] += np.sum(weixrm)/E_MM
        wr = 0
        for n, p in FMstep:
            wr = wr+G[n][p]['interaction']
        contactweightsexr[1, 0] += round(wr/E_FM, 2)
        contactweightsexr[0, 1] = contactweightsexr[1, 0]
        # classes
        for cl in diffclass:
            stepr = np.array(
                nodarray[[G.nodes[i]['classe'] == cl for i in G.nodes()]])
            subr = G.subgraph(stepr)
            weir = list(nx.get_edge_attributes(subr, 'interaction').values())
            if((cl, cl) in contactclassweightr):
                contactclassweightr[(
                    cl, cl)] += (np.sum(weir)/len(weir))/number_loops
            else:
                contactclassweightr[(cl, cl)] = (
                    np.sum(weir)/len(weir))/number_loops
        # densities inter class
        for cl in diffclass:
            for cl2 in diffclass:
                if cl != cl2:
                    stepr = edarray[[(G.nodes[i]['classe'] == cl and G.nodes[j]['classe'] == cl2) or (
                        G.nodes[i]['classe'] == cl2 and G.nodes[j]['classe'] == cl) for i, j in G.edges()]]
                    Er = len(stepr[:, 0])
                    wr = 0
                    for n, p in stepr:
                        wr = wr+G[n][p]['interaction']
                    if((cl, cl2) in contactclassweightr):
                        contactclassweightr[(cl, cl2)] += (wr/Er)/number_loops
                    else:
                        contactclassweightr[(cl, cl2)] = (wr/Er)/number_loops
    # creation of my contact matrix for classS
    contactclassweir = np.zeros((len(diffclass), len(diffclass)))
    i, j = 0, 0
    for i in range(len(diffclass)):
        for j in range(len(diffclass)):
            if (contactclassweir[i, j] == 0):
                contactclassweir[i, j] = contactclassweightr[(
                    diffclass[i], diffclass[j])]
    # difference between the average density of our weigh contact matrixes and the one of our network------------------------------------------------------------------------
    # sex
    contactweightsexr = np.matrix.round(
        ((-contactweightsexr/number_loops)+contactweightsex), 2)
    # classes
    contactclassweir = np.matrix.round((-contactclassweir+contactclasswei), 2)

    # showing the new obtained figure
    figr, axr = plt.subplots(2, 1, figsize=(8, 14))
    imr = axr[1].imshow(contactweightsexr, cmap='Purples')
    imr2 = axr[0].imshow(contactclassweir, cmap='Purples')
    axr[1].set_xticks(np.arange(len(labelsex)), labels=labelsex)
    axr[1].set_yticks(np.arange(len(labelsex)), labels=labelsex)
    axr[0].set_xticks(np.arange(len(diffclass)), labels=diffclass)
    axr[0].set_yticks(np.arange(len(diffclass)), labels=diffclass)
    for m in range(len(labelsex)):
        for l in range(len(labelsex)):
            text = axr[1].text(l, m, contactweightsexr[m, l],
                               ha="center", va="center", color="k")
    for m in range(len(diffclass)):
        for l in range(len(diffclass)):
            text = axr[0].text(l, m, contactclassweir[m, l],
                               ha="center", va="center", color="k")
    # labels on both sides and up/down
    axr[0].tick_params(labeltop=True, labelright=True)
    axr[1].tick_params(labeltop=True, labelright=True)
    # ylabels on both images
    axr[0].set_ylabel('Class')
    axr[1].set_ylabel('Sex')
    # position and dimension the colorbars
    figr.colorbar(imr, ax=axr[1], orientation='vertical', location='right',
                  fraction=0.05, shrink=0.5, label='difference in density', anchor=(1.5, 0.5))
    figr.colorbar(imr2, ax=axr[0], orientation='vertical', location='right',
                  fraction=0.05, shrink=0.5, label='diferences in density', anchor=(1.5, 0.5))
    # title and saving the figure
    plt.suptitle('Weight redistribution')
    figr.savefig('Weight redistribution_matrixx')
    return 0


weight_distri(G, datatri[:, 2], 100)
