from networtest import G  # import our ges raph G and all its properties from networtes
from networtest import infos
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# kini(V)>kouti(V),∀i∈V COMMUNITY

# subgraph to use for contact matrix!----------------------------------------------------------------------------------------------------------------------

# node(2) and node(478) added
# G.add_node(2,classe='PC*',sex='M')
# G.add_node(478,classe='2BIO2',sex='F')

nodarray = np.array(list(G.nodes()))
edarray = np.array(list(G.edges()))


Fstep = np.array(nodarray[[G.nodes[i]['sex'] == 'F' for i in G.nodes()]])
# we are not taking in consideration the 'Unknown sex'
Mstep = np.array(nodarray[[G.nodes[i]['sex'] == 'M' for i in G.nodes()]])
subF = G.subgraph(Fstep)
subM = G.subgraph(Mstep)
# subgraph where every sex='Unknown' are removed, leaving only male and female, so 322 nodes (7 Unknown sexs removed)
subFM = G.subgraph([*Mstep, *Fstep])
edarrayFM = np.array(list(subFM.edges()))
FMstep = edarrayFM[[subFM.nodes[i]['sex'] !=
                    subFM.nodes[j]['sex'] for i, j in subFM.edges()]]

NF = len(subF.nodes())  # number of F nodes
NM = len(subM.nodes())  # number of M nodes
E_FF = len(subF.edges())  # number of FF links
E_MM = len(subM.edges())  # number of MM links
E_FM = len(FMstep[:, 0])  # number of FM links

# densities inter_sex
pFF = round(2*E_FF/(NF*(NF-1)), 3)
pMM = round(2*E_MM/(NM*(NM-1)), 3)
pFM = round(E_FM/(NF*NM), 3)

# contact matrix and list of labels (sex)
contactsexdata = np.zeros((2, 2))
contactsexdata[0, 0] = pFF
contactsexdata[0, 1] = pFM
contactsexdata[1, 0] = pFM
contactsexdata[1, 1] = pMM
labelsex = ['F', 'M']

# weight contact matrix sex
contactweightsex = np.zeros((2, 2))
weix = list(nx.get_edge_attributes(subF, 'interaction').values())
contactweightsex[0, 0] = round(np.sum(weix)/E_FF, 3)
weix = list(nx.get_edge_attributes(subM, 'interaction').values())
contactweightsex[1, 1] = round(np.sum(weix)/E_MM, 3)
w = 0
for n, p in FMstep:
    w = w+G[n][p]['interaction']
contactweightsex[1, 0] = round(w/E_FM, 3)
contactweightsex[0, 1] = contactweightsex[1, 0]


# contact matrix for classes
diffclass = sorted(list(set(infos['classe'])))
contactclass = {}
numbstudclass = {}
contactclassweight = {}

# densities within class
for cl in diffclass:
    step = np.array(nodarray[[G.nodes[i]['classe'] == cl for i in G.nodes()]])
    sub = G.subgraph(step)
    E = len(sub.edges())
    N = len(sub.nodes())
    wei = list(nx.get_edge_attributes(sub, 'interaction').values())
    numbstudclass[cl] = N
    contactclassweight[(cl, cl)] = round(np.sum(wei)/len(wei), 2)
    contactclass[(cl, cl)] = round((2*E)/(N*(N-1)), 3)

# densities inter class
for cl in diffclass:
    for cl2 in diffclass:
        if cl != cl2:
            step = edarray[[(G.nodes[i]['classe'] == cl and G.nodes[j]['classe'] == cl2) or (
                G.nodes[i]['classe'] == cl2 and G.nodes[j]['classe'] == cl) for i, j in G.edges()]]
            E = len(step[:, 0])
            Ncl = numbstudclass[cl]
            Ncl2 = numbstudclass[cl2]
            w = 0
            for n, p in step:
                w = w+G[n][p]['interaction']
            contactclassweight[(cl, cl2)] = round(w/E, 2)
            contactclass[(cl, cl2)] = round(E/(Ncl*Ncl2), 3)


# figure where we have the contat matrix for both class and sex---------------------------------------------------------------------------------
contactclassdata = np.zeros((len(diffclass), len(diffclass)))
i, j = 0, 0
for i in range(len(diffclass)):
    for j in range(len(diffclass)):
        if (contactclassdata[i, j] == 0):
            contactclassdata[i, j] = round(
                contactclass[(diffclass[i], diffclass[j])], 3)

contactclasswei = np.zeros((len(diffclass), len(diffclass)))
i, j = 0, 0
for i in range(len(diffclass)):
    for j in range(len(diffclass)):
        if (contactclasswei[i, j] == 0):
            contactclasswei[i, j] = round(
                contactclassweight[(diffclass[i], diffclass[j])], 3)

# main, where we implement and draw the obtained figures
if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, figsize=(8, 14))
    im = ax[0].imshow(contactclassdata, cmap='Blues')
    ax[1].imshow(contactsexdata, cmap='Blues')

    # Show all ticks and label them with the respective list entries
    ax[0].set_xticks(np.arange(len(diffclass)), labels=diffclass)
    ax[0].set_yticks(np.arange(len(diffclass)), labels=diffclass)
    ax[1].set_xticks(np.arange(len(labelsex)), labels=labelsex)
    ax[1].set_yticks(np.arange(len(labelsex)), labels=labelsex)

    # Loop over data dimensions and create text annotations.
    for i in range(len(diffclass)):
        for j in range(len(diffclass)):
            text = ax[0].text(j, i, contactclassdata[i, j],
                              ha="center", va="center", color="k")

    for i in range(len(labelsex)):
        for j in range(len(labelsex)):
            text = ax[1].text(j, i, contactsexdata[i, j],
                              ha="center", va="center", color="k")

    # labels on both sides and up/down
    ax[0].tick_params(labeltop=True, labelright=True)
    ax[1].tick_params(labeltop=True, labelright=True)

    # ylabels on both images
    ax[0].set_ylabel('Class')
    ax[1].set_ylabel('Sex')
    fig.colorbar(im, ax=ax, orientation='vertical', location='right',
                 fraction=0.05, shrink=0.5, label='Average weight', anchor=(7, 0.5))

    # title
    fig.suptitle("Contact matrix")
    # plt.show()
    plt.savefig("contact_matrix_classes_sex")

    # weighted contact matrix for class and sex--------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8, 14))
    im = ax[0].imshow(contactclasswei, cmap='Oranges')
    ax[1].imshow(contactweightsex, cmap='Oranges')

    # Show all ticks and label them with the respective list entries
    ax[0].set_xticks(np.arange(len(diffclass)), labels=diffclass)
    ax[0].set_yticks(np.arange(len(diffclass)), labels=diffclass)
    ax[1].set_xticks(np.arange(len(labelsex)), labels=labelsex)
    ax[1].set_yticks(np.arange(len(labelsex)), labels=labelsex)

    # Loop over data dimensions and create text annotations.
    for i in range(len(diffclass)):
        for j in range(len(diffclass)):
            text = ax[0].text(j, i, contactclasswei[i, j],
                              ha="center", va="center", color="k")

    for i in range(len(labelsex)):
        for j in range(len(labelsex)):
            text = ax[1].text(j, i, contactweightsex[i, j],
                              ha="center", va="center", color="k")

    # labels on both sides and up/down
    ax[0].tick_params(labeltop=True, labelright=True)
    ax[1].tick_params(labeltop=True, labelright=True)

    # ylabels on both images
    ax[0].set_ylabel('Class')
    ax[1].set_ylabel('Sex')
    fig.colorbar(im, ax=ax, orientation='vertical', location='right',
                 fraction=0.05, shrink=0.5, label='Average weight', anchor=(7, 0.5))

    # title
    fig.suptitle("Weighted contact matrix")
    plt.savefig("weighted_contact_matrix_classes_sex")
