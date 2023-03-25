from networtest import G
from networtest import infos
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# The purpose of this part is to vizualize our netword focusing on different parameters

# network map, colored depending on the attributes
color_mapsex = nx.get_node_attributes(G, 'sex')
color_mapclass = nx.get_node_attributes(G, 'classe')
color_mapedges = nx.get_edge_attributes(G, 'interaction')

# coloring edges depending on same sex/different sex interactions
'''for key in color_mapedges:
    if (color_mapsex[key[0]]) != 'Unknown' or (color_mapsex[key[1]]) != 'Unknown':
        if color_mapsex[key[0]] != color_mapsex[key[1]]:
            color_mapedges[key] = 'orange'
        elif color_mapsex[key[0]] == color_mapsex[key[1]]:
            color_mapedges[key] = 'navy'
        else:
        	color_mapedges[key] = 'black'  # intercation with unknown
    else:
        color_mapedges[key] = 'black'  # intercation with unknown'''

# coloring edges depending on same class/different class interactions
for key in color_mapedges:
	if color_mapclass[key[0]][:2] != color_mapclass[key[1]][:2]: #verify if they are in the same speciality
		color_mapedges[key]= 'darkslateblue' 
	elif color_mapclass[key[0]][:2] == color_mapclass[key[1]][:2] :
		color_mapedges[key] = 'darkred'
	else : 
		color_mapedges[key] = 'black' #edges black if it does correspond to the two cases
		
	
# coloring nodes depending on the sex attributes
for key in color_mapsex:
    if color_mapsex[key] == 'F':
        color_mapsex[key] = 'green'
    elif color_mapsex[key] == 'M':
        color_mapsex[key] = 'red'
    else:
        color_mapsex[key] = 'blue'

# coloring nodes depending on the specialitiy attribute
for key in color_mapclass:
    if color_mapclass[key] == '2BIO1':
        color_mapclass[key] = 'lightgreen'
    elif color_mapclass[key] == '2BIO2':
        color_mapclass[key] = 'lightgreen'
    elif color_mapclass[key] == '2BIO3':
        color_mapclass[key] = 'lightgreen'
    elif color_mapclass[key] == 'MP*1':
        color_mapclass[key] = 'lightblue'
    elif color_mapclass[key] == 'MP*2':
        color_mapclass[key] = 'lightblue'
    elif color_mapclass[key] == 'MP':
        color_mapclass[key] = 'lightblue'
    elif color_mapclass[key] == 'PSI*':
        color_mapclass[key] = 'lightpink'
    elif color_mapclass[key] == 'PC':
        color_mapclass[key] = 'lightyellow'
    elif color_mapclass[key] == 'PC*':
        color_mapclass[key] = 'lightyellow'

# affecting the nodes color in the good ordere
class_colors = [color_mapclass.get(node) for node in G.nodes()]
inter_colors = [color_mapedges.get(ed) for ed in G.edges()]
sex_colors = [color_mapsex.get(node) for node in G.nodes()]


if __name__ == "__main__":

    nx.draw(G, with_labels=True, node_color=class_colors,edge_color=inter_colors)
    plt.show()
