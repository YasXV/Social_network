# Social_network
##This is explaining the purpose of each python script

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##We have two .dat files : 
------>"metadata_Thiers13.dat"<--------, wich is a list of all the ids of the student in column 1, column 2 is their class, column 2 their gender. 
------>tij_Thiers13.dat"<-------, which is the list of all interactions that occured between 2 ids ( the first two columns) during five days, the 3rd colun is the C-time in seconds in which the interaction occured. An interaction is counted every 20 seconds. 

##2)FIRST SCRIPT -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------->triinitial.py<--------------- 
an sql program that counts the number of interactions between 2 ids detailed in the "tij_Thiers13.dat" file. 
The counted interactions are stocked in the file "tij_tri.dat"

##3)SECOND SCRIPT-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------->networtest.py<--------
This script will load the "tij_tri.dat" and "metadata_Thiers13.dat" files in order to generate our graph with the right attributes. It will then calculate basic properties/parameters of the network, such as the number of nodes, the density, various distributions (link weights, betweeness centralities...). All those informations could and will be imported onto the other scripts for them to be used. 

##4)THIRD SCRIPT -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------>graph_visu.py<---------
This script is focused on having different type of visualizations of the networks, by highlighting some attributes/links using color codes. 


##5)FORTH SCRIPT -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
---------->commu.py<----------
This script is focused on creating both of the contact matrix (for classes and gender attributes) and display/save them into a figure. 


##6)SIXTH SCRIPT-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
------->null_models.py<---------
This script is focused on comparing the properties of a graph G to random networks with N (number of nodes) and E (number of edges fixed). We builded the function "caracGraph_ER(G, number_loops)" it returns a figure with various distributions and compares it to the properties of the graph G.


##7)SEVENTH SCRIPT--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------->weigh_distri.py<--------
This script is focused on weigh randomisation using the function "weight_distri(G, weight_tab, number_loops)", it generates the difference between the weighted contact matrix of our reference network G and the average weighted contact matrix of our "number_loops" networks. "Weight_tab" is the list of degree that is conserved and distributated randomly among all nodes. 

Good luck *_*
