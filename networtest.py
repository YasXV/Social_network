import numpy as np 
import networkx as nx 
import pandas as pd 
#from random import randint 
#Sfrom random import normalvariate
import matplotlib.pyplot as plt 

#creation of my network G 
G=nx.Graph()

datatri=np.loadtxt("tij_tri.dat",dtype=int)#load des donnees sur les intercations triées(dans un unique array)
infos0=np.loadtxt("metadata_Thiers13.dat",usecols=0,dtype=int)#load des ids(depuis metadata)
infos1=np.loadtxt("metadata_Thiers13.dat",usecols=1,dtype='str')#load des classes
infos2=np.loadtxt("metadata_Thiers13.dat",usecols=2,dtype='str')#load des sexes 

infosdict={'id':infos0,'classe' : infos1, 'sex' : infos2}#création d'un dictionnaire à partir de mes colonnes load précédement 
infos = pd.DataFrame(infosdict)#construction d'un Dataframe à partir du dictionnaire précédent!


#ajout des nodes (donc tout les ids), ainsi que de leurs attributs associées (classe,sexe) 
for indice in range(int(len(infos['id']))):
   G.add_node(infos['id'][indice],classe=infos['classe'][indice],sex=infos['sex'][indice])

#ajout des links/edges (interactions inter id), ainsi que de leurs attributs associés (nbre d'interactions) 
for indice in range(int(len(datatri[:,1]))):
  G.add_edge(datatri[indice,0],datatri[indice,1],interaction=datatri[indice,2])



#Basic analysis 
#We have an undirected network (we consider the inter-students, we do not care of who is reaching out to who, as long as they are interacting), it is unsigned (indeed we do not knwo if those are bad/good intercations), it is weighted ( the weight of an interaction is the number of times intercated for 2 students), there's no self-loops.

N=nx.number_of_nodes(G)
E=nx.number_of_edges(G)
E_max=int((N*(N-1))/2)
rho=nx.density(G)
AV=(2*E)/N

#liste des degrees de toutes les nodes!----------------------------------------------------------------
a=list(G.degree())
deglist=[]
for i in range(len(a)):
  deglist.append(a[i][1])
 
 #liste des poids de tout les edges!------------------------------------------------------------------
dW=nx.get_edge_attributes(G,"interaction")
#print(dW)
weightlist=list(dW.values())


#degree fluctuations (<K²>-<k>) ---------------------------------------------------------------------
deg=np.array(deglist)#array of degrees 
V=(np.sum(deg**2)/N)-AV**2


#dict of nodes's strenghts ----------------------------------------
strenghts = {}
for i in infos['id']:
   s=0
   for n,p in G.edges(i):
      s=s+G[n][p]['interaction']
   strenghts[i] = s

   
#average link weight= sum of weights divided by the number of edges----------------  
AVlink=np.sum(weightlist)/E


#a scatter plot of the average link weight in the egonet of a node as a function of the node’s degree-------
scatter={}
for i in a :
   if (i[1]!=0):
     t=round(strenghts[i[0]]/i[1],5)
   else :
     t=0
   scatter[i[0]]=[i[1],t]

'''deg_avw=np.array(list(scatter.values()))
plt.plot(deg_avw[:,0],deg_avw[:,1],'o')
plt.hlines(AVlink,0,85,colors='green',label=f'average linkweight= {AVlink : .5f}')
plt.grid()
plt.xlabel("node's degree k")
plt.ylabel('average link weight in the egonet of a node')
#plt.title("average link weight in the egonet of node as a function of the node's degree")
plt.vlines(AV, 0, 200, colors='red', label='average degree =35.36778' )
plt.legend()
plt.savefig('Average-link_weight_func-nodes_degree')'''



#distribution of degress p(k)=Nk/N : probability that a node has a degree k-------------------------
p_k=np.zeros(np.max(deg)+1)
for i in range(0,np.max(deg)+1):
	p_k[i]=len(deg[deg==i])/N
	
'''plt.plot(np.arange(0,np.max(deg)+1),p_k,color='mediumpurple')
plt.xlabel('k')
plt.ylabel('p(k)')
plt.grid()
#plt.xscale('log')
#plt.yscale('log')
plt.savefig('Distribution_of_degrees_pk')'''



#distribution of nodes strenght p(s)--------------------------------------------------------    
strplot=list(strenghts.values())
'''hh=plt.hist(strplot,bins=20,range=(0,max(strplot)),color='skyblue')
plt.grid()
plt.xlabel('strenght')
plt.ylabel('occurence')
plt.title('Distribution of nodes strenght')
plt.savefig('Distribution_of_node_strenght_ps')'''
avstr=np.sum(strplot)/N

#distribution of betweeness centrality 
betw=nx.betweenness_centrality(G)
dB = list(betw.values())
avbetw=np.sum(dB)/N 

'''h=plt.hist(dB,bins=10,range=(0,np.max(dB)),color='skyblue')
plt.xlabel('betweeness centrality')
plt.ylabel('Occurence')
plt.title("Distribution  of betweeness centrality ")
#plt.yscale('log')
plt.grid()
plt.savefig("Distribution_of_betweeness_centrality")'''

#distribution of node's clustering
clust=nx.clustering(G)
dclust=list(clust.values())
avclust=np.sum(dclust)/len(dclust)
'''h=plt.hist(dclust,bins=10,range=(0,1),color='skyblue')
plt.xlabel('clustering')
plt.ylabel('Occurence')
plt.title("Distribution of node's clustering ")
#plt.yscale('log')
plt.grid()
plt.savefig('Distribution_of_node_clustering_pc')'''
   
#distribution of link weights-----------------------------------------------------------------------
'''h=plt.hist(weightlist,bins=30,range=(0,max(weightlist)),color='skyblue')
plt.xlabel('weight')
plt.ylabel('Occurence')
plt.title('Distribution of link weight')
plt.yscale('log')
plt.grid()
plt.savefig('Distribution_of_link_weight_pw')'''



#a scatter plot of the average degree of a node’s neighbours as a function of the node’s degree--------
scattneib={}
#"a" est list(G.degrees()), list de tuple : (node, degree de la node)
for i in a:
     d=0
     neib=list(G.neighbors(i[0]))
     taille=len(neib)
     if (taille!=0):
       for n in neib:
       	d=d+G.degree(n)
       d=d/taille
     scattneib[i[0]]=[i[1],round(d,5)]

'''deg_neib=np.array(list(scattneib.values()))
plt.plot(deg_neib[:,0],deg_neib[:,1],'o')
plt.grid()
plt.hlines(AV,0,85,colors='red',label=f'average degree= {AV : .5f}')
plt.ylim([0,np.max(deg_neib[:,0])])
plt.legend()
plt.xlabel("node's degree")
plt.ylabel("average degree of a node's neighbor")
#plt.title("average degree of a node’s neighbours as a function of the node’s degree")
plt.savefig('Average-degree_neighbours_func-nodes_degree')'''
     

#scatter plots of the betweenness bi and clustering c i of a node i as a function of its degree ki    
clustbetdeg={}
for i in a:
 clustbetdeg[i[0]]=[i[1],betw[i[0]],clust[i[0]]]
cbd=np.array(list(clustbetdeg.values()))
'''fig, axs = plt.subplots(2,1,figsize=(6, 7))
#fig.suptitle('Betweenness and clustering of a node as a function of its degree')
axs[0].scatter(cbd[:,0],cbd[:,1],label='betweness centralities',color='y')
axs[1].scatter(cbd[:,0],cbd[:,2],label='clustering',color='c')
axs[0].set(xlabel='degree', ylabel='b')
axs[1].set(xlabel='degree', ylabel='c')
axs[0].grid()
axs[1].grid()
#axs[0].xlabel('degree')
#axs[1].xlabel('degree')
axs[0].hlines(avbetw,0,85,colors='green',label=f'average betweeness centrality= {avbetw :.5f}')
axs[0].vlines(AV, 0, 0.030, colors='red', label=f'average degree = {AV : .5f}' )
axs[1].hlines(avclust,0,85,colors='green',label=f'average clustering= {avclust :.5f}')
axs[1].vlines(AV, 0, 1, colors='red', label=f'average degree = {AV : .5f}' )
axs[0].legend()
axs[1].legend()
plt.savefig('betweenness_clustering_node_function_degree')'''


#remove nodes that dont interact 
G.remove_node(2) 
G.remove_node(478)

#average shortest path and diameter
diameter=nx.diameter(G)
shortest=nx.average_shortest_path_length(G)

#we add the nodes back up again 
#G.add_node(2,classe='PC*',sex='M')
#G.add_node(478,classe='2BIO2',sex='F')

if __name__=="__main__": 
   
	#Basic analysis results-------------------------------------------------------------------------------------
	print("We have only one big component with size 327 , indeed we have 2 nodes out of 329 that don't interact with anyone") 
	print("Basic analysis : \n")
	print(f'Number of nodes :  {N}')
	print(f'Number of edges :  {E}')
	print(f'Maximum number of edges : {E_max}')
	print(f'Network density : {rho:.5f}')
	print(f'Average degree : {AV: .5f}, AVdeg/N={AV/N : .5f}<<1, so we have a quite sparse network.')
	print(f'Average linkweight : {AVlink: .5f}')
	print(f'Now we want the average degree calculeted trough the sum : {np.sum(deglist)/N: .5f} --> it gives us the same result, wich makes sense')
	print(f'Variance² of degrees : {V : .5f} >>1, we have large degrees fluctuations')
	print(f'Average clustering <c>: {avclust : .5f} ')
	print(f'Average shortest path lenght : {shortest:.5f}')
	print(f'Diameter of the Graph : {diameter}')
	print(f"Average node's strenght <s>: {avstr : .5f} ")
	print(f'Average betweeness centrality <b>: {avbetw : .5f} ')	
