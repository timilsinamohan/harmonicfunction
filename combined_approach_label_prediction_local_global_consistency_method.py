__author__ = 'mohan'
from scipy.sparse import lil_matrix
from rescal import rescal_als
from sklearn import metrics
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.linalg import inv
from sklearn.metrics import precision_recall_curve, auc
from scipy import linalg
import random
from scipy.sparse.linalg import spsolve
np.set_printoptions(precision=3,suppress=True)
from scipy.sparse import dok_matrix, dia_matrix, identity
import numpy as np
import scipy as sp
import scipy.linalg
from sklearn.grid_search import GridSearchCV

random.seed(200)
def create_tensors():

    ##binding drugs function and genes by creating dictionary ###
    ent = set()
    cnt = 0
    for tumor_nodes in tumor_nodes_col:
        ent.add(tumor_nodes)
        ent.add(gene_nodes_col[cnt])
        cnt+=1

    cnt = 0
    for drug_nodes in drug_nodes_col:
        ent.add(drug_nodes)
        ent.add(drug_gene_nodes_col[cnt])
        cnt+=1

    ent = list(ent)
    ##create dictionary of nodes##
    nodes_dict ={}
    cnt = 0
    for n in ent:
        nodes_dict[n] = cnt
        cnt+=1

   
    e = len(ent)
    m = 2
    X = [lil_matrix((e,e)) for i in range(m)]

    #print tumor_nodes_col
    tumor_gene_edgelist = zip(tumor_nodes_col,gene_nodes_col)
    
    #### tumor gene bipartite graph nodes and edges##
    B = nx.Graph()
    B.add_nodes_from(tumor_nodes_col,bipartite = 0)
    B.add_nodes_from(gene_nodes_col,bipartite = 1)
    B.add_edges_from(tumor_gene_edgelist)
    top_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
    Tumor_gene_Matrix = nx.algorithms.bipartite.biadjacency_matrix(B,row_order=top_nodes)
   
    drug_gene_edgelist = zip(drug_gene_nodes_col,drug_nodes_col)
    
    #### drug gene bipartite graph nodes and edges##
    B = nx.Graph()
    B.add_nodes_from(drug_gene_nodes_col,bipartite = 0)
    B.add_nodes_from(drug_nodes_col,bipartite = 1)
    B.add_edges_from(drug_gene_edgelist)
    top_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
    drug_gene_Matrix = nx.algorithms.bipartite.biadjacency_matrix(B,row_order=top_nodes)
    genes_only_nodes= set()
    genes_name_only = set()
    tumor_only = set()

    ###creating hasGene networks####
    cnt_has_gene = 0
    for u,v in tumor_gene_edgelist:
        X[0][nodes_dict.get(u),nodes_dict.get(v)] = 1
        genes_only_nodes.add(nodes_dict.get(v))
        genes_name_only.add(v)
        tumor_only.add(nodes_dict.get(u))
        cnt_has_gene+=1
    
    
     ###creating drugGene networks####
    cnt_drug_gene_action = 0
    for u,v in drug_gene_edgelist:
        X[1][nodes_dict.get(v),nodes_dict.get(u)] = 1
        genes_only_nodes.add(nodes_dict.get(u))
        cnt_drug_gene_action += 1
      

    #print "created tumor gene tensor"
    gene_index = list(genes_only_nodes)
    gene_names = list(genes_name_only)
 
    return X,gene_names,gene_index,nodes_dict



def ground_truth_node_labels(gene_names):
    drug_function_edgelist = zip(drug_target_functions,drug_gene_nodes_col)
    labels= {}

    for gene_name,func in drug_function_edgelist:
        labels.setdefault(gene_name, []).append(func)

    G = nx.DiGraph(labels)
    B = nx.Graph()
    lab=[]
    genes=[]
    edges =[]
    for u,v in G.edges:
        lab.append(u)
        genes.append(v)
        edges.append([u,v])

    B.add_nodes_from(lab,bipartite = 0)
    B.add_nodes_from(genes,bipartite = 1)
    B.add_edges_from(edges)
   
    Ground_Truth_Matrix = nx.algorithms.bipartite.biadjacency_matrix(B,row_order= labels.keys(),
                                                                     column_order=gene_names)

    return Ground_Truth_Matrix,labels.keys()



def gene_interaction_network(network,gene_nodes):
    G = nx.read_edgelist(network, delimiter= " ",
                         nodetype=str,
                         data=(('weight',float),))
    
    Gene_Gene_Adj_mat = nx.adjacency_matrix(G, nodelist=gene_nodes)
    GG = nx.from_scipy_sparse_matrix(Gene_Gene_Adj_mat)
    return GG


def compute_propagation(order,idx_train,labels,emb,exp):
    ###here we need to get optimum k####
    k_range = range(1,12)
    param_grid = dict(n_neighbors =k_range)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid, cv = 10, scoring = "accuracy")
    grid.fit(gene_feature,labels)
    #print grid.best_params_,grid.best_params_['n_neighbors']
    GF = kneighbors_graph(gene_feature,grid.best_params_['n_neighbors'], mode='connectivity',include_self=False)
    G = nx.from_numpy_matrix(GF.A)
    nds = range(G.number_of_nodes())
    #print nds
    #print "nx.info embeddings:", nx.info(G)
    Laplacian_matrtix = nx.laplacian_matrix(G, nodelist= nds, weight='weight')
    L_exp = nx.laplacian_matrix(get_network, nodelist = nds, weight='weight')

    ####harmonic part####
    y = labels.copy
    I = identity(G.number_of_nodes())
    lamb = 1.0
    Laplacian_matrtix = np.add(Laplacian_matrtix*emb, L_exp*exp)
    fu = spsolve((I + Laplacian_matrtix*lamb), labels)
    
    return fu
    

def get_embeddings(Tenc):
    feature_vec,R, fit, itr, exectimes = rescal_als(Tenc,250,
                                                    init='nvecs',
                                                    conv=1e-2,
                                                    lambda_A=0.1,
                                                    lambda_R= 0.1 
                                                   )
    
    return feature_vec
    
if __name__ == '__main__':
    # load data
    df1 = pd.read_csv("data/tumor_gene_hasGene_data.csv",sep = "\t",names = ["tumor", "gene"])
    df2 = pd.read_csv("data/drugs_gene_copy.txt",sep = "\t",names = ["gene", "drugs","function"])

    tumor_nodes_col = df1["tumor"]
    gene_nodes_col =df1["gene"]

    drug_nodes_col = df2["drugs"]
    drug_gene_nodes_col =df2["gene"]
    drug_target_functions = df2["function"]
    
    ##creating tensors for factorization for drug-gene-tumor multi-graph###

    Tenc, gene_names, gene_idx, node_dictionary =  create_tensors() 
    
    ###getting gene embeddings#######
    
    features_vector_for_genes = get_embeddings(Tenc) 
    gene_feature = features_vector_for_genes[gene_idx,:]
    #####################################################
    
    gene_names=[]
    for id in gene_idx:
        gene_names.append(node_dictionary.keys()[node_dictionary.values().index(id)])

    ground_truth,labels_keys = ground_truth_node_labels(gene_names)
    
    ### types of labels available "blocker","antagonist","agonist","activator","inhibitor","channel blocker",
    ## "binder"
   
    ##"Positive Labels:"###
    GT = ground_truth[labels_keys.index("blocker"),:].A[0]
   
    ##Negative Labels###
    idx = np.where(GT==0)
    GT[idx] = -1

    get_labels = []

    for i in GT:
        get_labels.append(i)
    get_labels = np.array(get_labels)

    # Do cross-validation
    FOLDS = 10
    roc_score = np.zeros(FOLDS)
    auc_pr_score = np.zeros(FOLDS)
    node_list = gene_names
    IDX = list(range(len(node_list)))
    
    random.shuffle(IDX)
    fsz = int(len(node_list) / FOLDS)
    print "fold size:",fsz

    #####Graph from real gene-gene interaction network###
    
    ###this is for gene-gene interaction network selection####
    ###types of network available 
    ##"coexp_scores.txt","cooccurence_scores.txt","experimental_scores.txt",
    ## "fusion_scores.txt", "neighborhood_scores.txt", "textmining_scores.txt","database_scores.txt,
    ## "combined_scores.txt""
    
    network = "data/combined_scores.txt"
        
    ##percentage of the labelled nodes###
    
    label_percentage = 0.5
    ##########################
    get_network = gene_interaction_network(network,gene_names)
    
    ########### To Enable only embedding Network make the flag EMB = 1 and EXP = 0 and vice versa##
    ### EMB = Embedding Network
    ### EXP = Combined Genetic Interaction Network
    EMB = 1
    EXP = 1
    ########################################
    
    np.random.seed(123)
    offset = 0
    for f in range(FOLDS):

        labels= get_labels.copy()

        idx_test = IDX[offset:offset + fsz]
        idx_train = np.setdiff1d(IDX, idx_test)
        random.shuffle(idx_train)
        idx_train = idx_train.tolist()
        labels[idx_test] = 0

        labeled_in_training_set = np.random.choice(idx_train,size = int(len(idx_train)*(1-label_percentage)), 
                                                   replace = False)
        labels[labeled_in_training_set] = 0   
        order = idx_train + idx_test
        total_score = compute_propagation(order,idx_train,labels,EMB,EXP)   
        roc_score[f] = roc_auc_score(GT[idx_test], total_score[idx_test])
        prec, recall, _ = precision_recall_curve(GT[idx_test], total_score[idx_test])
        auc_pr_score[f] = auc(recall,prec)
        offset += fsz

    print "AUC-ROC mean score after 10 FOld CV and standard deviation", roc_score.mean(),roc_score.std()
    print "AUC-PR mean score after 10 FOld CV and standard deviation",auc_pr_score.mean(),auc_pr_score.std()


