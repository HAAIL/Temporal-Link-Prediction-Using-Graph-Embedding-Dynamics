import time
import glob
import numpy as np
import networkx as nx
from dynamicgem.embedding.dynAERNN import DynAERNN
import pickle5 as pickle
#from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
#from dynamicgem.visualization import plot_dynamic_sbm_embedding


dim_emb = 100 #latent space dimension
lookback = 1
batchNumber = 32
graphPath = 'data/Generated'
graphlist = sorted(glob.glob(f'{graphPath}/*.net'))
graphs = []

for i in graphlist:
    
    graphs.append(pickle.load(open(i, "rb")))
print(len(graphs))
length = len(graphs)
embedding = DynAERNN(d=dim_emb,
                     beta=5,
                     n_prev_graphs=lookback,
                     nu1=1e-6,
                     nu2=1e-6,
                     n_aeunits=[500, 300],
                     n_lstmunits=[500, dim_emb],
                     rho=0.3,
                     n_iter=30,
                     xeta=1e-3,
                     n_batch=batchNumber,
                     modelfile=['enc_model_dynAERNN.json',
                                'GEL/dec_model_dynAERNN.json'],
                     weightfile=['GEL/enc_weights_dynAERNN.hdf5',
                                 'GEL/dec_weights_dynAERNN.hdf5'],
                     savefilesuffix="testing")

embs = []
t1 = time.clock()
for temp_var in range(lookback + 1, length + 1):
    emb, _ = embedding.learn_embeddings(graphs[:temp_var])
    print('emb type: ', type(emb))
    embs.append(emb)
embs = np.asarray(embs)
e = embs.round(decimals=3)

print(embedding._method_name + ':\n\tTraining time: %f' % (time.clock() - t1))
np.save(f'{graphPath}/embeddeds_dim{dim_emb}_batchNumber{batchNumber}_length{length}_lookback{lookback}.npy', embs)
