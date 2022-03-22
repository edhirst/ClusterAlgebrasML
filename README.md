# ClusterAlgebrasML
Supervised machine learning techniques, and general network analysis methods are applied to Cluster Algebras and their exchange graphs.  

The *ExchangeGraphs.ipynb notebook details the function to generate the exchange graphs (built on the sage ClusterSeed() object):  
...as described in the script there is functionality to generate the exchange graphs, perform various network analyses and plot certain cycle embeddings, and also generate data (as seeds in a tensor format) for machine learning.  

The *ML.py script performs machine learning with dense feed-forward neural networks from the sci-kit learn package:  
...one must first ensure the filepath is correct for the investigation one wishes to perform, then cells can be run sequentially.  
...sample datasets for the investigations in the paper are available in the *TensorData directory.  

