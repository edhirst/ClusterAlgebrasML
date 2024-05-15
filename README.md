# ClusterAlgebrasML
Supervised machine learning techniques, and general network analysis methods are applied to Cluster Algebras and their exchange graphs.  

The `ExchangeGraphs.ipynb` notebook details the function to generate the exchange graphs; built on the sage ClusterSeed object, please run with a sage kernel ([sagemath.org](https://www.sagemath.org/)), or via their online cell ([CoCalc](https://cocalc.com/)):  
~ As described in the script there is functionality to generate the exchange graphs, perform various network analyses and plot certain cycle embeddings, and also generate data (as seeds in a tensor format) for machine learning.  

The `ML.py` script performs machine learning with dense feed-forward neural networks from the sci-kit learn package:  
~ One must first ensure the filepath is correct for the investigation one wishes to perform, then cells can be run sequentially.  
~ Sample datasets for the investigations in the paper are available in the `TensorData` directory (to be unzipped before using).  

# BibTeX Citation
``` 
@article{Dechant:2022ccf,
    author = "Dechant, Pierre-Philippe and He, Yang-Hui and Heyes, Elli and Hirst, Edward",
    title = "{Cluster Algebras: Network Science and Machine Learning}",
    eprint = "2203.13847",
    archivePrefix = "arXiv",
    primaryClass = "math.CO",
    reportNumber = "LIMS-2022-011",
    doi = "10.1016/j.jaca.2023.100008",
    journal = "J. Comput. Algebra",
    volume = "8",
    year = "2023"
}
```
