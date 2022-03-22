'''ML Cluster Algebras (from data generated using ExchangeGraphs.ipynb)'''
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from math import floor
from itertools import chain
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.decomposition import PCA   #...PCA was performed on the data but results not included in final script

#%% #Import data
with open('./TensorData/A4D4_d[4, 4].txt','r') as file: #...set the datapath and filename to the data one wishes to investigate
    data = literal_eval(file.read())
del(file)

print('Class frequencies: '+str(list(map(len,data)))+'\nTensor length: '+str(len(data[0][0])))
    
#%% #Pre-process data, and generate fake data if required
#Remove exchange matrix info from the tensors if desired
EM_check = True       #...if False will remove the exchange matrix information from the data
number_variables = 4  #...manually input the algebra rank (i.e. number of variables in the clusters / EM dimension size)
if not EM_check:
    data = [[clust[number_variables*number_variables:] for clust in class_set] for class_set in data]

#If only one class imported, generate a fake class for the real vs fake investigation
if len(data) == 1:
    all_entries = Counter(chain(*data[0]))  #...compute frequency distribution of all entries across all clusters in the real data
    data.append([])                         #...add data list for fake data
    for fake_tensor in range(len(data[0])):
        #Generate a fake tensor (same length as real data tensors) where each entry is drawn from the frequency distribution of true entries in the real data
        new_fake = list(np.random.choice(list(all_entries.keys()),len(data[0][0]),p=np.array(list(all_entries.values()))/sum(all_entries.values())))
        if new_fake in data[0]: print('repeated entry') #...make note where a true seed is accidentally generated (note this never occurs)
        data[1].append(new_fake)
    print('Fake data generated and added')
    if len(all_entries.values()) < 50: print('...with frequency distribution for entries:\n'+str(all_entries))

#########################################################################################################
#%% #Set-up data for ML
#Select number of k-fold cross-validations to perform (k = 5 => 80(train) : 20(test) splits approx.)
k = 5    

#Separate inputs (X) and outputs (Y)
X = np.array([clust for class_set in data for clust in class_set]) 
Y = np.array([i for i in range(len(data)) for clust in data[i]]) 

#Zip input and output data together
data_size = len(X)
ML_data = [[X[index],Y[index]] for index in range(data_size)]

#Shuffle data ordering
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #...number of datapoints in each validation split

#Define data lists, each with k sublists witht he relevant data for that cross-validation run
Train_inputs, Train_outputs, Test_inputs, Test_outputs = [], [], [], []
for i in range(k):
    Train_inputs.append([datapoint[0] for datapoint in ML_data[:i*s]]+[datapoint[0] for datapoint in ML_data[(i+1)*s:]])
    Train_outputs.append([datapoint[1] for datapoint in ML_data[:i*s]]+[datapoint[1] for datapoint in ML_data[(i+1)*s:]])
    Test_inputs.append([datapoint[0] for datapoint in ML_data[i*s:(i+1)*s]])
    Test_outputs.append([datapoint[1] for datapoint in ML_data[i*s:(i+1)*s]])

del(ML_data,i) #...zipped list no longer needed

#%% #Train & test NN
Acc_list, MCC_list, CM_list = [], [], [] #...lists of ML performance measures to average over the k cross-validation runs

#Run cross-validation loops
for i in range(k):
    #Intialise and train NN classifier on the train data
    nn_clf = MLPClassifier((256,256,256),activation='relu',solver='adam',batch_size=32,max_iter=50,verbose=False) #...can edit the NN hyperparameters here if one wishes
    nn_clf.fit(Train_inputs[i], Train_outputs[i]) 
    
    #Calculate predictions with NN classifier on the test data
    Test_pred = nn_clf.predict(Test_inputs[i])
    CM_list.append(CM(Test_outputs[i],Test_pred,normalize='all')) #...row is true label, column predicted label
    Acc_list.append(np.sum(Test_pred == Test_outputs[i])/len(Test_outputs[i]))
    MCC_list.append(MCC(Test_outputs[i],Test_pred))

print('\nNN Classifier:\nAccuracy:\t'+str(sum(Acc_list)/k)+' +- '+str(np.std(Acc_list)/np.sqrt(k))+'\nMCC:\t\t\t'+str(sum(MCC_list)/k)+' +- '+str(np.std(MCC_list)/np.sqrt(k)))
#print('\nConfusion-Matrix:\n'+str([np.matrix(cm) for cm in CM_list]))

#########################################################################################################
#%% #Additional Data analysis
#PCA
pca = PCA(n_components=0.999)
pca.fit(X)
pcad_data = [pca.transform(np.array(class_set)) for class_set in data]
labels = ['F4','I1']
#Output PCA information
covar_matrix = pca.get_covariance()
print('Covariance Matrix: '+str(covar_matrix)+'\n\nEigenvalues: '+str(pca.explained_variance_)+'\nExplained Variance ratio: '+str(pca.explained_variance_ratio_)+' (i.e. normalised eigenvalues)\nEigenvectors: '+str(pca.components_)) #...note components gives rows as eigenvectors
#Plot 2d PCA
plt.figure('2d Data PCA')
#plt.title()
for class_idx in range(len(data)):
    plt.scatter(pcad_data[1-class_idx][:,0],pcad_data[1-class_idx][:,1],alpha=0.3,label=labels[1-class_idx])
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
#plt.xlim(-19.6,-19.1)
#plt.ylim(-4.5,0.5)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.tight_layout()
#plt.savefig('PCA_Clust.pdf')

#%% #Sparsity information
sparsedata = X #pca.components_ #... can change to sparsity analysis of PCA components (note sparsity reduces significantly with PCA, as expected)
count=[]
for c in sparsedata:
    if len(c) != len(sparsedata[0]): print('inconsistent lengths!')
    count.append(0)
    for i in c:
        if i != 0: count[-1] += 1
print('Avg # of non-zero entries:',np.mean(count),'\nAvg proportion:',np.mean(count)/len(sparsedata[0]))

