import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

P = np.array([[0.8,0.1,0.1],
              [0.2,0.6,0.2], 
              [0.1,0.2,0.7]])

state=np.array([[1.0,0.0,0.0]]) 
stateHist=state 
dfStateHist=pd.DataFrame(state) 
distr_hist = [[0,0,0]] 

for x in range(100): 
    state=np.dot(state,P)
    stateHist=np.append(stateHist,state,axis=0) 
    dfDistrHist = pd.DataFrame(stateHist) 

plt.figure(figsize=(10,5)) 
print(dfDistrHist) 
sns.lineplot(data=dfDistrHist)