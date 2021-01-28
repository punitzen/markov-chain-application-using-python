import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

# Three Market Trends - Bull, Bear and Stagnant 
P = np.array([[0.9,0.075,0.025],
              [0.15,0.8,0.05], 
              [0.25,0.25,0.5]])

state=np.array([[0.0,0.0,1.0]]) 
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