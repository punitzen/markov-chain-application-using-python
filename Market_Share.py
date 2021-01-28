import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

P = np.array([[0.92,0.02,0.01,0.05],
              [0.03,0.94,0.01,0.02], 
              [0.02,0.02,0.9,0.06], 
              [0.01,0.01,0.01,0.97]])

state=np.array([[0.4,0.32,0.18,0.10]]) 
stateHist=state 
dfStateHist=pd.DataFrame(state) 
distr_hist = [[0,0,0,0]] 

for x in range(100): 
    state=np.dot(state,P) 
    stateHist=np.append(stateHist,state,axis=0) 
    dfDistrHist = pd.DataFrame(stateHist) 

plt.figure(figsize=(10,5)) 
print(dfDistrHist) 
sns.lineplot(data=dfDistrHist)