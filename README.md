# Application of Markov Chain in Finance - Credit Risk Modeling using Python
Application of Markov Chain in Finance using Python and ML Libraries like numpy, pandas, seaborn etc. The purpose of this project is to develop an understanding of the underlying Markov Chains and then use the concepts to take on the financial problems that can be solved using applications of Markov Chain. I have taken on four different industrial examples of Markov chain that are really important in todays modern time. These include Brand loyalty, Predicting Market Share, Market Trend and Credit Risk Management. 

## Pre - Requisites
- Markov Chains
- Transition Probabilities
- States
- Properties

## Brand Loyalty
Brand loyalty is the tendency of consumers to continuously purchase one brand's products over another. Consumer behaviour patterns demonstrate that consumers will continue to buy products from a company that has fostered a trusting relationship.

I have taken a hypothetical situation based over some probabilities which I'll be putting in the form of a question and then implement using python.

There are three network based companies which sell router based internet connection to customers homes. These companies are here quite some time now and customers also switches from company to other company.

Data shows that 

10% of Hathway customers will switch to ADN and 10% to Excitel. 

20% of ADN customers will switch to Hathway and 20% to Excitel. 

10% of Excitel customers will switch to Hathway and 20% to ADN .

After significantly long time we will be finding the loyalty % that customers will keep loyal to one company.

![alt text](/State_Transition_Diagrams/Brand_Loyality.JPG)

### Script
```python
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
```

### Output
```
            0         1         2
0    1.000000  0.000000  0.000000
1    0.800000  0.100000  0.100000
2    0.670000  0.160000  0.170000
3    0.585000  0.197000  0.218000
4    0.529200  0.220300  0.250500
..        ...       ...       ...
96   0.421053  0.263158  0.315789
97   0.421053  0.263158  0.315789
98   0.421053  0.263158  0.315789
99   0.421053  0.263158  0.315789
100  0.421053  0.263158  0.315789
```

![alt text](/Graphs/Brand_Loyality.JPG)

## Credit Risk Managment
In credit risk management the transition matrix represents the likelihood of the future evolution of the ratings. The transition matrix will describe the probabilities that a certain company, country, etc. will either remain in their current state, or transition into a new state. The following probability transition matrix has been taken from the data of the credit rating agencies such as Standard & Poor, Moody’s and Fitch. Where the table reports for AAA, AA, A, BBB, BB, B, CCC and D bonds in the financial and industrial sectors.

![alt text](/State_Transition_Diagrams/Credit_Risk_Management.JPG)

### Script
```python
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

P = np.array([[0.9193, 0.0746, 0.0048, 0.0008, 0.0004, 0.0000, 0.0000, 0.0000],
              [0.6400, 0.9181, 0.0676, 0.0060, 0.0006, 0.0012, 0.0003, 0.0000], 
              [0.0700, 0.0227, 0.9169, 0.0512, 0.0056, 0.0025, 0.0001, 0.0004], 
              [0.0400, 0.0270, 0.0556, 0.8788, 0.0483, 0.0102, 0.0017, 0.0024], 
              [0.0400, 0.0010, 0.0061, 0.0775, 0.8148, 0.0790, 0.0111, 0.0101], 
              [0.0000, 0.0010, 0.0028, 0.0046, 0.0695, 0.8280, 0.0396, 0.0545], 
              [0.1900, 0.0000, 0.0037, 0.0075, 0.0243, 0.1213, 0.6045, 0.2369], 
              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]])

X = pd.DataFrame(P, columns=[ 'AAA', 'AA', 'A', 'BBB', 'BB' , 'B' , 'CCC' , 'D' ],
                            index= [ 'AAA' , 'AA' , 'A' , ' BBB' , ' BB' , ' B' , 'CCC ' , 'D' ])

X

num = int(input("Enter Years: "))
T = np.power(P,num)
H = pd.DataFrame(T, columns=[ 'AAA', 'AA', 'A', 'BBB', 'BB' , 'B' , 'CCC' , 'D' ],
                            index= [ 'AAA' , 'AA' , 'A' , ' BBB' , ' BB' , ' B' , 'CCC ' , 'D' ])

H
```

### Output
![alt text](/Graphs/Credit_Risk_Managment.JPG)

## Predicting Market Share
These are Hypothetical situation, based on real scenerios

There are three Product based companies Mcaffe , Quickheal and Kaspersky selling antivirus software. These three companies were doing great but a new company Avira entered the market.

They designed and implemented marketing strategies that shows it will attract

2% of customer base of MCAFFE

6% of Quickheal users

5% of KASPERSKEY users And will retain 97% of its users.

Based on this, we will develop a model to predict market share.

![alt text](/State_Transition_Diagrams/Market_Share.JPG)

### Script
```python
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
```

### Output
```
            0         1         2         3
0    0.400000  0.320000  0.180000  0.100000
1    0.382200  0.313400  0.170200  0.134200
2    0.365772  0.306986  0.161478  0.165764
3    0.350607  0.300769  0.153715  0.194908
4    0.336605  0.294759  0.146807  0.221830
..        ...       ...       ...       ...
96   0.161254  0.179236  0.090910  0.568599
97   0.161235  0.179211  0.090910  0.568643
98   0.161217  0.179188  0.090910  0.568685
99   0.161200  0.179166  0.090910  0.568723
100  0.161185  0.179146  0.090910  0.568760
```

![alt text](/Graphs/Market_Share.JPG)

## Predicting Market Trend
So we basically have three types of trend in a market. These are
- Bull markets: periods of time where prices generally are rising, due to the actors having optimistic hopes of the future.
- Bear markets: periods of time where prices generally are declining, due to the actors having a pessimistic view of the future.
- Stagnant markets : periods of time where the market is characterized by neither a decline nor rise in general prices

Consider a fair market environment lets suppose a market condition.

After a week characterized of a bull market trend there is a 90% chance that another bullish week will follow. Additionally, there is a 7.5% chance that the bull week instead will be followed by a bearish one, or a 2.5% chance that it will be a stagnant one. After a bearish week there’s an 80% chance that the upcoming week also will be bearish, and so on. 

![alt text](/State_Transition_Diagrams/Market_Trend.JPG)

### Script
```python
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
```

### Output
```
           0         1         2
0    0.00000  0.000000  1.000000
1    0.25000  0.250000  0.500000
2    0.38750  0.343750  0.268750
3    0.46750  0.371250  0.161250
4    0.51675  0.372375  0.110875
..       ...       ...       ...
96   0.62500  0.312500  0.062500
97   0.62500  0.312500  0.062500
98   0.62500  0.312500  0.062500
99   0.62500  0.312500  0.062500
100  0.62500  0.312500  0.062500
```

![alt text](/Graphs/Market_Trend.JPG)

## Conclusion
In **Brand Loyalty problem**, according to our solution
- 42.1% of Hathway customers will remain loyal to them 
- 26.3% of ADN customers will remain loyal to them 
- 31.5% of Excitel customers will remain loyal to them

In predicting **Market Share**, according to our solution
- Mcaffe will attain 16.1 % Market share 
- Quickheal will attain 17.9% Market share 
- Kasperskey will attain 9.09% Market share 
- Avira will attain 56.7% Market share

In predicting **Market Trend**, according to our solution
- There are 62.5% chances that market will be bullish  
- There are 31.25% chances that market will be bearish
- There are 62.5% chances that market will be stagnant

### Reference to an article which helped me
[Herman Scheepers article](https://towardsdatascience.com/markov-chain-analysis-and-simulation-using-python-4507cee0b06e)
