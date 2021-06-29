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


## Credit Risk Managment
In credit risk management the transition matrix represents the likelihood of the future evolution of the ratings. The transition matrix will describe the probabilities that a certain company, country, etc. will either remain in their current state, or transition into a new state. The following probability transition matrix has been taken from the data of the credit rating agencies such as Standard & Poor, Moody’s and Fitch. Where the table reports for AAA, AA, A, BBB, BB, B, CCC and D bonds in the financial and industrial sectors.

![alt text](/State_Transition_Diagrams/Credit_Risk_Management.JPG)

## Predicting Market Share
These are Hypothetical situation, based on real scenerios

There are three Product based companies Mcaffe , Quickheal and Kaspersky selling antivirus software. These three companies were doing great but a new company Avira entered the market.

They designed and implemented marketing strategies that shows it will attract

2% of customer base of MCAFFE

6% of Quickheal users

5% of KASPERSKEY users And will retain 97% of its users.

Based on this, we will develop a model to predict market share.

![alt text](/State_Transition_Diagrams/Market_Share.JPG)

## Predicting Market Trend
So we basically have three types of trend in a market. These are
- Bull markets: periods of time where prices generally are rising, due to the actors having optimistic hopes of the future.
- Bear markets: periods of time where prices generally are declining, due to the actors having a pessimistic view of the future.
- Stagnant markets : periods of time where the market is characterized by neither a decline nor rise in general prices

Consider a fair market environment lets suppose a market condition.

After a week characterized of a bull market trend there is a 90% chance that another bullish week will follow. Additionally, there is a 7.5% chance that the bull week instead will be followed by a bearish one, or a 2.5% chance that it will be a stagnant one. After a bearish week there’s an 80% chance that the upcoming week also will be bearish, and so on. 

![alt text](/State_Transition_Diagrams/Market_Trend.JPG)

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
