# Financial Project
### Project Description
-	Language: Python
-	Main Purpose: Build portfolio
-	Features: Risk diversification, Asset allocation
-	Packages Used: Pandas yfinance seaborn

## Features
1. Take in n markets
2. Compute mean return, stdev(risk), correlation (diversification)
3. Generate portfolio to maximise risk-return ratio
4. Create asset allocation plan- balance risk free asset and mve portfolio
5. End goal: make capital allocation plot

## Steps
1. Extract data of markets (risky assets: small cap, large cap, japan, uk etc)
2. Data wrangling by finding for the asset:
  a.	Find Daily Returns
  b.	Mean, stdev and correlation of each individual market’s return
3. Apply Prescriptive analytics:
  -	Find minimum variance portfolio
  -	Find maximum sharpe ratio portfolio
4.  Use The 2 portfolios to plot efficient portfolio frontier
  -	Include all the individual assets to show that the frontier is better than each individual asset
  -	Mark out the mve point and the minimum variance point
5.   Create a separate capital allocation line plot
  -	Use the MVE portfolio and the risk-free asset’s return
  -	Plot a line that cuts through both of them
6.  (Bonus) Given Risk aversion(input), provide weightage (output):
  -	Given a guy’s risk aversion value, give him a recommended portfolio weightage
