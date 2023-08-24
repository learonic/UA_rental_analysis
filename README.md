# UA rental analysis project

This project goal is to analyze apartment rental offers for 5 major cities in Ukraine and build ML model to predict price based on available information 
First version of analysis in rolled out, work in progress on V 2.0

### Project background
Looking at the apartment rental market in our country I wondered what are the key factors which impact rental prices. Does the location matter? Is the kitchen area impactful? What factors truly impacts rental price and can it be **modelled**? To improve my understanding, I came up with this project


### Brief summary of analysis  
#### Data sources
There are no publicly available dataset on Ukraine rental. Also, the market is quite dynamic and I thought that it would be good to have up to date offers as a source. Therefore, first step was scrapping some data. As the most famous platform www.olx.ua was selected 
##### Data summary
The data was collected for 2 months. Over 35k ads was collected during this time. 
After some data cleaning and EDA the portion of data was discarded. Also, as I am not interested in higher end level of property I excluded:
+ apartment with areas over 150 sqm
+ price-based outliers
+ luxurious apartments (having penthouse or vip in description)  
Final distribution of the apartments by price looks like this: 
![Apts by price](/other/Area_price.png)

#### Modeling
A family of ML models were built with pycaret library. Expectedly highest R<sup>2</sup> score was reached by lightgbm model. Unfortunately, even after tuning the score reach only ~0.64. Adding feature interaction/polynomial only lifted model R<sup>2</sup> by 0.01. Therefore I concluded that features taken into analysis are insufficient for the modeling and considered several feature steps to improve the model


### Project structure  
#### Main folder 
##### V 1.0 files
`scrap.ipynb` - This script scraps olx pages and updates `Data/rent_offers.csv` with newly found ads. Uses  
`cleaning.ipynb` -  This script cleans `Data/rent_offers.csv`, removing outliers anomalities and undesired data and stores result as `Data/rent_offers_clean.csv`  
`EDA.ipynb` - Here you can see exploration of the dataset.  
`model.ipynb` - Building of an ML model, selecting best, tuning hyperparameters and exploring feature importance  
  
##### V 2.0 files
`links.ipynb` - This script scraps olx search result pages and updates `Data/rent_links.csv`. Uses beautifullSoup  
`scrap_pages.ipynb` - This script scraps olx ad content and updates `rent_details.parquet`. Uses beautifullSoup  
`cleaning_fe` -  This script process ad tags and do some data cleaning. Takes `rent_details.parquet` as input. Work in progress.

#### Data subfolder
`rent_offers.csv` - raw file scrapped from olx.ua by v 1.0 spider  
`rent_offers_clean.csv` - cleaned file, used for modeling  
`rent_links.csv` - basic scraped information collected by v 2.0 spider. Does not contain full details, used for further scrapping  
`rent_details.parquet`  - full information collected by v 2.0 spider. Saved in parquet format in order to preserve underlying data structure

### Feature steps
1. Finishing `cleaning_fe.ipynb`. New features such as renovation type, district etc. to be added
2. Building new models based on v2 dataset. FF NN also to be implemented
3. Extract location and adding location based features to the model (distance from city center, district center, subway station)
4. Consider sentiment analysis on description
5. Consider analyzing images