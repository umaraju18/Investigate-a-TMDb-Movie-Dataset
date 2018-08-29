
# coding: utf-8

# # Project: Investigate a TMDb Movie Dataset 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis (EDA)</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#limitations">Limitations</a></li>
#     
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > I selected TMDb movie dataset for this Data Analyst project. The original data from Kaggle was cleaned and considered for this analysis. AS per TMDb, it is a community built movie and TV database. This data set contains information about 10,000 movies, including user ratings and revenue data. 
# 
# ### Details about Dataset
# 
# > TMDb movie dataset mainly contains attributes related to measure the successful movie and properties associated with success of movie. The attributes to measure the successful of movie are popularity,revenue and vote average. The metrics associated with the movie success are budget,cast,director,tagline,runtime,genres,production company and release date.
# 
# > As per [TMDb](https://developers.themoviedb.org/3/getting-started/popularity), the popularity is an cumulative factor considering Number of votes for the day, views for the day, users who marked it as a "favourite" for the day, users who added it to their "watchlist" for the day, release date, total votes and previous days score.
# 
# > The final two columns ending with “_adj” show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.
# 
# > After eyeing through the dataset, the following questions came into my mind. In this report,the answers for that questions were explored through systematic data analysis process.
# 
# ### Research Questions to Explore
# 
# ### Research Part 1: General Exploration
# 
# <li><a href="#Analysis 1">Analysis 1: Budget/Revenue Trend over the Period of Time</a></li>
# <li><a href="#Analysis 2">Analysis 2: Popularity Trend over the Period of Time</a></li>
# <li><a href="#Analysis 3">Analysis 3: Average Vote Trend over the Period of Time</a></li>
# <li><a href="#Analysis 4">Analysis 4: Number of Movies Released Over the Time</a></li>
# <li><a href="#Analysis 5">Analysis 5: Runtime Distribution of Movies</a></li>
# 
# <a id='research part 2a'></a>
# ### Research Part 2a (Quantitative Analyses): Properties associated with High Revenue Movies
# 
# <li><a href="#Question 1a">Research Question 1 (Which Revenue Level receives the highest popularity?)</a></li>
# <li><a href="#Question 2a">Research Question 2 (Which Budget Level Receives the Highest Popularity?)</a></li>
# <li><a href="#Question 3a">Research Question 3 (Which Runtime Level Receives the Highest Popularity?</a></li>
# <li><a href="#Question 4a">Research Question 4 (Which Runtime Level Receives the Highest Avergae Voting?</a></li>
# 
# <a id='research part 2b'></a>
# ### Research Part 2b (Categorical Analyses): Properties associated with High Revenue Movies 
# 
# <li><a href="#Question 1b">Research Question 1 (Which Movies are Top 10 highest popularity?)</a></li>
# <li><a href="#Question 2b">Research Question 2: (What kind of Genres are top in High Revenue movies?)</a></li>
# <li><a href="#Question 3b">Research Question3: (Which Production Companies are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 4b">Research Question 4: (Which Actors are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 5b">Research Question 5: (Which directors are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 6b">Research Question 6: (Which keywords are top in Best Revenue movies?)¶</a></li>
# <li><a href="#Question 7b">Research Question 7: (Which Genres are top during 60s and 2Ks Best Revenue movies?)¶</a></li>
# <li><a href="#Question 8b">Research Question 8: (Which Keywords are top during 60s and 2Ks Best Revenue movies?)¶</a></li>
# 
# 

# In[327]:


# importing all packages related to this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### Loading Dataset
# 
# > First, we will read the data set for proceeding to investigate
# 

# In[328]:


# Loading data and print out a few lines. 
df = pd.read_csv('tmdb-movies.csv')
df.head()


# > The above dataset looks perfect interms of formatting and column index names. There are some odd characters in the ‘cast’ column. I am going to leave them as it is. It also shows 21 columns in the dataset

# ### Exploring Dataset 

# In[329]:


#   shape of the dataset
df.shape


# > The dataset contains total of 10866 rows and 21 columns

# In[330]:


# exploring data
df.info()


# > The above information indicates types and number of non-null for all column index. After exploring the dataset, From the table above, there are totally 10866 entries and total 21 columns. And there exists some null value in the cast, director, overview and genres columns. But some columns are with a lot of null value rows like homepage, tagline, keywords and production_companies, especially the homepage and tagline columns are not required for this analysis,so I decided to drop homepage, tagline and keywords along with imdb_id. Column indexes such as cast, director and genres  are having few missing nonnull values. I decided to drop small quantity of null values in columns cast, director and genres

# In[331]:


df.describe()


# > The above table shows the descriptive statistics information of the dataset. The popularity shows the outlier information. AS per [TMDb](https://developers.themoviedb.org/3/getting-started/popularity), the popularity is the cumulative number of favourites and number of watched list etc, since it has no upperbond, I decided to retain the original data. This table shows lot of "0" for budget, revenue, runtime, budget_adj and revenue_adj. Are those movies not released? But there is no minimum zero value in release year. So, I assume these values are missing values and not "real" values. In order to confirm this, I decided to find how many zeroes in those column indexes.

# In[332]:


# finding how many zeroes in budget

df_budget_zero = df['budget'].value_counts()
df_budget_zero.head(2)


# In[333]:


# filtering the zero in budget
df_budget_zero = df.query('budget == 0')
df_budget_zero.head()


# > In order to confirm zeroes in budget, i quickly checked Mr. Holmes budget in the internet [wikipedia](https://en.wikipedia.org/wiki/Mr._Holmes#cite_note-2) and there is information for budget. So, I decided to assume missing values for all zero values in budget column and replace with "NaN"

# In[334]:


# finding how many zeroes in revenue
df_revenue_zero = df['revenue'].value_counts()
df_revenue_zero.head(2)


# In[335]:


# filtering the zero in revenue
df_revenue_zero = df.query('revenue == 0')
df_revenue_zero.head()


# > Similarly in order to confirm zeroes in revenue, i quickly checked wild card budget in the internet [wikipedia](https://en.wikipedia.org/wiki/Wild_Card_(2015_film)) and there is information for revenue. So, I decided to assume missing values for all zero values in budget column and replace with "NaN".

# In[336]:


#count zero values in runtime data using groupby
df_runtime_count =  df.groupby('runtime').count()
df_runtime_count.head(2)


# In[337]:


# filtering the zero in revenue
df_runtime_zero = df.query('runtime == 0')
df_runtime_zero.head()


# > The above table shows 31 rows of runtime has zero values. Because of small number i decided to drop zero values in runtime

# ### Data Cleaning
# 
# #### Summary of Actions
# 
# > Drop unnecessary columns: homepage, tagline, imdb_id, overview, budget_adj, revenue_adj.
# 
# > Drop duplicates
# 
# > Drop null values with small quantity of null values in column: cast, director and genres
# 
# > Replace zero values with "NaN" null values in columns: budget, revenue
# 
# > Drop zero values in column with small quantity: runtime
# 

# In[338]:


# drop imdb_id, homepage, tagline, overview, budget_adj, revenue_adj.
df.drop(['imdb_id','homepage','tagline','overview','budget_adj','revenue_adj'],axis=1,inplace=True)
df.head()


# In[339]:


#sum of duplicates
sum(df.duplicated())


# In[340]:


# Drop Duplicates
df.drop_duplicates(inplace=True)


# In[341]:


#drop the null values in cast, director, genres columns
cal2 = ['cast', 'director', 'genres']
df.dropna(subset = cal2, how='any', inplace=True)


# In[342]:


# check if nulls are dropped.
df.isnull().sum()


# In[343]:


# directly filter the runtime data with nonzero value
df.query('runtime != "0"', inplace=True)
#check
df.query('runtime == "0"')


# In[344]:


#replace zero values with null values in the budget and revenue column.
df['budget'] = df['budget'].replace(0, np.NaN)
df['revenue'] = df['revenue'].replace(0, np.NaN)
# check if nulls are added in budget and revenue columns
df.info()


# In[345]:


# check number of unique values 
df.nunique()


# In[346]:


# column, row information after cleaning data
df.shape


# In[347]:


# Descriptive Statistics information after cleaning data
df.describe()


# > From table above shows the final statistics data info after transfer all zero values to null values in `budget` and `revenue` data. Now budget and revenue colums have some value without zero values accumulation. After deleting the zero values from runtime, the minimum value of runtime looks better. Budget and revenue columns minimum values are 1.0 dollar. This looks suspicious. When i lokeed into the data, i noticed small number of data has budget values ranging from 1 dollar to 100 dollar. Because of small quantity, i leave as it is.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Research Part 1: General Exploration
# 
# <li><a href="#Analysis 1">Analysis 1: Budget/Revenue Trend over the Period of Time</a></li>
# <li><a href="#Analysis 2">Analysis 2: Popularity Trend over the Period of Time</a></li>
# <li><a href="#Analysis 3">Analysis 3: Average Vote Trend over the Period of Time</a></li>
# <li><a href="#Analysis 4">Analysis 4: Number of Movies Released Over the Time</a></li>
# <li><a href="#Analysis 5">Analysis 5: Runtime Distribution of Movies</a></li>
# 
# <a id='research part 2a'></a>
# ### Research Part 2a (Quantitative Analyses): Properties associated with High Revenue Movies
# 
# <li><a href="#Question 1a">Research Question 1 (Which Revenue Level receives the highest popularity?)</a></li>
# <li><a href="#Question 2a">Research Question 2 (Which Budget Level Receives the Highest Popularity?)</a></li>
# <li><a href="#Question 3a">Research Question 3 (Which Runtime Level Receives the Highest Popularity?</a></li>
# <li><a href="#Question 4a">Research Question 4 (Which Runtime Level Receives the Highest Avergae Voting?</a></li>
# 
# <a id='research part 2b'></a>
# ### Research Part 2b (Categorical Analyses): Properties associated with High Revenue Movies 
# 
# <li><a href="#Question 1b">Research Question 1 (Which Movies are Top 10 highest popularity?)</a></li>
# <li><a href="#Question 2b">Research Question 2: (What kind of Genres are top in High Revenue movies?)</a></li>
# <li><a href="#Question 3b">Research Question3: (Which Production Companies are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 4b">Research Question 4: (Which Actors are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 5b">Research Question 5: (Which directors are top in Best Revenue movies?)</a></li>
# <li><a href="#Question 6b">Research Question 6: (Which keywords are top in Best Revenue movies?)¶</a></li>
# <li><a href="#Question 7b">Research Question 7: (Which Genres are top during 60s and 2Ks Best Revenue movies?)¶</a></li>
# <li><a href="#Question 8b">Research Question 8: (Which Keywords are top during 60s and 2Ks Best Revenue movies?)¶</a></li>
# 
# 
# 
# 

# ## Research Part 1: General Exploration
# 
# > In this part, the general trend of various attributes over the period of time were analysed.   

# <a id='Analysis 1'></a>
# 
# ### Analysis 1: Budget/Revenue Trend over the Period of Time
# 
# > Below plot shows the mean Budget and Revenue price trend over the periof of time. Budget increased over the period of time. Particularly from 1995 onwards, budget of the movie were increased double. Similarly revenue also increased over the period of time. 

# In[348]:


# plotting parameters
rc_fonts = {'figure.figsize': (15, 10)}
matplotlib.rcParams.update(rc_fonts)
plt.tick_params(labelsize=10)
plt.xticks(rotation=90,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')

# the width of the bars
width = 0.35       

# plot bars
df_budget = df.groupby(['release_year']).mean().budget
df_revenue = df.groupby(['release_year']).mean().revenue
df_release_year = df['release_year'].unique()
df_release_year.sort()
ind = np.arange(len(df_budget))  
budget_bars = plt.bar(ind, df_budget, width, color='r', alpha=.7, label='Budget')
revenue_bars = plt.bar(ind + width, df_revenue, width, color='g', alpha=.7, label='Revenue')

# title and labels
plt.ylabel('Mean Value($)',fontsize=15)
plt.xlabel('Release Year',fontsize=15)
plt.title('Budget/Revenue Trend Over the Time',fontsize=18,weight='bold')
locations = ind + width / 2  # xtick locations
labels = df_release_year  # xtick labels
plt.xticks(locations,labels)
plt.legend(loc='upper left')


# legend
plt.show()




# <a id='Analysis 2'></a>
# ### Analysis 2: Popularity Trend over the Period of Time
# 
# > The mean popularity trend over the period of time plot is shown below. The popularity score does not have upper limit, hence the mean value is affected by outlier which is reflected in 2015. Overall the mean popularity increases slowly with time. It is due to number of people watching movies and voting from various sources increased over the period of time. 

# In[349]:


# plotting parameters
rc_fonts = {'figure.figsize': (15, 10)}
matplotlib.rcParams.update(rc_fonts)
plt.tick_params(labelsize=10)
plt.xticks(rotation=90)
x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55])
my_xticks= df['release_year'].unique()
my_xticks.sort()
y = df.groupby(['release_year']).mean().popularity

plt.xticks(x, my_xticks,fontsize = 12, weight='bold')
plt.title('Popularity Trend Over the Time',fontsize =18,weight='bold')
plt.ylabel('Mean Popularity',fontsize=15)
plt.xlabel('Release Year',fontsize=15)
plt.xticks(rotation=90,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')


plt.plot(x, y);



# <a id='Analysis 3'></a>
# ### Analysis 3: Average Vote Trend over the Period of Time
# 
# > Surprisingly the average vote trend is decreasing slowly over the period of time. AS per [IMDb](https://help.imdb.com/article/imdb/track-movies-tv/weighted-average-ratings/GWT2DSBYVT2F25SK#), the vote average is the weighted average and not raw vote average. Various filters are applied to the raw data in order to eliminate and reduce attempts at vote stuffing by people more interested in changing the current rating of a movie than giving their true opinion of it. The Reason for the decreasing trend may be due to applying strict filter to the vote average over the period of time inorder to get accurate average vote 

# In[350]:


#  plotting parameters
rc_fonts = {'figure.figsize': (15, 10)}
matplotlib.rcParams.update(rc_fonts)
plt.tick_params(labelsize=10)
plt.xticks(rotation=90)
x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55])
my_xticks= df['release_year'].unique()
my_xticks.sort()
plt.xticks(rotation=90,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')


y = df.groupby(['release_year']).mean().vote_average

plt.xticks(x, my_xticks,fontsize = 12, weight='bold')
plt.title('Vote Average Trend Over the Time',fontsize=18,weight='bold')
plt.ylabel('Mean Vote avergae',fontsize = 15)
plt.xlabel('Release Year',fontsize=15)

plt.plot(x, y);


# <a id='Analysis 4'></a>
# ### Analysis 4: Number of Movies Released Over the Time
# 
# > Below plot indicates the drastic increase in number of movies released over the period of time. During economic downtime, the number of movies released were less and it is reflected in the chart. After 2005, the number of movies released were so high compared to 60s to 90s.This may be due to growth in economy and increase in number of people watching movies through different platforms all over the world. 

# In[351]:


#plotting a histogram of runtime of movies
rc_fonts = {'figure.figsize': (15, 10)}
#giving the figure size(width, height)
plt.figure(figsize=(10,8), dpi = 100)
#x-axis label name
plt.xlabel('Release Year', fontsize = 12)
#y-axis label name
plt.ylabel('Number of Movies Released', fontsize=12)
#title of the graph
plt.title('Number of Movies Released over the period of Time', fontsize=15, weight='bold')

plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')


#giving a histogram plot
plt.hist(df['release_year'], rwidth = 0.5, bins =50)
#displays the plot
plt.show()


# <a id='Analysis 5'></a>
# ### Analysis 5: Runtime Distribution of Movies
# 
# > Below plot shows the bell curve distribution for the runtime. Statistical correlations for runtime are tabulated below. The mean runtime is 103 min which is shown in the plot. Maximum number of movies are produced with mean runtime of 103 min.

# In[352]:


#plotting a histogram of runtime of movies
rc_fonts = {'figure.figsize': (15, 10)}
#giving the figure size(width, height)
plt.figure(figsize=(10,8), dpi = 100)
#x-axis label name
plt.xlabel('Runtime', fontsize = 12)
#y-axis label name
plt.ylabel('Number of Movies', fontsize=12)
#title of the graph
plt.title('Runtime Distribution', fontsize=15, weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xlim(0,250)

#giving a histogram plot
plt.hist(df['runtime'], rwidth = 0.5, bins =200)
#displays the plot
plt.show()
# descriptive statistical information for runtime
df['runtime'].describe()


# <a id='Research Part 2a'></a>
# ## Research Part 2a (Quantitative Analyses): Properties associated with High Revenue Movies 
# 
# > IN order to find the associated properties  for high revenue movies, two things need to be considered. One is quantitative analyses and other one is categorical analyses. In quantitative analyses, BUdget and runtime key factors are considered for the quantification of popularity and vote average for the successful movies. 
# 
# > NOw we will see how BUdget and runtime key factors are associated with successful movies.

# <a id='Question 1a'></a>
# ### Research Question 1 (Which Revenue Level receives the highest popularity?)

# > In order to determine the differnt revenue levels to categorize, pandas describe function is used to get the statistical properties. min, 25%, 50%, 75%, max revenue values were shown below.
# 
# > The revenue levers were categorized based on min,25%,50%,75% and max revenue value. 
# 
# >Below chart shows the mean popularity for different Revenue levels. HIgh revenue movies are in high popularity level compared to low reveue movies. 

# In[353]:


# View the min, 25%, 50%, 75%, max revenue values with Pandas describe
df.describe().revenue


# In[354]:


df.median().revenue


# In[355]:


# plotting parameters
rc_fonts = {'figure.figsize': (15, 10)}
# Bin edges that will be used to "cut" the data into groups
bin_edges = [2.00, 7.8e+06, 3.2e+07, 1.0e+08, 2.8e+09]

# Labels for the four revenue level groups
bin_names = ['Low (< $7.8MM)', 'Medium ($7.8MM to $32MM)', 'Mod-High ($32MM to $100MM)', 'High ($100MM to $200MM)']

# Creates arevenue_levels column
df['Revenue_Levels'] = pd.cut(df['revenue'], bin_edges, labels=bin_names)


#plotting
colors=('red','blue', 'yellow', 'green')
x= df.groupby('Revenue_Levels').mean().popularity
x.plot(kind = 'bar',alpha=0.8,color=colors)
plt.xlabel('Revenue Levels', fontsize=15)
plt.ylabel('Mean Popularity',fontsize=15)
plt.title('Popularity for Different Revenue Levels',fontsize = 18,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.show()



# <a id='Question 2a'></a>
# ### Research Question 2  (Which Budget Level Receives the Highest Popularity?)
# 
# > In order to determine the differnt budget levels to categorize, pandas describe function is used to get the statistical properties. min, 25%, 50%, 75%, max budget values were shown below.
# 
# > The budget levels were categorized based on min,25%,50%,75% and max budget value. 
# 
# >Below chart shows the mean popularity for different budget levels. As in the case of Revenue case, here also, the HIgh budget movies are in high popularity level compared to low budget movies. 

# In[356]:


# View the min, 25%, 50%, 75%, max budget values with Pandas describe
df.describe().budget


# In[357]:



rc_fonts = {'figure.figsize': (15, 10)}



# Bin edges that will be used to "cut" the data into groups
bin_edges = [1.00, 6.0e+06, 1.8e+07, 4.0e+07, 4.3e+08]

# Labels for the four budget level groups
bin_names = ['Low (< $6.0MM)', 'Medium ($6.0MM to $18MM)', 'Mod-High ($18MM to $40MM)', 'High ($40MM to $43MM)']

# Creates budget column
df['Budget_Levels'] = pd.cut(df['budget'], bin_edges, labels=bin_names)

#plotting
colors=('red','blue', 'yellow', 'green')
x= df.groupby('Budget_Levels').mean().popularity
x.plot(kind='bar',alpha=0.8,color = colors)
plt.xlabel('Budget Levels', fontsize=15)
plt.ylabel('Mean Popularity',fontsize=15)
plt.title('Popularity for Different Budget Levels',fontsize = 18,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xticks(rotation=0,weight='bold')
plt.show()


# <a id='Question 3a'></a>
# ### Research Question 3 (Which Runtime Level Receives the Highest Popularity?
# 
# > In order to determine the differnt runtime levels to categorize, pandas describe function is used to get the statistical properties. min, 25%, 50%, 75%, max runtime values were shown below.
# 
# > The runtime levels were categorized based on min,25%,50%,75% and max runtime value. 
# 
# >Below chart shows the mean popularity for different runtime levels. As in the case of Revenue, budget cases, here also, the HIgh runtime movies are in high popularity level compared to low runtime movies. 
# 

# In[358]:


# View the min, 25%, 50%, 75%, max runtime values with Pandas describe
df.describe().runtime


# In[359]:


rc_fonts = {'figure.figsize': (15, 10)}

# Bin edges that will be used to "cut" the data into groups
bin_edges = [3.0, 90.0, 99.0, 112, 900.0]

# Labels for the four runtime level groups
bin_names = ['Low (< 90min)', 'Medium (90 to 99min)', 'Mod-High(99 to 112 min)', 'High(112 to 900min)']

# Creates runtime_levels column
df['Runtime_Levels'] = pd.cut(df['runtime'], bin_edges, labels=bin_names)

# plotting
colors=('red','blue', 'yellow', 'green')
x= df.groupby('Runtime_Levels').mean().popularity
x.plot(kind='bar',alpha=0.8,color=colors)
plt.xlabel('Runtime Levels', fontsize=15)
plt.ylabel('Mean Popularity',fontsize=15)
plt.title('Popularity for Different Runtime Levels',fontsize = 18,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xticks(rotation=0,weight='bold')
plt.show()


# <a id='Question 4a'></a>
# ### Research Question 4 (Which Revenue Level Receives the Highest Avergae Voting?
# 
# > In order to determine the differnt revenue levels to categorize, pandas describe function is used to get the statistical properties. min, 25%, 50%, 75%, max revenue values were shown below.
# 
# > The revenue levels were categorized based on min,25%,50%,75% and max revenue value. 
# 
# >Below chart shows the mean vote average for different Revenue levels. Surprisingly, all revenue levels are in same vote average. As we discussed earlier, this may be due to implication of strict filter in the voting avergae calculation system over the period of time. 
# 

# In[360]:


# plotting
colors=('red','blue', 'yellow', 'green')
data= df.groupby('Revenue_Levels').mean().vote_average
data.plot(kind='bar',alpha=0.8,color = colors)
plt.xlabel('Revenue Levels', fontsize=15)
plt.ylabel('Vote Average',fontsize=15)
plt.title('Voting for Different Revenue Levels',fontsize=18,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xticks(rotation=0)
plt.show()


# > Based on the above analyses, the popular movies are largely associated with high BUdget movies and the runtime of movies.  

# <a id='Question 4a'></a>
# ### Research Question 4 (Which Runtime Level Receives the Highest Avergae Voting?
# 

# In[361]:


rc_fonts = {'figure.figsize': (15, 10)}

# Bin edges that will be used to "cut" the data into groups
bin_edges = [3.0, 90.0, 99.0, 112, 900.0]

# Labels for the four runtime level groups
bin_names = ['Low (< 90min)', 'Medium (90 to 99min)', 'Mod-High(99 to 112 min)', 'High(112 to 900min)']

# Creates runtime_levels column
df['Runtime_Levels'] = pd.cut(df['runtime'], bin_edges, labels=bin_names)

# plotting
colors=('red','blue', 'yellow', 'green')
x= df.groupby('Runtime_Levels').mean().vote_average
x.plot(kind='bar',alpha=0.8,color = colors)
plt.xlabel('Runtime Levels', fontsize=15)
plt.ylabel('Mean Vote Average',fontsize=15)
plt.title('Voting Average for Different Runtime Levels',fontsize = 18,weight='bold')
plt.xticks(rotation=0,fontsize=12,weight='bold')
plt.yticks(fontsize=12,weight='bold')
plt.xticks(rotation=0,weight='bold')
plt.show()


# The above chart shows voting average for different runtime levels. HIgh runtime has high voting average compared to low runtime movies

# In[362]:


# adding year levels
# Bin edges that will be used to "cut" the data into groups
bin_edges = [1960, 1970, 1980, 1990, 2000, 2015]

# Labels for the four budget level groups
bin_names = ['60s', '70s', '80s', '90s','2Ks']

# Creates runtime_levels column
df['Year_Levels'] = pd.cut(df['release_year'], bin_edges, labels=bin_names)


# <a id='Research Part 2b'></a>
# ## Research Part 2b (Categorical Analyses): Properties associated with High Revenue Movies 
# 
# > IN this part, we will analyse catregorical keyfactors associated with high revenue movies. The categorical analyses include cast,director, genres, keywords and producer for the successful movies. 
# 
# > NOw we will see how these categorical key factors are associated with successful movies.

# <a id='Question 1b'></a>
# ### Research Question 1 (Which Movies are Top 10 highest popularity?)

# #### Collection of Best Movies Dataset
# 
# > IN order to find top movies with highest revenue, i decided to extract data having revenue greater than or equal to 90 Million dollar. The reason for that number is from earlier analyses, the high revenue levels are associated with high popularity. Hence i decided to sort out data based on high level (ie > 90 million dollar). Then the dataframe was sorted out based on high popularity
# 

# In[363]:


# extracting data with revenue >= $90M
profit_movie_data = df[df['revenue'] >= 90000000]

#reindexing new dataframe
profit_movie_data.index = range(len(profit_movie_data))

#initialize dataframe from 1 instead of 0
profit_movie_data.index = profit_movie_data.index + 1
best_movies= profit_movie_data.sort_values(by='popularity', ascending=False)
#print(type(best_movies))
best_movies.head(10)


# > The above table shows top 10 movies dataset with revenue greater than or equal to 90 million dollar and high popularity. We will do more analyses based on this extracted dataset

# ### Collection of Worst Movies Dataset 
# 
# >IN order to find worst movies with low revenue, i decided to extract data having revenue less than or equal to 7.8 Million dollar. The reason for that number is from earlier analyses, the low revenue levels are associated with low popularity. Hence i decided to sort out data based on low level (ie > 7.8 million dollar).Then the dataframe was sorted out based on low popularity
# 

# In[364]:


# extracting data with revenue <= $7.8M
worst_profit_movie_data = df[df['revenue'] <= 7800000]

#reindexing new dataframe
worst_profit_movie_data.index = range(len(worst_profit_movie_data))
#initialize dataframe from 1 instead of 0
worst_profit_movie_data.index = worst_profit_movie_data.index + 1
worst_movies= worst_profit_movie_data.sort_values(by='popularity', ascending=True)

worst_movies.head(10)


# > The above table shows 10 movies dataset with low revenue and low popularity. We will do more categorical analyses based on this extracted dataset. 

# #### Top 10 movies with high popularity in High revenue movies 

# In[365]:


#showing the top 10 movies original title
#best_movies['original_title'].head(10)
best_movies.iloc[:11,np.r_[1:2,4:8,9:11,13:15]]


# #### Top 10 movies with low popularity in low revenue movies

# In[366]:


#showing the top 10 movies wih low revenue
worst_movies.iloc[:11,np.r_[1:2,4:8,9:11,13:15]]


# Above tables show top 10 movies, director, cast, keywords, genres, production companies,vote average and release year with high/low revenue and high/low popularity. As we can see in the dataset, attributes cast,keywords and production companies have special character '|'. It needs to be splitted. 
# We will see how we can do this

# #### Splitting String data in dataset
# 
# > Splitting string data is done for the dataset with high revenue (i.e > 90 million dollar
# 
# > Splitting string data is done for the dataset with low revenue (i.e < 7.8 million doallar)

# In[367]:


#function which will take any column as argument from and keep its track 
def data(column):
    #will take a column, and separate the string by '|'
    data = profit_movie_data[column].str.cat(sep = '|')
    
    #giving pandas series and storing the values separately
    data = pd.Series(data.split('|'))
    
    #arranging in descending order
    count = data.value_counts(ascending = False)
    
    return count


# In[368]:


#function which will take any column as argument from and keep its track 
def data_worst(column):
    #will take a column, and separate the string by '|'
    data_worst = worst_profit_movie_data[column].str.cat(sep = '|')
    
    #giving pandas series and storing the values separately
    data_worst = pd.Series(data_worst.split('|'))
    
    #arranging in descending order
    count = data_worst.value_counts(ascending = False)
    
    return count


# #### Top Genres in HIgh revenue movies

# In[369]:


# Top genres in best MOvies 
genr = data('genres')
genr.sort_values(ascending = False, inplace = True)
genr.head()



# #### Less popular genres in low revenue movies

# In[370]:


# Less popular genres in best movies 
genr_worst = data_worst('genres')
genr_worst.sort_values(ascending = False, inplace = True)
genr_worst.tail()


# #### Top directors in HIgh revenue movies

# In[371]:


# top directors in best movies
director = data('director')
director.sort_values(ascending = False, inplace = True)
director.head()


# #### Less Popular directors in Low revenue movies

# In[372]:


# less popular directors in best movies
director_worst = data_worst('director')
director_worst.sort_values(ascending = False, inplace = True)
director_worst.tail()


# #### Top actors in HIgh revenue movies

# In[373]:


# top actors in best movies
casts = data('cast')
casts.sort_values(ascending = False, inplace = True)
casts.head()


# #### Less POpular Directors in low revenue movies

# In[374]:


# less popular actors in best movies
casts_worst = data_worst('cast')
casts_worst.sort_values(ascending = False, inplace = True)
casts_worst.tail()


# #### Top Production Companies in High Revenue movies

# In[375]:


# top production companies in best movies 
prod = data('production_companies')
prod.sort_values(ascending = False, inplace = True)
prod.head()


# #### Less POpular Production Companies in Low revenue movies

# In[376]:


# less popular production companies in best movies 
prod_worst = data_worst('production_companies')
prod_worst.sort_values(ascending = False, inplace = True)
prod_worst.tail()


# #### Top Keywords in High revenue MOvies

# In[377]:


# top keywords in best movies 
key = data('keywords')
key.sort_values(ascending = False, inplace = True)
key.head()


# #### Less POpular keywords in Low revenue movies

# In[378]:


# less popular keywords in best movies 
key_worst = data_worst('keywords')
key_worst.sort_values(ascending = False, inplace = True)
key_worst.tail()


# We will see above analyses results visually

# <a id='Question 2b'></a>
# ### Research Question 2: (What kind of Genres are top in High Revenue movies?)

# In[379]:


#lets plot the points in descending order top to bottom as we have data in same format.
#genr.sort_values(ascending = False, inplace = True)

#ploting
genr. plot.barh(color = 'b', fontsize = 13)

#title
plt.title('Top Genres in HIgh Revenue Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in the dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Genres', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
genr.head()


# > The above chart indicates Comedy genres play vital role in about 487 movies for the popularity. It is followed by Action and Drama. Adventure and Thriller genres also play vital rolse in popularity of the movie 

# <a id='Question 3b'></a>
# ### Research Question 3: (Which Production Companies are top in Best Revenue movies?)

# In[380]:


#lets plot the points in descending order top to bottom as we have data in same format.

#prod.sort_values(ascending = False,inplace=True)

#ploting
prod.head(20).plot.barh(color = 'g', fontsize = 12)

#title
plt.title('Top Production Companies in Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in the dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Production Companies', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
prod.head()


# > From above chart, it is clear that WarnerBrothers tops the list with high number of movies. it is followed by UNiversal Pictures and paramount pictures

# <a id='Question 4b'></a>
# ### Research Question 4: (Which Actors are top in Best Revenue movies?)

# In[381]:


#lets plot the points in descending order top to bottom as we have data in same format.
#director.sort_values(ascending = False, inplace = True)

#ploting
casts.head(20).plot.barh(color = 'y', fontsize = 12)

#title
plt.title('Top Actors in Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in the dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Actors', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
casts.head()


# Abvoe chart shows top actor Tom Cruise played 26 movies in high revenue movies. he is followed by Brad Pitt and Tom Hanks.

# <a id='Question 5b'></a>
# ### Research Question 5: (Which directors are top in Best Revenue movies?)

# In[382]:


#lets plot the points in descending order top to bottom as we have data in same format.
#director.sort_values(ascending = False, inplace = True)

#ploting
director.head(20).plot.barh(color = 'orange', fontsize = 12)

#title
plt.title('Top Directors in Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in the dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Directors', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
director.head()


# > It is clear from the above chart that director Steven Spielberg tops the list with 23 movies in high revenue movies. He is followed by Ron Howard and Robert Zemeckis

# <a id='Question 6b'></a>
# ### Research Question 6: (Which keywords are top in Best Revenue movies?)

# In[383]:


#lets plot the points in descending order top to bottom as we have data in same format.
#director.sort_values(ascending = False, inplace = True)

#ploting
key.head(20).plot.barh(color = 'y', fontsize = 12)

#title
plt.title('Top Keywords in Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in the dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Keywords', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
key.head()


# Above chart indicates that novel keyword plays important role in popular movies. 

# <a id='Question 7b'></a>
# ### Research Question 7: (Which Genres are top during 60s and 2Ks Best Revenue movies?)

# > IN order to find top generes during 60s and 2Ks , i selected high revenue movies from dataset for 60s and 2KS. 

# In[384]:


#  finding number of genres in 1960s

best_movies_60s_data = best_movies[best_movies['Year_Levels']=='60s']

best_movies_60s_genre= best_movies_60s_data['genres'].str.cat(sep = '|')


best_movie_60s_genre_words = pd.Series(best_movies_60s_genre.split('|'))


genre_60s= best_movie_60s_genre_words.value_counts()

#plotting
genre_60s.plot.barh(color = 'y', fontsize = 12)

#title
plt.title('Top Genres in 60s Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in 60s dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Genres', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
print(genre_60s)


# In[385]:


#  finding number of genres in 2Ks

best_movies_2Ks_data = best_movies[best_movies['Year_Levels']=='2Ks']

best_movies_2ks_genre= best_movies_2Ks_data['genres'].str.cat(sep = '|')

best_movie_2ks_genre_words = pd.Series(best_movies_2ks_genre.split('|'))

genre_2ks= best_movie_2ks_genre_words.value_counts()

#plotting
genre_2ks.plot.barh(color = 'b', fontsize = 12)

#title
plt.title('Top Genres in 2ks Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in 2Ks dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Genres', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
print(genre_2ks)


# > Above plot shows the 60s and 2Ks, top genres in best movies. It shows during 60s drama was popular. During 2Ks Comedy was popular. 60s dataset is less compared to 2Ks.

# <a id='Question 8b'></a>
# ### Research Question 8: (Which Keywords are top during 60s and 2Ks Best Revenue movies?)

# In[386]:


#  finding number of keywords in 1960s

best_movies_60s_data = best_movies[best_movies['Year_Levels']=='60s']

best_movies_60s_keywords = best_movies_60s_data['keywords'].str.cat(sep = '|')

best_movie_60s_keywords_words = pd.Series(best_movies_60s_keywords.split('|'))

keywords_60s= best_movie_60s_keywords_words.value_counts()

#plotting
keywords_60s.head(20).plot.barh(color = 'y', fontsize = 12)

#title
plt.title('Top Keywords in 60s Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in 60s dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Keywords', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
print(keywords_60s)


# In[387]:


#  finding number of keywords in 2Ks

best_movies_2Ks_data = best_movies[best_movies['Year_Levels']=='2Ks']

best_movies_2ks_keywords= best_movies_2Ks_data['keywords'].str.cat(sep = '|')

best_movie_2ks_keywords_words = pd.Series(best_movies_2ks_keywords.split('|'))

keywords_2ks= best_movie_2ks_keywords_words.value_counts()

#plotting
keywords_2ks.head(20).plot.barh(color = 'b', fontsize = 12)

#title
plt.title('Top Keywords in 2ks Best Movies',fontsize = 15,weight='bold')

# on x axis
plt.xlabel('Number of Movies in 2Ks dataset', color = 'black', fontsize = 13, weight = 'bold')
plt.ylabel('Keywords', color = 'black', fontsize = 13, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#ploting the graph
plt.show()
print(keywords_2ks)


# > Above plot shows the 60s and 2Ks, top keywords in best movies. It shows during 60s keyword "london" was popular. During 2Ks keyword "based on novel" was popular. 60s dataset is less compared to 2Ks.

# <a id='conclusions'></a>
# ## Conclusions
# 
# ### General Explore
# 
# > Here, I explored some general questions. overall, budget increased over the period of time. Particularly from 1995 onwards, budget of the movie were increased double. Similarly revenue also increased over the period of time. Overall the mean popularity increases slowly with time. It is due to number of people watching movies and voting from various sources increased over the period of time. Surprisingly the average vote trend is decreasing slowly over the period of time. The Reason for the decreasing trend may be due to applying strict filter to the vote average over the period of time inorder to get accurate average vote. The drastic increase in number of movies released over the period of time. During economic downtime, the number of movies released were less. After 2005, the number of movies released were so high compared to 60s to 90s.This may be due to growth in economy and increase in number of people watching movies through different platforms all over the world. Maximum number of movies are produced with mean runtime of 103 min. 
# 
# ### Properties Associated with Successful Movies
# 
# > At this part, I found out the properties that are associated with high popularity movies. HIgh budget movies are in high popularity level compared to low budget movies. Similarly, HIgh revenue movies are in high popularity level compared to low revenue movies. HIgh runtime movies are in high popularity level compared to low runtime movies. popular movies are largely associated with high BUdget movies and the high runtime movies. HIgh runtime has high voting average compared to low runtime movies. 
# 
# > Comedy genres play vital role in about maximum number of movies for the popularity. It is followed by Action and Drama. Adventure and Thriller genres also play vital rolse in popularity of the movie. Production company "Warner Brothers" tops the list with high number of movies. It is followed by UNiversal Pictures and paramount pictures. Actor "Tom Cruise" played maximum number of movies in high revenue popularity movies. He is followed by Brad Pitt and Tom Hanks. Director "Steven Spielberg" tops the list with maximum number of movies in high revenue popularity movies. He is followed by Ron Howard and Robert Zemeckis. Novel keyword plays important role in popular movies
# 
# > During 60s, genre Drama was popular, but during 2Ks Comedy genre was popular. Similarly, during 60s, keyword "london" was popular, but during 2Ks keyword "based on novel" was popular.
# 
# 
# 

# <a id='limitations'></a>
# ## Limitations
# 
# 1. Data quality: Some Values in BUdget and revenue columns are very small number with value less than 100. Some revenue and budget columns are having zero values and missing. I assume the zero values in revenue and budget column are missing, there are still a lot of unreasonable small/big value in the both of the columns. 
# 
# 2. As per TMDb, the popularity doesn't have the upperbound , but it actually have the high probability of having outliers. 
# 
# 3. Units of revenue and budget column: It is not sure whether the budget and revenue columns are in US dollar or not. 
# 
# 4. The inflation effect: I used the revenue and budget data to explore and I didn't use the adjusted data due to inflation.
# 
# 5. I dicussed the properties are associated with successful movies. The successful I defined here are high revenue. But I didn't find the properties of high popularity and voting score. I just assume the high revenue level are with higher popularity, which I found in general exploration part. 
# 
# 6. The categorical data, when I analysed them, I just split them one by one, and count them one by one. But the thing is, there must be some effect when these words combine. For example, the keyword based on novel is popular, but what truly keyword that makes the movie sucess maybe the based on novel&adventure.
# 

# In[388]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

