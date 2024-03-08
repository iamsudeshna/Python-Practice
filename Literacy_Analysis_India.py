#!/usr/bin/env python
# coding: utf-8

 
# 

# In[1]:


#!pip install seaborn     ----> installed seaborn
#!pip install stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from decimal import Decimal
from scipy import stats
from scipy.stats import norm
import statistics
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path_file = r"./Indian_cities.csv"
df = pd.read_csv(path_file)
df.head()


# In[3]:


print(df.describe())
print("=================================================")
print(df.shape)
print("The count of rows is", df.shape[0])
print("The count of columns is", df.shape[1])
print("=================================================")
datatypes = df.dtypes
print(datatypes)
df.info()


# As we can see , there are neither null values nor any missing fields

# 
# Questions during Analysis:
#     1. What is the state with highest population , 
#        show a stacked bar graph distribution
#     2. Scatterplot for population vs sex-ratio in a state, to check the relation
#     3. Child sex ratio distribution
#     4. How is effective literacy rate related to total graduates% (who gets graduated more male vs female)
#     5. In which state has most colleges
#     6. Which district has more colleges 
#     

# In[4]:


df1 = df[
    ['state_name','population_total','population_male','population_female']
         ].groupby(["state_name"]).sum().reset_index().sort_values(['population_total'], ascending=False)
                                                            # To group the population by state and sort in descending order
df1.index.name = 'index'
var1 = df1.iloc[0]['state_name']                            # To get the name of state which has highest population 
print("The name of state which has highest population :", var1)
df1.sort_values(['population_total'], ascending=False).head()



# In[5]:


fig, ax = plt.subplots(figsize=(35, 10))
rects1 = ax.bar(df1.state_name, df1['population_total'], label='Total Population')
ax.legend(fontsize=40) 
ax.legend(fontsize="x-large")
fig.tight_layout()
plt.show()


# In[6]:


fig, ax = plt.subplots(figsize=(35, 10))
rects2 = ax.bar(df1.state_name ,df1['population_male'] , label='Men')
ax.legend(fontsize=40) 
ax.legend(fontsize="x-large")
fig.tight_layout()
plt.show()


# In[7]:


fig, ax = plt.subplots(figsize=(35, 10))
rects2 = ax.bar(df1.state_name ,df1['population_female'] , label='Women')
ax.legend(fontsize=40) 
ax.legend(fontsize="x-large")
fig.tight_layout()
plt.show()


# In[8]:


L1=[]
sr = Decimal(0.00)
for i in range(max(df1.index)+1):
    sr = df1.iloc[i]['population_male']/df1.iloc[i]['population_female']
    L1.insert(i,sr)
df1['sex_ratio']=L1

# Applying log transformations
df1['population_total_log'] = np.log(df1['population_total'])
plt.scatter(df1['population_total_log'],df1['sex_ratio'])
plt.xlabel("X-axis : logarithm of total population")
plt.ylabel("Y-axis : Sex-ratio")
plt.show()
df1.head()


# From the scatter plot we cannot determine any relationship between sex-ratio & population

# In[9]:


#Sex-ratio Distribution per state that is ratio of males to females
fig, ax = plt.subplots(figsize=(35, 10))
rect4 = ax.bar(df1.state_name ,df1['sex_ratio'] , label='Sex-ratio')
ax.legend(fontsize=40) 
ax.legend(fontsize="x-large")
fig.tight_layout()
plt.show()


# From the above plot we can see most states have greater male population than that of females

# Now to determine if & how child sex-ratio is interlinked with child-literacy rate , then we can say there if boy child or a girl child gets more priviledge to primary education in a state

# In[10]:


df2 = df[['state_name','0-6_population_total', 
          '0-6_population_male','0-6_population_female','population_total']].groupby(["state_name"]).sum()
df2.head()


# In[11]:


df2['0-6_sex_ratio'] =  round(df2['0-6_population_male']/df2['0-6_population_female'],2)#ratio of male child to feamle child 
df2.head()


# In[12]:


plt.scatter(df2['0-6_population_male'],df2['0-6_population_female'])
plt.show()
sns.displot(df2['0-6_sex_ratio'])
figure1 = plt.figure()
res = stats.probplot(df2['0-6_sex_ratio'],plot=plt)


print("Mean = ",sum(df2['0-6_sex_ratio'])/28)
print("Median = ",statistics.median(df2['0-6_sex_ratio']))
print("Mode = ",statistics.mode(df2['0-6_sex_ratio']))


# Conclusions from the above scatterplot graph, Since the gradient or the slope of the graph is more than 1 we can say that for most states no.of boy child is more than that of girl child, and we can see that mean,median,mode  is somewhat be > 1.1

# In[13]:


df3 = df[['state_name','dist_code','literates_total','literates_male','literates_female',
          'total_graduates','male_graduates','female_graduates','population_total']].groupby(["state_name","dist_code"]).sum()

df3['literacy_rate'] = round(df3['literates_total'] / df3['population_total'],2) #literacy rate as a whole

df3['literacy_rate_male/female'] = round(df3['literates_male'] / df3['literates_female'],2)

df3['graduates_male/female%'] = round(df3['male_graduates']/df3['female_graduates'],2)
df3.head()


# In[14]:


plt.scatter(df3['literacy_rate'],df3['graduates_male/female%'])
                                      #trying to find the correlation between graduates_male-female% & literacy rate
plt.xlabel("Literacy Rate")
plt.ylabel("graduates_male-female%")

df4 = df3[['graduates_male/female%','literacy_rate']]
df4.corr(method='pearson')

sns.displot(df3['graduates_male/female%'])
figure2 = plt.figure()
res = stats.probplot(df3['graduates_male/female%'],plot=plt)

sns.displot(df3['literacy_rate'])
figure2 = plt.figure()
res = stats.probplot(df3['graduates_male/female%'],plot=plt)


# In[15]:


#Trying to plot male-vs-female graduates wrt male-vs-female literacy rate
plt.scatter(df3['literacy_rate_male/female'],df3['graduates_male/female%'])
plt.xlabel("Literacy Rate Male-Female%")
plt.ylabel("Graduates_male-female%")
plt.show()

df4 = df3[['graduates_male/female%','literacy_rate_male/female']]
df4.corr(method='pearson')


# In[16]:


#Trying to plot male-vs-female graduates wrt male-vs-female literacy rate
plt.scatter(df3['literates_female'],df3['female_graduates'])
plt.xlabel("Female Literacy")
plt.ylabel("Female Graduates")
plt.show()

df4 = df3[['literates_female','female_graduates']]
df4.corr(method='pearson')


# In[17]:


#Trying to plot male-vs-female graduates wrt male-vs-female literacy rate
plt.scatter(df3['literates_male'],df3['male_graduates'])
plt.xlabel("Male Literacy")
plt.ylabel("Male Graduates")
plt.show()

df4 = df3[['literates_male','male_graduates']]
df4.corr(method='pearson')


# In[18]:


#Trying to plot male-vs-female graduates wrt male-vs-female literacy rate
plt.scatter(df3['literates_total'],df3['total_graduates'])
plt.xlabel("Total Literates")
plt.ylabel("Total Graduates")
plt.show()

df4 = df3[['literates_total','total_graduates']]
df4.corr(method='pearson')


# Conclusion from this is that when literacy is greater than 70 percent then numbers of females graduating is also increasing
# Also from the normal distribution, of male-to-female_graduates% , we can see it is right-skewed 
# Also from the normal distribution, of literacy_rate% , we can see it is a bit-skewed , so literacy rate increases with decrease in male-to-female_graduates%, that is increase in count of females in graduation.
# 
# State-wise, Also we can say that correlation of women literates & women graduaduates is slightly more than that of males, given a chance of education , women tend to complete the graduation more than men completing graduation
# 
# If, literates increase, chances of graduates will also increase.
# 

# In[19]:


df4 = df[['state_name','literates_total',
'total_graduates','male_graduates','female_graduates']].groupby(["state_name"]).sum().reset_index().sort_values(['total_graduates'], ascending=False)
print("State which has the highest graduates", df4.iloc[0]["state_name"])
df5 = df[['state_name','literates_total',
'total_graduates','male_graduates','female_graduates']].groupby(["state_name"]).sum().reset_index().sort_values(['literates_total'], ascending=False)
print("State which has the highest literates", df4.iloc[0]["state_name"])


# So, we can conclude that the state of Maharashtra might have best educational infrastructure like more number of schools and colleges than most states






