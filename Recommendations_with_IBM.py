#!/usr/bin/env python
# coding: utf-8

# In[306]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/Users/doyindav/Desktop/data science/recommendations_with_IBM/user-item-interactions.csv')
df_content = pd.read_csv('/Users/doyindav/Desktop/data science/recommendations_with_IBM/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()


# In[3]:


df_content.head()


# ### <a class="anchor" id="Exploratory-Data-Analysis">Part I : Exploratory Data Analysis</a>
#
# Use the dictionary and cells below to provide some insight into the descriptive statistics of the data.
#
# `1.` What is the distribution of how many articles a user interacts with in the dataset?  Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.

# In[12]:


df.head()


# In[74]:


user_inter = df.groupby(['article_id']).count().reset_index()
user_inter = user_inter[['article_id', 'email']]
user_inter.head()


# In[75]:


plt.bar(data =user_inter, height='email', x='article_id', width=5, linewidth=5)
plt.title('A distribution of user interactions with articles')
plt.xlabel('Article ids')
plt.ylabel('Number of intercations for each article')


# In[78]:


print('The article with the most views was assessed by the user, {} times'.format(user_inter['email'].max()))


# In[136]:


art_most_ass = np.array(user_inter[user_inter['email']==max(user_inter['email'])])[0][0]
np.array(df[df['article_id']==art_most_ass]['title'])[0]


# In[143]:


print('The article assessed most by a single user was: \n {}'.format(np.array(df[df['article_id']==art_most_ass]['title'])[0]))


# In[153]:


# Maximum number of user interactions by any user is
np.array(df['article_id'].value_counts())[0]


# In[272]:


median_val = int(df['email'].value_counts().median())
median_val


# `2.` Explore and remove duplicate articles from the **df_content** dataframe.

# In[158]:


np.sum(df_content.duplicated())


# `3.` Use the cells below to find:
#
# **a.** The number of unique articles that have an interaction with a user.
# **b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
# **c.** The number of unique users in the dataset. (excluding null values) <br>
# **d.** The number of user-article interactions in the dataset.

# In[295]:


# Number of unique articles that have an interaction with a user
def unique(user_inter):
    at_least_1_inter = []
    np_user_inter = np.array(user_inter)
    for art_id, inter in np_user_inter:
        if inter >= 1:
            at_least_1_inter.append(np_user_inter)

    return(len(at_least_1_inter))

unique(user_inter)


# In[273]:


# Number of unique articles in the dataset(whether they have any interactions or not)

len(df_content['article_id'].unique())


# In[289]:


# Number of unique users in the dataset
len(df['email'].value_counts())


# In[283]:


# Number of user-interactions in the dataset
np.sum(df['email'].value_counts(dropna=False))


# `4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).

# In[237]:


print('The article with the most views was assessed by the user, {} times'.format(user_inter['email'].max()))


# In[239]:


print("The most viewed article_id is {}".format(np.array(user_inter[user_inter['email']==max(user_inter['email'])])[0][0]))


# In[389]:


## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1

        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


# In[291]:


df['email'].value_counts().max()


# In[270]:


df['email'].value_counts().median()


# In[296]:





# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
#
# Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.
#
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.

# In[386]:


def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    top_articles = np.array(pd.DataFrame(df['title'].value_counts()).index[:n])

    return top_articles # Return the top article titles from df (not df_content)


def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    '''
    top_articles = list(pd.DataFrame(df['title'].value_counts()).index[:5])

    top_articles_id = list(df[df['title'].isin(top_articles)]['article_id'].unique())


    return top_articles_id # Return the top article ids


# In[387]:


print(get_top_articles(10))
print(get_top_article_ids(10))


# In[ ]:





# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>
#
#
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.
#
# * Each **user** should only appear in each **row** once.
#
#
# * Each **article** should only show up in one **column**.
#
#
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.
#
#
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**.
#
# Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.

# In[928]:


# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id','article_id'])['title'].agg(lambda x: 1).unstack().fillna(0)

    return user_item # return the user_item matrix

user_item = create_user_item_matrix(df)


# In[929]:


user_item


# `2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users.
#
# Use the tests to test your function.

# In[609]:


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first

    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered

    '''
    # compute similarity of each user to the provided user
    user_item_pd = pd.DataFrame(user_item)
    similar_matrix = user_item_pd.dot(np.transpose(user_item))

    # sort by similarity
    sorted_users = similar_matrix[user_id].sort_values(ascending=False)

    # create list of just the ids
    similar_users = sorted_users.index.tolist()

    # remove the own user's id
    similar_users.remove(user_id)

    return similar_users # return a list of the users in order from most to least similar



# In[612]:


# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))


# In[ ]:





# `3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user.

# In[925]:


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the title column)
    '''
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique().tolist()

    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    new_df = pd.DataFrame(user_item).reset_index()
    article_ids = new_df.apply(lambda user_id: user_id[user_id ==1].index, axis=1)[user_id]

    user_articles = get_article_names(article_ids)

    return article_ids, user_articles


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user

    OUTPUT:
    recs - (list) a list of recommendations for the user

    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found

    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user

    For the user where the number of recommended articles starts below m
    and ends exceeding m, the last items are chosen arbitrarily

    '''
    # Your code here

    return recs # return your recommendations for this user_id



# In[927]:


set(get_user_articles(5))


# In[914]:


(get_article_names(['1024.0', '1176.0']))


# In[930]:





# In[ ]:
