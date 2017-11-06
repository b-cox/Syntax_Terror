# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 13:51:19 2017

@author: bcox2
"""
# =============================================================================

import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split

# =============================================================================
# Query Vertica for Data Set Creation
# =============================================================================


vertica_con = pyodbc.connect("DSN=Vertica")

query = '''
select pam.auth_id
        , pam.first_neauth_date_adj
        , pam.tto_segment
        , pam.start_sku
        , case when pam.first_fed_efile_rejected_date_adj is not null 
                then 1 else 0 end as efile_reject_flag
        , left(dob, 4)::! numeric as birth_year
        , bam.feelings_answer_f
        , hs.care_first_area
        , min(hs.hit_date) as min_care_hit_date
        , case when sum(care.search_manual_cnt) > 50 then 50
                when sum(care.search_manual_cnt) = 0 then 0 
                else sum(care.search_manual_cnt) end as manual_search_sum
        , sum(care.hs_open_cnt) as hs_opens_sum
        , case when count(cu.trans_id) > 20 then 20
                when count(cu.trans_id) > 0 then count(cu.trans_id)
                else 0 end as contact_sum
        , case when count(cu.trans_id) >= 1 then 1 
                else 0 end as contact_flag
from ctg_analyst_layer.product_analytics_master pam
inner join ctg_analyst_layer.behavioral_analytics_master bam
        on pam.auth_id = bam.auth_id
        and pam.tax_year = bam.tax_year
        and bam.dob <> 'null'
        and bam.dob is not null
        and bam.feelings_answer_f is not null
left join
        (select auth_id
                , first_value(care_start_area ignore nulls)
                        over (partition by auth_id 
                        order by first_care_hit_ts asc) 
                        as care_first_area
                , hit_date
                , tax_year
        from ctg_analyst_layer.care_visitor_agg_daily) hs
                on pam.auth_id = hs.auth_id
                and pam.tax_year = hs.tax_year
left join ctg_analyst_layer.care_visitor_agg_daily care
        on pam.auth_id = care.auth_id
        and pam.tax_year = care.tax_year
left join ctg_analyst_layer.cu_confirmation_hits cu
        on pam.auth_id = cu.auth_id
        and pam.tax_year = cu.tax_year
where pam.tax_year = 2016
        and pam.core_flag = 1
        and pam.nonffa_flag = 1
group by 1,2,3,4,5,6,7,8
order by random()
limit 10000
'''

dat1 = pd.read_sql(query, vertica_con)
vertica_con.close()


# Looking at the data structure and basic info

dat1.head()
dat1.describe()
dat1.info()
dat1.dtypes

# =============================================================================
# Adjusting some columns to better formats for modeling
# =============================================================================

cleanup_data = {"feelings_answer_f": {"3|Don't ask": 1, "2|Not so good": 2,
                                      "1|Good": 3, "None": None},
        "tto_segment": {"New": 1, "Skip Year": 2, "1st Year Returning": 3,
                        "Veteran Returning": 4}}

dat1.replace(cleanup_data, inplace=True)

dat1['manual_search_sum'] = dat1['manual_search_sum'].fillna(0)

dat1.corr()

# =============================================================================
# Creating Dummy Variables
# =============================================================================

#segment = pd.get_dummies(dat1, columns = ['start_sku','feelings_answer_f'])
#dat2 = pd.concat([dat1,segment],axis=1)
#dat2.head()
#dat2.corr()

dat1['feelings_answer_f'] = (pd.to_numeric(dat1['feelings_answer_f'], 
     downcast = 'integer'))

dat1['feelings_answer_f'].dtype
dat1.head()

# =============================================================================
# Simple plot functions
# =============================================================================

# Using basic matplotlib 
color = ['red' if flag == 1 else 'blue' for flag in dat1.contact_flag]
dat1.plot.scatter(x='birth_year', y='manual_search_sum',c=color)

# Using Seaborn package which has nice visuals (especially categorical variables)
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

sns.boxplot(x='start_sku', y='birth_year', hue='contact_flag', data=dat1)
sns.barplot(x="start_sku", y="contact_flag", hue="feelings_answer_f", data = dat1);
sns.stripplot(x='birth_year', y='manual_search_sum', hue='contact_flag', data=dat1)

# =============================================================================
# kNN Modeling
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dat_model = dat1[['birth_year', 'manual_search_sum', 'feelings_answer_f', 
                  'efile_reject_flag','contact_flag']]

dat_model = dat_model.dropna(axis=0, how='any') # Removing NA/Nulls

X = np.array(dat_model[['birth_year', 'manual_search_sum', 
                        'feelings_answer_f','efile_reject_flag']])
y = np.array(dat_model['contact_flag'])

# Test/Train Split (default is 75% / 25% train-test split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4)

n_neighbors = 25

knn = KNeighborsClassifier(n_neighbors = n_neighbors)

# ### Train the classifier (fit the estimator) using the training data

m1 = knn.fit(X_train, y_train)

# =============================================================================
# Estimate the accuracy of the classifier on test holdout data
# =============================================================================

m1.score(X_test, y_test)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(m1, X_test, y_test)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# =============================================================================
# Confusion Matrix
# =============================================================================

from sklearn import metrics

# BotLeft = FalsePos / Bot Right = TruePos / Top Right = False Neg
print(metrics.confusion_matrix(y_true = y_test,  # True labels
                         y_pred = m1.predict(X_test))) # Predicted labels

# View summary of common classification metrics
print(metrics.classification_report(y_true = y_test,
                              y_pred = m1.predict(X_test)))



# =============================================================================
# Plotting a simple model (can only plot 2 input variables)
# =============================================================================

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Create color maps
cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#00FF00', '#FF0000', '#0000FF'])

n_neighbors = 25
                            
h = .1  # step size in the mesh

plot_X = X[:, 0:2]
plot_y = y #y_train

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(plot_X,plot_y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = plot_X[:, 0].min(), plot_X[:, 0].max() + 1
    y_min, y_max = plot_X[:, 1].min(), plot_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Contact Flag Classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# =============================================================================
# Automating the selection of k (how many neighbors?)
# =============================================================================
k_range = range(1,200)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks(np.arange(min(k_range),max(k_range),5));

# =============================================================================
# Parameter Tuning - What is the best test/train proportion
# =============================================================================

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 10)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');

# =============================================================================
# Plotting a 3D scatter plot
# =============================================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(dat_model['birth_year'], dat_model['manual_search_sum'], 
           dat_model['feelings_answer_f'], c = dat_model['contact_flag'], marker = 'o', s=100)
ax.set_xlabel('birth_year')
ax.set_ylabel('searches')
ax.set_zlabel('feelings answer')
plt.show()