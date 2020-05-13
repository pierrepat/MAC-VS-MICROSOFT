# MACVSMICROSOFT 
Machine learning algorithm to predict prospect purchase decision on Apple or Microsoft laptop based on a survey done on Hult student preferences

```python
########################################
# importing packages
########################################
import pandas            as pd  # data science essentials
import matplotlib.pyplot as plt                  # fundamental data visualization
import seaborn           as sns                  # enhanced visualization
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA            # pca
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

########################################
# loading data and setting display options
########################################
# loading data
final_df = pd.read_excel('survey_data.xlsx')


# setting print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
```


```python
########################################
# scree_plot
########################################
def scree_plot(pca_object, export = False):
    # building a scree plot

    # setting plot size
    fig, ax = plt.subplots(figsize=(10, 8))
    features = range(pca_object.n_components_)


    # developing a scree plot
    plt.plot(features,
             pca_object.explained_variance_ratio_,
             linewidth = 2,
             marker = 'o',
             markersize = 10,
             markeredgecolor = 'black',
             markerfacecolor = 'grey')


    # setting more plot options
    plt.title('Scree Plot')
    plt.xlabel('PCA feature')
    plt.ylabel('Explained Variance')
    plt.xticks(features)

    if export == True:
    
        # exporting the plot
        plt.savefig('top_customers_correlation_scree_plot.png')
        
    # displaying the plot
    plt.show()
```


```python
########################################
# inertia
########################################
def interia_plot(data, max_clust = 50):
    """
PARAMETERS
----------
data      : DataFrame, data from which to build clusters. Dataset should be scaled
max_clust : int, maximum of range for how many clusters to check interia, default 50
    """

    ks = range(1, max_clust)
    inertias = []


    for k in ks:
        # INSTANTIATING a kmeans object
        model = KMeans(n_clusters = k)


        # FITTING to the data
        model.fit(data)


        # append each inertia to the list of inertias
        inertias.append(model.inertia_)



    # plotting ks vs inertias
    fig, ax = plt.subplots(figsize = (12, 8))
    plt.plot(ks, inertias, '-o')


    # labeling and displaying the plot
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


########################################
# scree_plot
########################################
def scree_plot(pca_object, export = False):
    # building a scree plot

    # setting plot size
    fig, ax = plt.subplots(figsize=(10, 8))
    features = range(pca_object.n_components_)


    # developing a scree plot
    plt.plot(features,
             pca_object.explained_variance_ratio_,
             linewidth = 2,
             marker = 'o',
             markersize = 10,
             markeredgecolor = 'black',
             markerfacecolor = 'grey')


    # setting more plot options
    plt.title('Scree Plot')
    plt.xlabel('PCA feature')
    plt.ylabel('Explained Variance')
    plt.xticks(features)

    if export == True:
    
        # exporting the plot
        plt.savefig('top_customers_correlation_scree_plot.png')
        
    # displaying the plot
    plt.show()
```


```python
final_df.head()
```


```python
#printing value counts for cat variables

print("""What laptop do you currently have?\n""",
final_df['What laptop do you currently have?'].value_counts()
)

print("""\nWhat laptop would you buy in next assuming if all laptops cost the same?\n""",
final_df['What laptop would you buy in next assuming if all laptops cost the same?'].value_counts()
)

print("""\nWhat program are you in?\n""",
final_df['What program are you in?'].value_counts()
)

print("""\nWhat is your age?\n""",
final_df['What is your age?'].value_counts()
)

print("""\nWhat is your ethnicity?\n""",
final_df['What is your ethnicity?'].value_counts()
)

print("""\nWhat is your gender?\n""",
final_df['Gender'].value_counts()
)

print("""\nWhat is your nationality?\n""",
final_df['What is your nationality? '].value_counts()
)
```


```python
#Changing to lower case

final_df['What is your nationality? '] = final_df['What is your nationality? '].str.lower()

final_df['What is your nationality? '].value_counts()
```


```python
#Chinese to China
#Dominican to DR
#America to USA
#indian. to Indian to India
#Congolose dr congo to Congolese to Congo
#Ecuadorian to Ecuador
#Russian to Russia
#Brazilian to Brazil
#Colombian to Colombia
#Republica of Korea/Korea to South Korea
#Czech to Czech Republic 
#German to Germany
#Indonesian to Indonesia
#Peruvian to Peru
#Nigerian to Nigeria
#german/american , spanish/italian, british/indian ---> multi ethnic


final_df['What is your nationality? '] = final_df['What is your nationality? '].map({

      'ecuador':'ecuador',

       'indian': 'india',

       'china': 'china',

       'dominican ': 'dominican republic',

        'belgian': 'belgium',

        'swiss':'swiss',

        'japan' : 'japan',

        'costarrican': 'costa rica',

        'ugandan': 'uganda',

        'usa': 'usa',

        'nigerian':'nigeria' ,

        'chinese': 'china',

       'filipino ': 'philippines',

        'philippines': 'philippines',

        'indonesia': 'indonesia',

         'german': 'germany',

         'thai': "thailand",

       'italian': 'italy',

        'turkish': 'turkey',

         'mexican': 'mexico',

        'south korea': 'south korea',

         'norwegian': 'norwegia',

        'korea' : 'south korea',

        'german/american': 'multi-ethnic',

        'peruvian': 'peru',

        'vietnamese': 'vietnam',

        'russian': 'russia',

       'filipino': 'philippines',

       'czech republic': 'cezch republic',

        'peru': 'peru',

       'indonesian': 'indonesia',

        'colombian': 'colombia',

       'brazil' : 'brazil',

       'american': 'usa',

       'italian and spanish': 'multi-ethnic',

        'mauritius' : 'mauritius',

        'brazilian': 'brazil',

       'colombia': 'colombia',

        'taiwan': 'taiwan' ,

        'british, indian': 'multi-ethnic',

         'belarus': 'belarus',

        'venezuelan': 'venezuela',

        'indian.': "india",

        'czech' :  'cezch republic',

        'congolese': 'congo',

        'ukrainian': 'ukraine',

         'nigeria': 'nigeria',

         'kenyan': 'kenya',

          'belgian ': 'belgium',

         'kyrgyz' :'kyrgyz',

           'palestinian': 'palestine',

          'germany': 'germany',

        'republic of korea': 'south korea',

         'british': 'uk',

         'prefer not to answer': 'prefer not to answer',

          'panama': 'panama',

        'portuguese': 'portugal',

       'spain': 'spain',

        'russia': 'russia',

        'canada': 'canada' ,

        'pakistani': 'pakistan',

       'multi-ethnic': 'multi-ethnic',

       'spanish': 'spain',

        'dominican' : 'dominican republic',

        'ghanaian': 'ghana',

        'ecuadorian' :'ecuador',

        'congolese (dr congo)': 'congo',

        'canadian': 'canada'

})
```


```python
#Validating data getting rid of rows where duplicate questions answers don't make sense

for index, row in final_df.iterrows():
    if row['Encourage direct and open discussions'] - row['Encourage direct and open discussions.1'] > 3:
        final_df.drop(index, inplace = True)

for index, row in final_df.iterrows():
    if row["Take initiative even when circumstances, objectives, or rules aren't clear"] - row["Take initiative even when circumstances, objectives, or rules aren't clear.1"] > 3:
        final_df.drop(index, inplace = True)

for index, row in final_df.iterrows():
    if row['Respond effectively to multiple priorities'] - row['Respond effectively to multiple priorities.1'] > 3:
        final_df.drop(index, inplace = True)
    
final_df.shape
```


```python
#Validating data getting rid of rows where reponse is contradictory
for index, row in final_df.iterrows():
    if row["Am not interested in other people's problems"] - row['Am interested in people'] > 3:
        final_df.drop(index, inplace = True)
        
for index, row in final_df.iterrows():
    if row["Am easily disturbed"] - row['Am relaxed most of the time'] > 3:
        final_df.drop(index, inplace = True)
        
for index, row in final_df.iterrows():
    if row["Have excellent ideas"] - row['Do not have a good imagination'] > 3:
        final_df.drop(index, inplace = True)  
        
final_df.shape
```


```python
#Removing Categorical and SurveyID from Data

final_df_new = final_df.drop['What laptop do you currently have?', 'What laptop would you buy in next assuming if all laptops cost the same?',
                             'What program are you in?', 'What is your age?', 'What is your ethnicity?', 'Gender', 'What is your nationality? ']

```


```python
#Scaling the data

scaler = StandardScaler()
scaler.fit(final_df)
X_scaled = scaler.transform(final_df)

#Converting scaled data into a dataframe
final_scaled = pd.DataFrame(X_scaled)

#Reattaching column names
final_scaled.columns = final_df.columns

#Checking pre- and post scaling variance
print(pd.np.var(final_df), '\n\n')
print(pd.np.var(final_scaled))


```


```python
#Instantiating PCA object with no limit to principal components for BIG 5
pca = PCA(n_components = None,
         random_state = 802)

#Fitting and transforming
personality_pca = pca.fit_transform(final_scaled)

#Calling scree plot function
scree_plot(pca_object = pca)

#Splitting
personality_s = final_scaled.iloc[:, 0:52]
personality_s.head()

```


```python

```


```python

```
