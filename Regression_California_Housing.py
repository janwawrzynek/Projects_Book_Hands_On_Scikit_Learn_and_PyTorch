from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()

#housing_full.info()
# All attributes are numerical, except for ocean_proximity, which is categorical. There are no missing values in the dataset.

#print(housing_full["ocean_proximity"].value_counts())
# The ocean_proximity attribute has 5 categories: <1H OCEAN, INLAND, ISLAND, NEAR BAY, and NEAR OCEAN.
#print(housing_full.describe())

housing_full.hist(bins = 50, figsize=(12, 8))
#plt.show()
# can see that the median income attribute is not expressed in dollars, but in tens of thousands of dollars. The median house value attribute is also capped at $500,000, which may be a problem for training machine learning algorithms.
# Must be aware of preprocessed attributes
# The housing median age and the median house value were also capped. This is a serious problem as tthe target attribute is the median house value.

#first must create a test set
# Must set aside data at this early stage to avoid data snooping bias,
#  which occurs when you look at the test data even indirectly during the data exploration and preparation phases.
#  This can lead to overly optimistic estimates of your model's performance on unseen data.

def shuffle_and_split_data(data, test_ratio,rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

rng = np.random.default_rng()
train_set, test_set = shuffle_and_split_data(housing_full, 0.2,rng)

print(len(train_set), "train +", len(test_set), "test")
#If the program is run again it will generate a different test set , overtime the ml algorithm
# will get to see all the data which we want to avoid.
# One solution is to save the test set on the initial run and use it later
# Or set the random number generators seed to ensure the same sequence of random numbers is generated each time the program is run. This way, the same test set will be generated every time.
# Both of these break with an updated dataset
# Best solution is to use a hash function  ( assuming instances have unique and immutable identifiers)
# Only works if new data is appended to the end of the data set to ensure consistent test set
# This method ensures that even with a refreshed dataset will contain 20% of the new instances,
# but will not contain any instance that was previously in the training set

from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#If no identifier column use row index as identifier
housing_with_id = housing_full.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# can also use most stable features to build a unique identifier, such as in this case the longitude and latitude attributes.
housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

#or we can implement a random splt using scikit-learn's train_test_split() function

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)
#Stratified Sampling ensures that the test set is representative of the whole dataset,
# especially when their is an important attribute (Median income in this case)

# important to have a sufficient number of instances in your dataset for each stratum, or else the estimate of a stratum’s importance may be biased. 

housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

# Create 10 splits for cross validation

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# For now use the first split

strat_train_set, strat_test_set = strat_splits[0]

# If only one is required an alternate way of doing this is using the stratisfy argument for train_test_split()
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)

# We wont use the income column again so lets drop it from the data set
for set_ in (strat_train_set, strat_test_set): set_.drop("income_cat",axis =1, inplace = True)

housing = strat_train_set.copy()

#Plotting data, used alpha =  0.2 parameter to make it easier to visualize the density of the data points. The plot reveals that the housing districts are mostly located along the coast, where the population density is higher. There are also some districts located inland, but they are less densely populated. The plot also shows that there are some outliers in the dataset, such as a few districts located far from the coast with very high median house values.
#housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
#plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()

# Correlations
#compute the standard correlation coefficient (also called Pearson’s r)
#  between every pair of numerical attributes using the corr() method:
#corr_matrix = housing.corr(numeric_only=True)
#print("correlations\n", corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
#By default scatter_matrix plots histograms on diagonal, but you can set the diagonal argument to "kde" to plot a kernel density estimation instead:
# Most promising attribute seems to be median income
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
#Notice a few lines indicating price caps, 500, 450, 350, 280, should remove these
#districts from the data set to prevent the model from learning these artificial limits as patterns in the data. 

########
# Attribute Combinations
#Use the base attributes to make some new interessting attributes
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print("correlations\n", corr_matrix["median_house_value"].sort_values(ascending=False))
# These new attibutes are more correlated
# houses with a lower bedroom/room ratio tend to be more expensive. 

##################################
#Prepare the Data for ML Algorithms
#separate the predictors and the labels, since you don’t necessarily want to apply the same transformations to the predictors and the target values 
# drop() creates a copy of the data and does not affect the original housing dataframe
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

##########
#Clean the Data
#Most ML algorithms cannot work with missing features
#For missing values there are 3 options

#Get rid of the corresponding districts.
#Get rid of the whole attribute.
#Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
#accomplish these easily using the Pandas DataFrame’s dropna(), drop(), and fillna() methods:
#housing.dropna(subset=["total_bedrooms"], inplace=True)  # option 1

#housing.drop("total_bedrooms", axis=1, inplace=True)  # option 2

median = housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
#Use handy Scikit-Learn class: SimpleImputer instead
#will store the median value of each feature: this will make it possible to impute missing values not only on the training set, but also on the validation set, the test set, and any new data fed to the model.
imputer = SimpleImputer(strategy="median") #  (strategy="constant", fill_value=…​). The last two strategies support non-numerical data.
#inputer can only be applied to numerical attributes, drop the non-numerical attributes before applying it
housing_num = housing.select_dtypes(include=[np.number]) 
#fit the imputer instance to the training data using the fit() 
imputer.fit(housing_num)
#you cannot be sure that there won’t be any missing values in new data after the system goes live, so it is safer to apply the imputer to all the numerical attributes:
#you can use this “trained” imputer to transform the training set by replacing missing values with the learned medians:
X = imputer.transform(housing_num)
#more powerful imputers available in the sklearn.​impute package (both for numerical features only):
#KNNImputer replaces each missing value with the mean of the k-nearest neighbors’ values for that feature. The distance is based on all the available features.
#IterativeImputer trains a regression model per feature to predict the missing values based on all the other available features. It then trains the model again on the updated data, and repeats the process several times, improving the models and the replacement values at each iteration.
#All transformers also have a convenience method called fit_transform(), which is equivalent to calling fit() and then transform() (but sometimes fit_transform() is optimized and runs much faster).
#Scikit-Learn transformers output NumPy arrays (or sometimes SciPy sparse matrices) even when they are fed Pandas DataFrames as input
# Must wrap X in a DataFrame and recover the column names and index from housing_num:
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
#Most machine learning algorithms prefer to work with numbers, so let’s convert these categories from text to numbers. 
# For this, we can use Scikit-Learn’s OrdinalEncoder class:
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#You can get the list of categories using the categories_ instance variable
# one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).
# Scikit-Learn provides a OneHotEncoder class to convert categorical values into one-hot vectors:
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# By default, the output of a OneHotEncoder is a SciPy sparse matrix, instead of a NumPy array:
# but if you want to convert it to a (dense) NumPy array, just call the toarray() method:
housing_cat_1hot.toarray()
#  set sparse_output=False when creating the OneHotEncoder, in which case the transform() method will return a regular (dense) NumPy array directly:
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)  # now a dense array
cat_encoder.categories_
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
cat_encoder.transform(df_test)
#a DataFrame containing an unknown category (e.g., "<2H OCEAN"),
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
# handle_unknown hyperparameter set to  "ignore", in which case it will just represent the unknown category with zeros:
cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)
print(cat_encoder.feature_names_in_)
print(cat_encoder.get_feature_names_out())
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)
#####################
#Feature Scaler
#MinMaxScaler or StandardScaler 
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# Standard Scaler substracts mean (result mean = 0) and divides by standard deviation (result sd = 1 )
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
# if heavy tails replace features with square roots or log transormations to reduce the impact of outliers and make the data more Gaussian-like.
# alternatively use bucketization (ex replace each value with its percentile)
#When a feature has a multimodal distribution (i.e., with two or more clear peaks, called modes),
# t can also be helpful to bucketize it, but this time treating the bucket IDs as categories, rather than as numerical values. 
# This means that the bucket indices must be encoded, for example using a OneHotEncoder 

#Another approach to transforming multimodal distributions is to add a feature for each of the modes (at least the main ones), 
# representing the similarity between the housing median age and that particular mode. 
# The similarity measure is typically computed using a radial basis function (RBF)—any function that depends only on the distance between the input value and a fixed point

from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
#  if the target distribution has a heavy tail, you may choose to replace the target with its logarithm.
# You will need to compute the exponential of the model’s prediction 
# in a simple linear regression model on the resulting scaled labels and use it to make predictions on some new data, 
# which we transform back to the original scale using the trained scaler’s inverse_transform() method.
# convert the labels from a Pandas Series to a DataFrame, since the StandardScaler expects 2D inputs.

from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# above is error prone use TransformedTarget​Regressor, avoiding potential scaling mismatches. 

from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# Custom Transformers
#Let’s create a log-transformer and apply it to the population feature:
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])
# feature that will measure the geographic similarity between each district and San Francisco:
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

from sklearn.pipeline import Pipeline

# Pipeline class to help with such sequences of transformations 
# Explicitly named version
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

#ColumnTransformer will apply num_pipeline (the one we just defined) to the numerical attributes, and cat_pipeline to the categorical attribute:

from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# ake_column_selector class that you can use to automatically select all the features of a given type, such as numerical or categorical
# if you don’t care about naming the transformers, you can use make_column_transformer(), which chooses the names for you, just like make_pipeline() does
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

# ready to apply this ColumnTransformer to the housing data:
housing_prepared = preprocessing.fit_transform(housing) # As before this outputs a NumPy array.

# Make a single pipeline that does all of the above transformations



# Use k-means clustering to identify data clusers and add a feature measuring the similarity to each cluster
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted



class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])



#I now want to create a single pipeline that will perform all the transformations Ive experimented with up to now.


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape

preprocessing.get_feature_names_out()

##################################
# Training and Evaluating the Training Set
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

# measure this regression model’s RMSE on the whole training set 
from sklearn.metrics import root_mean_squared_error
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print("RMSE:", lin_rmse)
# The median house value is around $206,000, so an RMSE of $68,000 is not good, but it is not terrible either.
# The model is underfitting the data. 
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print("RMSE:", tree_rmse)
# Model is badly overfitting the data.

# Better Evaluation using  k-fold cross-validation feature
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)

#print(pd.Series(tree_rmses).describe())

#  We know there’s an overfitting problem because the training error is low (actually zero) while the validation error is high.

from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(forest_rmses).describe())


# However, if you train a RandomForestRegressor and measure the RMSE on the training set,
#  you will find roughly 17,551: that’s much lower, meaning that there’s still quite a lot of overfitting going on.
#  Possible solutions are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.

# The goal is to shortlist a few (two to five) promising models.

####################
#Fine-tune the Model
# Grid Search
# Scikit-Learn’s GridSearchCV class to use cross-validation to evaluate all the possible combinations of hyperparameter values.

from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
print(grid_search.best_params_)
# Look at the results of the grid search:
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res.head()
# RandomizedSearchCV is often preferable over GridSearchCV, especially when the hyperparameter search space is large.
#For each hyperparameter, you must provide either a list of possible values, or a probability distribution:
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)

# Scikit-Learn also has HalvingRandomSearchCV and HalvingGridSearchCV hyperparameter search classes.
# 
# For example, the RandomForestRegressor can indicate the relative importance of each attribute for making accurate predictions: 
final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)

sorted(zip(feature_importances,
final_model["preprocessing"].get_feature_names_out()),
reverse=True)
# sklearn.feature_selection.SelectFromModel transformer can automatically drop the least useful features 


# Evaluate the Final Model on the Test Set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)  # prints 41445.533268606625

# Computea  a confidence interval for the RMSE
from scipy import stats

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse,
                              confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")




