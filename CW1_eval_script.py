# NOTE: Needs to be adapted to fit the previous script at a later date (the X_tst file format, etc)
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Set seed
np.random.seed(123) 

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)
logging.info("Training and test data loaded")

# Dropping 0 values for x, y, z 
trn = trn.drop(trn[trn['x']==0].index)
trn = trn.drop(trn[trn['y']==0].index)
trn = trn.drop(trn[trn['z']==0].index)

# Dropping outliers for relevant features
trn = trn[(trn["depth"]>52.5)]
trn = trn[(trn["carat"]<3.5)]
trn = trn[(trn["table"]<70)&(trn["table"]>45)]
trn = trn[(trn["y"]<50)]
trn = trn[(trn["z"]<7)&(trn["z"]>2)]

logging.info("Outliers removed")

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# Make copy to avoid changing original data
trn_copy = trn.copy()

# Label encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    trn_copy[col] = label_encoder.fit_transform(trn_copy[col])
    
logging.info("Categorial variable encoding finished")

# Training the model
# X_trn, y_trn = trn_copy[['depth', 'table', 'a1', 'a4', 'b1', 'b3']].values, trn_copy['outcome'].values
# X_tst = X_tst[['depth', 'table', 'a1', 'a4', 'b1', 'b3']].values

X, y = trn_copy[['depth', 'table', 'a1', 'a4', 'b1', 'b3']].values, trn_copy['outcome'].values
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.25, random_state=0)

pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                           ('regressor', GradientBoostingRegressor())])
model = pipeline.fit(X_trn, (y_trn))
logging.info("Inital training finished")

# Use the best found estimator, and find the best hyperparameter combination for R2 score
alg = GradientBoostingRegressor()


logging.info("Training with hyperparameters...")
hyperparameters = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, hyperparameters, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_trn, y_trn)

model = gridsearch.best_estimator_
logging.info("Training finished")

# Test set predictions
logging.info("Predicting outcomes with test data...")
yhat_lm = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23171764.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
#tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# How does the linear model do?
print(r2_fn(yhat_lm))




