import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sagemaker.amazon.common as smac
# from sklearn.model_selection import train_test_split
!pip install -U imbalanced-learn
from sklearn.preprocessing import StandardScaler
# !pip install -U  scikit-learn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


import boto3
import sagemaker
bucket='accidentsagemaker'

file_key = 'US_Accidents_Dec20_updated.csv'

s3uri = 's3://{}/{}'.format(bucket, file_key)
training_file ='train.csv'
testing_file ='test.csv'
train_file_location = 's3://{}/{}'.format(bucket, training_file)
test_file_location = 's3://{}/{}'.format(bucket, testing_file)
container = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',

              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',

              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',

              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
sess = sagemaker.Session()
model_output_location = r's3://{0}/AccidentPrediction/model'.format(bucket)
my_region =boto3.session.Session().region_name
estimator = sagemaker.estimator.Estimator(container[my_region],
                                          role,
                                          train_instance_count = 1,
                                          train_instance_type='ml.t2.medium',
                                          output_path = model_output_location, 
                                          sagemaker_session = sess,
                                          base_job_name = 'xgboost-accidentprediction'
                                         )
estimator.set_hyperparameters(max_depth=5, 
                              objective = 'binary:logistic', 
                              eta=0.1,
                              subsample=0.7,
                              num_round=10,
                              eval_metric = 'auc')
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + container[my_region] + " container for your SageMaker endpoint.")
training_file = sagemaker.session.s3_input(s3_data=train_file_location, content_type = "csv")
data_channels = {'train':training_file, 'validation':training_file}
estimator.fit(inputs=data_channels, logs=True)
predictor = estimator.deploy(initial_instance_count = 1,
                             instance_type = 'ml.m4.xlarge',
                             endpoint_name = 'xgboost-accidentprediction-ver1')
