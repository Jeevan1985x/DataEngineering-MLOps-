import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
import boto3
import argparse

import sagemaker
import re
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import zipfile     # Amazon SageMaker's Python SDK provides many helper functions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--featuregroupname",type=str, required=True)
    parser.add_argument("--default-bucket",type=str, required=True)
    parser.add_argument("--region",type=str, required=True)
    
    args = parser.parse_args()
    bucket = args.default_bucket
    region = args.region
    
    base_dir = "/opt/ml/processing"
        
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
    featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )

    feature_group_name = args.featuregroupname
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    fs_query = feature_group.athena_query()
    fs_table = fs_query.table_name
    query_string = 'SELECT * FROM "'+fs_table+'"'
    
    fs_query.run(query_string=query_string, output_location='s3://'+bucket+'/fs_query_results/')
    fs_query.wait()
    model_data = fs_query.as_dataframe()

    df = model_data.drop(['fs_id', 'fs_time', 'write_time', 'api_invocation_time', 'is_deleted'], axis=1)
    
    # Make a copy of the data frame and remove one column
    base_data = df.copy()
    base_data.pop("retained")
    
    # For analysis config file generation for Clarify in generate_config.py
    baseline_sample = base_data.sample(frac=0.0002)
    baseline_sample.to_csv(f"{base_dir}/baseline/baseline.csv", header=False, index=False)
    
    # To support batch transform step
    batch_sample = base_data.sample(frac=0.2)
    batch_sample.to_csv(f"{base_dir}/batch/batch.csv", header=False, index=False)
    
    y = df.pop("retained")
   
    # Split into train, validation and test datasets  
    X_pre = df
    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X_pre), axis=1)
    np.random.shuffle(X)
    
    # Split in Train, Test and Validation Datasets
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    train_rows = np.shape(train)[0]
    validation_rows = np.shape(validation)[0]
    
    test_rows = np.shape(test)[0]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    validation = pd.DataFrame(validation)
    
    # Convert the label column to integer
    train[0] = train[0].astype(int)
    test[0] = test[0].astype(int)
    validation[0] = validation[0].astype(int)
    
    # Save the Dataframes as csv files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
