# This script does some basic data processing such as removing duplicates, transforming the target column into a column containing two labels, one hot encoding, and an 80-20 split to produce training and test datasets.

import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "capital_gain",
    "capital_loss",
    "income",
]
class_labels = [" <=50K", " >50K"]


def print_shape(df):
    negative_examples, positive_examples = np.bincount(df["income"])
    print(
        "Data shape: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    input_data_path = os.path.join("/opt/ml/processing/input", "adult_data.csv")

    # Downloading the data from S3 into a Dataframe
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)

    # Dropping the duplicates
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)

    negative_examples, positive_examples = np.bincount(df["income"])

    # Split the dataset into 80-20 train and test datasets
    split_ratio = args.train_test_split_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
    )

    # Applying Standard Scaler, one-hot encoding transformations to the features
    t = [('kbins', KBinsDiscretizer(encode="onehot-dense", n_bins=10), ['age']), ('standard', StandardScaler(), ['capital_gain']), ('ohe', OneHotEncoder(sparse_output=False),['education', 'workclass'])]
    preprocess = ColumnTransformer(transformers=t)

    # Running preprocessing and feature engineering transformations
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    # Saving Train and Test processed data to output locations
    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    y_train.to_csv(train_labels_output_path, header=False, index=False)

    y_test.to_csv(test_labels_output_path, header=False, index=False)