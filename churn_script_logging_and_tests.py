'''The churn_script_logging_and_tests.py file contains 
functions to test the churn library module.

Author: Dung Nguyen Manh
Date: 10/11/2023
'''


import os
import logging
import joblib
import churn_library as cls

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
# Fix path to data
TEST_DATA_PATH = "./data/bank_data.csv"
CATEGORY_LST = ["Gender", "Education_Level", "Marital_Status",
                    "Income_Category", "Card_Category"]


def test_import(import_data):
    '''
    test data import - this example is completed for you 
        to assist with the other test functions
    '''
    try:
        df = import_data(TEST_DATA_PATH)
        assert df.shape[0] > 0
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    # except AssertionError as err:


def test_eda(perform_eda):
    '''
    Test perform eda function
    
    input:
        - perform_eda: output from import_data function
    '''
    save_path = "./tmp_data"
    try:
        dataframe = cls.import_data(TEST_DATA_PATH)
        perform_eda(dataframe, save_path)
    except FileNotFoundError:
        logging.error("Testing perform_eda: data not found")

    for item in ["churn_hist", "customer_age_hist", "marital_status_hist",
                "total_trans_ct_hist", "corr_heatmap"]:
        try:
            assert os.path.exists(os.path.join(save_path, item+'.png'))
        except AssertionError as err:
            logging.error("Testing perform_eda: The figure %s is missing", item)
            raise err
    logging.info("Testing perform_eda: SUCCESS")



def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        dataframe = cls.import_data(TEST_DATA_PATH)
        cls.perform_eda(dataframe)
        dataframe = encoder_helper(dataframe, CATEGORY_LST, 'Churn')

        logging.info('test_encoder_helper: mean categorical for churn successed')
    except ModuleNotFoundError:
        logging.error('test_encoder_helper: mean categorical for churn failed')

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("test_encoder_helper: data appear to no have rows or no columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    try:
        dataframe = cls.import_data(TEST_DATA_PATH)
        cls.perform_eda(dataframe)

        dataframe = cls.encoder_helper(
            dataframe, CATEGORY_LST, 'Churn')

        x_train, x_test, \
            _, _ = perform_feature_engineering(dataframe, "Churn")

        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0

        logging.info(
            'test_perform_feature_engineering: '
            'Data succesfully split into training and testing')
    except AssertionError:
        logging.error(
            'test_perform_feature_engineering: '
            'Problem occured during splitting into training and testing')


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        dataframe = cls.import_data(TEST_DATA_PATH)
        cls.perform_eda(dataframe)

        dataframe = cls.encoder_helper(
            dataframe, CATEGORY_LST, 'Churn')

        x_train, x_test, \
            y_train, y_test = cls.perform_feature_engineering(dataframe, "Churn")

        train_models(X_train=x_train, X_test=x_test ,y_train=y_train, y_test=y_test)

    except ModuleNotFoundError:
        logging.error('test_train_models: Failed to train')

    try:
        joblib.load('./models/rfc_model.pkl')
        joblib.load('./models/logistic_model.pkl')
        logging.info(
            "test_train_models: Found checkpoint")

    except FileNotFoundError as err:
        logging.error(
            "test_train_models: File checkpoints of models NOT exits")
        raise err

    for image_name in ["shap_feature_importance", "classification_report",
                       "feature_importances", "roc_curve",]:
        try:
            assert os.path.isfile(f"images/results/{image_name}.png")
        except AssertionError as err:
            logging.error("test_train_models: Figures %s NOT found", image_name)
            raise err

    logging.info("test_train_models: Training models SUCCESS")


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
