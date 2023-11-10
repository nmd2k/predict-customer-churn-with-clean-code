"""library doc string"""


# import libraries
import os
import logging
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError:
        logger.error("File not found when importing data")
    return


def perform_eda(df, save_path: str = "./images/eda"):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''
    os.makedirs(save_path, exist_ok=True)

    try:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        _ = df['Churn'].hist()
        plt.savefig(os.path.join(save_path, "churn_hist.png"))
        plt.clf()
    except ModuleNotFoundError:
        logger.error("Error creating Churn column")

    try:
        _ = df['Customer_Age'].hist()
        plt.savefig(os.path.join(save_path, "customer_age_hist.png"))
        plt.clf()
    except ModuleNotFoundError:
        logger.error("Error creating Customer_Age histogram")

    try:
        _ = df.Marital_Status.value_counts(
            'normalize').plot(kind='bar')
        plt.savefig(os.path.join(save_path, "marital_status_hist.png"))
        plt.clf()
    except ModuleNotFoundError:
        logger.error("Error creating Marital_Status histogram")

    try:
        _ = sns.histplot(
            df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(os.path.join(save_path, "total_trans_ct_hist.png"))
        plt.clf()
    except ModuleNotFoundError:
        logger.error("Error creating Total_Trans_Ct histogram")

    try:
        _ = sns.heatmap(df.corr(), annot=False,
                        cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(save_path, "corr_heatmap.png"))
        plt.clf()
    except ModuleNotFoundError:
        logger.error("Error creating correlation heatmap")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument 
            that could be used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        results = []
        groups = df.groupby(col).mean()[response]
        for val in df[col]:
            results.append(groups.loc[val])

        df[col + "_" + response] = results

    return df


def perform_feature_engineering(df, response):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument 
            that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]

    x_data = pd.DataFrame()
    y_data = df[response]
    x_data[keep_cols] = df[keep_cols]
    x_train, x_test, y_split_train, y_split_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.3,
                                                        random_state=42)
    return x_train, x_test, y_split_train, y_split_test


def classification_report_image(y_training,
                                y_testing,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf) -> None:
    '''
    produces classification report for training and testing results 
        and stores report as image in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    # Classification reports
    rf_test_report = classification_report(y_testing, y_test_preds_rf)
    rf_train_report = classification_report(y_training, y_train_preds_rf)

    lr_test_report = classification_report(y_testing, y_test_preds_lr)
    lr_train_report = classification_report(y_training, y_train_preds_lr)

    # Create subplots
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    def plot_report(ax, title, report):
        ax.text(0.01, 0.99,
                title + "\n\n" + report,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top')

        ax.axis('off')

    # Plot reports on subplots
    plot_report(axs[0, 0], "Training Set Logistic Regression", lr_train_report)
    plot_report(axs[0, 1], "Training Set Random Forest", rf_train_report)
    plot_report(axs[1, 0], "Testing Set Logistic Regression", lr_test_report)
    plot_report(axs[1, 1], "Testing Set Random Forest", rf_test_report)

    plt.show()

    # Save report
    plt.savefig("images/results/classification_report.png")
    plt.clf()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    # Using shap to explain =====
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_pth, "shap_feature_importance.png"))
    plt.clf()
    #, bbox_inches='tight', dpi=1000)

    # Analysis of feature importance =====
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, "feature_importance.png"))
    plt.clf()


def train_models(x_train, x_test, y_train, y_test, save_path="./"):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    logger.info("Initializing models")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy'],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)
    logger.info("Models trained")
    logger.info("Evaluation")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    logger.info("Analyzing results")
    # Save classification report
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Plot ROC and save to images/results folder
    img_save_path = os.path.join(save_path, "images/results")
    model_save_path = os.path.join(save_path, "models")
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    _ = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)

    lrc_plot.plot(ax=ax, alpha=0.8)

    plt.savefig(os.path.join(img_save_path, "roc_curve.png"))
    plt.clf()

    # Save feature_importance visualize
    feature_importance_plot(cv_rfc, x_test, img_save_path)

    # Save models
    joblib.dump(cv_rfc.best_estimator_, os.path.join(model_save_path, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(model_save_path, 'logistic_model.pkl'))


if __name__ == "__main__":
    logger.info("Begin")
    # Load data
    dataframe = import_data("./data/bank_data.csv")

    # Perform EDA
    perform_eda(dataframe)

    CATEGORY_LST = ["Gender", "Education_Level", "Marital_Status",
                    "Income_Category", "Card_Category"]

    new_dataframe = encoder_helper(dataframe, CATEGORY_LST, "Churn")

    # Start training
    train_data, test_data, \
        tgt_train, tgt_test = perform_feature_engineering(new_dataframe, "Churn")

    train_models(x_train=train_data, x_test=test_data,
                 y_train=tgt_train, y_test=tgt_test)
    logger.info("Finish")
