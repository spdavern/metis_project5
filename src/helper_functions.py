import numpy as np
import pandas as pd
from scipy import io
import os
from joblib import dump, load
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score,\
                            precision_score, recall_score, accuracy_score,\
                            average_precision_score, precision_recall_curve
from keras.models import load_model

def load_data():
    """Function that loads the training and test data as provided by Mayr, et.al.
    There are 12,060 training compounds and 647 test compounds.  Sets have 801
    chemical features and 833 sparsely populated structural features.
    -----
    inputs: none
    returns: x_tr, y_tr, x_te, y_te
        x_tr, x_te are numpy arrays
        t_tr, y_te are pandas dataframes
    """
    raw_data = './data/raw/tox21/'
    y_tr = pd.read_csv(raw_data+'tox21_labels_train.csv.gz', index_col=0, compression="gzip")
    y_te = pd.read_csv(raw_data+'tox21_labels_test.csv.gz', index_col=0, compression="gzip")
    # There are 801 "dense features" that represent chemical descriptors, such as:
    # molecular weight, solubility or surface area, etc.
    x_tr_dense = pd.read_csv(raw_data+'tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
    x_te_dense = pd.read_csv(raw_data+'tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
    # There are 272,776 "sparse features" that represent chemical substructures:
    # (ECFP10, DFS6, DFS8)
    x_tr_sparse = io.mmread(raw_data+'tox21_sparse_train.mtx.gz').tocsc()
    x_te_sparse = io.mmread(raw_data+'tox21_sparse_test.mtx.gz').tocsc()
    # This code filters out the very sparse features:
    sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
    x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
    x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])
    # The resulting x's have 1,644 features.
    return x_tr, y_tr, x_te, y_te

def get_model_path(target):
    """Returns path for model information.
    -----
    input: target: case sensitive string for the toxicity target.
    returns: string containing the relative path to either a file or folder containing
             the model metrics dataframe or model objects.
    """
    return './models/'+target.replace('.','_')

def init_model_perfs():
    """Initializes the model performance metrics dataframe.
    -----
    returns: An empty pandas dataframe with columns for performance metrics kept for models.
    """
    cols=['model','threshold','accuracy','precision','recall','f1',\
          'auc_roc','avg_precision','confusion_matrix','model_filename']
    df=pd.DataFrame(columns=cols)
    return df

def get_model_perfs(target):
    """Returns a dataframe for performance metrics for models for target.  If previous
    metrics exist for target it reads the pickle file containing that dataframe.
    Otherwise returns an empty initialized data frame.
    -----
    input: target: case sensitive string for the toxicity target.
    returns: Pandas dataframe
    """
    path=get_model_path(target)+'.pkl'
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        df = init_model_perfs()
    return df

def save_model_perfs(target, df=None):
    """Pickles a dataframe for target.  If a dataframe is provided it saves it to the
    intended location.  If no dataframe is provided an empty one is created and saved.
    -----
    inputs:
      target: case sensitive string for the toxicity target.
      df: Pandas dataframe containing performance metrics.
    return: Nothing is returned.
    """
    path=get_model_path(target)+'.pkl'
    if os.path.exists(path):  # Rename existing file if one exists.
        os.rename(path,path[:-3]+'old')  # Previous 'old' file gets overwriten!
    if df is None:
        df = init_model_perfs()
    df.to_pickle(path)
    return

def check_is_best(target, metric_vals):
    """Checks to see if F1 or AUC_ROC is better for any of the available, previously
    stored metrics for the model of target indicated in metrics_vals['model'].
    -----
    inputs:
      target: case sensitive string for the toxicity target.
      metrics_vals: Pandas dataframe containing a single line of model metrics.
    returns: Boolean, True or False depending on if the supplied metric_vals are
      better than those previously recorded.  Either F1 or AUC_ROC being better 
      returns True.
    """
    if metric_vals is None:
        return False
    df = get_model_perfs(target)
    if len(df)==0:
        # No results for this target have been saved yet.
        return True
    # Get only the results for the model that metric_vals has results for. 
    model_df = df[df['model']==metric_vals['model']]
    if len(model_df)==0:
        # No results for this model have been saved yet.
        return True
    isBest = (metric_vals['f1'] > model_df['f1'].max()) or \
             (metric_vals['auc_roc'] > model_df['auc_roc'].max())
    return isBest

def read_model(target, filename):
    """ Reads a Python model object from a folder named target in the models folder.
    Objects are stored using file formats established by joblib.
    -----
    inputs:
        target: case sensitive string for the toxicity target.
        filename: string containing the filename containing the model object.
    returns: A Python model object in whatever state it was in when saved - usually fitted.
    """
    path = get_model_path(target)+'/'
    if os.path.isdir(path):
        pathname = os.path.join(path,filename)
        if os.path.exists(pathname):
            if filename[-2:].lower()=='h5':
                model = load_model(pathname)
            else:
                model = load(pathname)
        else:
            print("Can't find ",pathname)
            return
    else:
        print("Can't file target folder: ",path)
        return
    return model

def save_model(target, model_name, model,is_keras_model=False):
    """ Saves the fitted model object in a folder named target in the models folder.
    -----
    inputs:
        target: string identifying the toxicity measure
        model_name: string that describes the model
        model: python model object
        is_keras_model: boolean, True if the model being saved is a keras model (e.g. DNN)
    returns:
        filename(s): string, if a single file, of the filename of the saved model object.
                     list of filename strings if more than one file was saved.
    """
    path = get_model_path(target)+'/'
    # Verify the path of the folder of models exists, create it if not.
    if not os.path.isdir(path):
        os.mkdir(path)
    # Figure out the next filename to use:
    ext = 'h5' if is_keras_model else 'joblib'
    i = 0
    while os.path.exists(path+model_name+'%s.' % i +ext):
        i += 1
    fname = model_name+'%s.' % i +ext
    if is_keras_model:
        model.save(os.path.join(path,fname))
        filename=[fname]
    else:
        # use joblib to store the model object per sklearn documentation.
        # https://scikit-learn.org/stable/modules/model_persistence.html
        filename = dump(model, os.path.join(path,fname))
    if len(filename)==1:
        filename=filename[0]
    return str(filename).replace(path,'')

def evaluate_model_predictions(target, model_name, threshold, y_test, y_hat_test, auc_roc, AP):
    """ Evaluates model performance metrics and compiles them into a standardized
    single line Pandas dataframe ready for appending.
    -----
    inputs:
      target: case sensitive string for the toxicity target.
      model_name: string, short name for model variant used
      threshold: float, value of classifier threshold used for model
      y_test: dataframe or ndarray of test data set classification labels (0,1)
      y_hat_test: ndarray of predicted test data set classification labels (0,1)
      auc_roc: float, value of auc_roc_score for model
      AP: float, value of average_precision_score for model
    """
    metric_vals = pd.Series({'model'           :model_name,
                             'threshold'       :threshold,
                             'accuracy'        :accuracy_score(y_test,y_hat_test),
                             'precision'       :precision_score(y_test,y_hat_test),
                             'recall'          :recall_score(y_test,y_hat_test),
                             'f1'              :f1_score(y_test,y_hat_test),
                             'auc_roc'         :auc_roc,
                             'avg_precision'   :AP,
                             'confusion_matrix':confusion_matrix(y_test,y_hat_test)})
    return metric_vals

def update_model_perfs(target, performance_metrics_df, metric_vals, model,is_keras_model=False):
    """ Updates the dataframe containing the target model performance metrics including
    saving the model using a unique filename.  This method does not check to verify that
    metrics are better than previously saved models - use check_is_better to do that
    before calling this method.
    -----
    inputs:
      target: case sensitive string for the toxicity target
      performance_metrics_df: Pandas dataframe of performance metrics for models of target.
      metric_vals: Pandas dataframe of performance metrics for 1 new model (currently)
      model: Python reference to model object.
      is_keras_model: boolean, True if the model is a keras model (e.g. DNN)
    returns: Aggregated Pandas dataframe including the filename where the model was
      saved.
    """
    filename=save_model(target,metric_vals['model'],model,is_keras_model)
    performance_metrics_df=performance_metrics_df.append(metric_vals,ignore_index=True)
    performance_metrics_df.loc[len(performance_metrics_df)-1,'model_filename']=filename
    return performance_metrics_df

def check_and_save(target, metric_vals, model, is_keras_model=False):
    """ Checks if metric_vals performance metrics are better than previously recorded.
    If so, saves the model, updates the recorded metrics dataframe and saves it.
    -----
    inputs:
      target: case sensitive string for the toxicity target
      metric_vals: Pandas dataframe of performance metrics for 1 new model (currently)
      model: Python reference to model object.
      is_keras_model: boolean, True if the model is a keras model (e.g. DNN)
    returns:
      Reports outcome to stdout.
    """
    if check_is_best(target,metric_vals):
        df=get_model_perfs(target)
        df=update_model_perfs(target,df,metric_vals,model,is_keras_model)
        save_model_perfs(target,df)
        print('Model saved and metrics table updated.')
    else:
        print('Model performance not better than that previously recorded.')
    return

def adjusted_classes(y_scores, t):
    """This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    -----
    inputs:
      y_scores: ndarray of prediction scores (probabilities)
      t: threshold probability equal to or above which the predicted class is 1.
    returns: ndarray of classes predicted by y_scores given t.

    Source: 
      Kevin Arvai, Fine tuning a classifier in scikit-learn, Towards Data Science, Jan 18, 2018.
      https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    """
    return [1 if y >= t else 0 for y in y_scores]