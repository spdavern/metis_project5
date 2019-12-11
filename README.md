## Final Project

Author: Sean Davern 

Desciption:	This project aims at using dense neural networks to model chemical compound toxicities based on physiochemical and stereochemical properties.  It was the fifth and final project of my Metis Data Science boot camp.

## Project Documentation

Documentation for this project is represented by Jupyter Notebook (.ipynb files) comments, the slide presentation in `./reports/slide_deck.pptx (and .pdf)` and in the summary report in `./reports/summary.pages (and .pdf)`.  Finally, see brief explanations of what project work each file is associated with below.

## Data

Data used for this project came from the [Institute of Bioinformatics Johannes Kepler University](http://bioinf.jku.at/research/DeepTox/tox21.html) which provided data from the National Institutes of Health's [2014 Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/index.jsp).

Reference:

> [Mayr2016] Mayr, A., Klambauer, G., Unterthiner, T., & Hochreiter, S. (2016). DeepTox: Toxicity Prediction using Deep Learning. *Frontiers in Environmental Science*, **3**:80. [doi/10.3389/fenvs.2015.00080](https://www.frontiersin.org/articles/10.3389/fenvs.2015.00080/full)

> [Huang2016] Huang, R., Xia, M., Nguyen, D. T., Zhao, T., Sakamuru, S., Zhao, J., Shahane, S., Rossoshek, A., & Simeonov, A. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. *Frontiers in Environmental Science*, **3**:85.

## Project Organization 
------------

(generated with [datasciencemvp](https://github.com/cliffclive/datasciencemvp/))

(modified from [cookiecutter-datascience](https://drivendata.github.io/cookiecutter-data-science/))



```
.
├── GPU_attempt.ipynb		My attempt to use my mac's GPU
├── Imbalance+metric.ipynb	Working out balancing classes in data and using alternate
│					 metric for Keras optimizer
├── LICENSE
├── README.md		This file
├── bayesian_optimization_singe_dnn.ipynb		Implementation of hyperparameter tuning with
│					HyperOpt for the NR.AhR
├── data
│   ├── interim
│   ├── processed		Each subfolder named with a numeral contains pickled pandas dataframes
│   │   │	containing a single compound.  The numeral is the number of targets the compound
│   │   │	test positive for
│   │   ├── 1
│   │   │   ├── NCGC00260696-01.pkl
│   │   │   ├── ...108 other similar files
│   │   │   └── NCGC00357289-01.pkl
│   │   ├── 2
│   │   │   ├── NCGC00260789-01.pkl
│   │   │   ├── ...51 other similar files
│   │   │   └── NCGC00357254-01.pkl
│   │   ├── 3
│   │   │   ├── NCGC00260731-01.pkl
│   │   │   ├── ...29 other similar files
│   │   │   └── NCGC00357288-01.pkl
│   │   ├── 4
│   │   │   ├── NCGC00260831-01.pkl
│   │   │   ├── ...17 other similar files
│   │   │   └── NCGC00357284-01.pkl
│   │   ├── 5
│   │   │   ├── NCGC00261052-01.pkl
│   │   │   ├── ...8 similar itesm
│   │   │   └── NCGC00357109-01.pkl
│   │   ├── 6
│   │   │   ├── NCGC00261776-01.pkl
│   │   │   ├── NCGC00357007-01.pkl
│   │   │   ├── NCGC00357011-01.pkl
│   │   │   └── NCGC00357249-01.pkl
│   │   ├── 7
│   │   │   ├── NCGC00261332-01.pkl
│   │   │   ├── NCGC00261662-01.pkl
│   │   │   └── NCGC00261683-01.pkl
│   │   ├── 8
│   │   │   ├── 3-Chloro-4-methyl-7-hydroxycoumarin\ |\ C10H7ClO3\ -\ PubChem.webloc
│   │   │   ├── NCGC00357111-01.pkl
│   │   │   └── SID\ 251919981\ -\ PubChem.webloc
│   │   ├── modeled_feature_names.pkl		A pickled list of names of the targets
│   │   └── non-toxic
│   │       ├── NCGC00260691-01.pkl
│   │       ├── ...414 other similar files
│   │       └── NCGC00357287-01.pkl
│   └── raw
│       ├── tox21		This folder is what tox21.zip expands to  (it is excluded from Github)
│       │   ├── sampleCode.R
│       │   ├── sampleCode.py
│       │   ├── tox21.sdf.gz
│       │   ├── tox21_compoundData.csv
│       │   ├── tox21_dense_test.csv.gz
│       │   ├── tox21_dense_train.csv.gz
│       │   ├── tox21_labels_test.csv.gz
│       │   ├── tox21_labels_train.csv.gz
│       │   ├── tox21_sparse_colnames.txt.gz
│       │   ├── tox21_sparse_rownames_test.txt.gz
│       │   ├── tox21_sparse_rownames_train.txt.gz
│       │   ├── tox21_sparse_test.mtx.gz
│       │   └── tox21_sparse_train.mtx.gz
│       └── tox21.zip		The data as provided by 
├── flask_app
│   ├── html
│   │   ├── index.html
│   │   ├── javascripts
│   │   │   └── main.js
│   │   └── stylesheets
│   │       └── main.css
│   ├── lr.pkl		This is the model the original flask app example used.
│   ├── models		A set of my models that predict toxicity for the indicated target.
│   │   ├── NR_AR.h5
│   │   ├── NR_AR_LBD.joblib
│   │   ├── NR_AhR.joblib
│   │   ├── NR_Aromatase.joblib
│   │   ├── NR_ER.joblib
│   │   ├── NR_ER_LBD.joblib
│   │   ├── NR_PPAR_gamma.h5
│   │   ├── SR_ARE.joblib
│   │   ├── SR_ATAD5.joblib
│   │   ├── SR_HSE.joblib
│   │   ├── SR_MMP.joblib
│   │   └── SR_p53.joblib
│   ├── predictor_api.py		The primary flask app python file
│   ├── predictor_app.py		The python methods that support the predictions (incomplete)
│   ├── templates
│   │   └── predictor.html
│   └── uploads		A molecule parameters file uploaded using the app
│       └── NCGC00357111-01.pkl
├── individual_compound_generation.ipynb		Used to generate all the /Data/Processed/  files
├── input_parameter_exploration.ipynb		Used to do EDA
├── keras_save-load_issue.ipynb		Documents the Keras issue with loading models with 
│				and InputLayer layer
├── main.py		Not used
├── matthews_correlation_coeff.ipynb	Used to explore using the Matthews Corr Coef
├── model_performance_tables.ipynb	Allow easy exploration of all obtained model results
├── models		Contains pickled model objects for each target and summary tables
│   ├── NR_AR
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_AR.pkl
│   ├── NR_AR_LBD
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_AR_LBD.pkl
│   ├── NR_AhR
│   │   ├── DNN0.h5
│   │   ├── DNN1.h5
│   │   ├── DNN1y.joblib
│   │   ├── DNN1z.pkl
│   │   ├── DNN2.h5
│   │   ├── DNN3.h5
│   │   ├── DNN4.h5
│   │   ├── DNN5.h5
│   │   ├── DNN6.h5
│   │   ├── DNN7.h5
│   │   ├── DNN_modT0.h5
│   │   ├── DNNtest0.h5
│   │   ├── DNNtest1.h5
│   │   ├── DNNtest2.h5
│   │   ├── RF0.joblib
│   │   ├── RF0.joblib\ copy
│   │   ├── tracking-v1.csv
│   │   ├── tracking.csv
│   │   ├── trials.pkl
│   │   └── trials_v1.pkl
│   ├── NR_AhR.pkl
│   ├── NR_Aromatase
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_Aromatase.pkl
│   ├── NR_ER
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_ER.pkl
│   ├── NR_ER_LBD
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_ER_LBD.pkl
│   ├── NR_PPAR_gamma
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── NR_PPAR_gamma.pkl
│   ├── SR_ARE
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── SR_ARE.pkl
│   ├── SR_ATAD5
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── SR_ATAD5.pkl
│   ├── SR_HSE
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── SR_HSE.pkl
│   ├── SR_MMP
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── SR_MMP.pkl
│   ├── SR_p53
│   │   ├── DNN0.h5
│   │   └── RF0.joblib
│   ├── SR_p53.pkl
│   └── first_model.h5
├── mvp.ipynb		Contains my MVP
├── performance_summary.ipynb		Generates the summary of performance for all targets
├── predictor.ipynb		Intended to develop the predictor function for the flask app
├── proposal.md		The original project proposal in markdown format
├── proposal.pdf		The original project proposal in pdf format
├── random_forests.ipynb		Fits random forest to all targets
├── references		Technical references relevant to the project
│   ├── Analytical\ Tests		Details about the analytical tests behind targets
│   │   ├── Tox21\ Data.webloc
│   │   └── tox21-ahr-p1
│   │       ├── tox21-ahr-p1.aggregrated.txt
│   │       ├── tox21-ahr-p1.description.txt
│   │       ├── tox21-ahr-p1.slp.doc
│   │       └── tox21-ahr-p1.txt
│   ├── DeepTox-\ Deep\ Learning\ for\ Toxicity\ Prediction.webloc
│   ├── Extended-Connectivity\ Fingerprints\ -\ ECFPs.webloc
│   ├── Frontiers\ |\ DeepTox-\ Toxicity\ Prediction\ using\ Deep\ Learning\ |\ Environmental\ Science.webloc
│   ├── Graph_kernels_for_chemical_informatics\ -\ DFS.pdf
│   ├── Handbook\ of\ Molecular\ Descriptors\ -\ Roberto\ Todeschini,\ Viviana\ Consonni\ -\ Google\ Books.webloc
│   ├── JKU_site
│   │   ├── DeepTox_\ Deep\ Learning\ for\ Toxicity\ Prediction.pdf
│   │   ├── Huang\ etal.pdf
│   │   └── Mayr-etal.pdf
│   ├── NIH_site
│   │   ├── about.pdf
│   │   ├── data.pdf
│   │   ├── home.pdf
│   │   └── leaderboard.pdf
│   ├── Other\ Literature
│   │   ├── PR\ and\ ROC\ Curves\ -\ Davis\ and\ Goadrich.pdf
│   │   └── Receiver\ operating\ characteristic\ -\ Wikipedia.webloc
│   ├── PyBioMed\ Chem.pdf
│   ├── Wikipedia\ -\ Drug\ Discovery\ Datasets.webloc
│   ├── data_dictionary
│   ├── github-gadsbyfly-PyBioMed.webloc
│   └── purchased_articles
│       └── Kola_et_al-2004-Nature_Reviews_Drug_Discovery.pdf
├── reports		Documentation generated for this project
│   ├── figures
│   │   ├── Confusion\ Matrix.xlsx
│   │   ├── NR.AhR_DNN_PrecisionRecallCurve2.svg
│   │   ├── NR.AhR_RandomForest_PrecisionRecallCurve\ copy.svg
│   │   ├── NR.AhR_RandomForest_PrecisionRecallCurve.svg
│   │   └── overall_performance.svg
│   ├── slide\ deck.pptx		The final presentation slide deck
│   ├── summary.pages		The project final summary in mac Pages format
│   ├── summary.pdf		The project final summary in pdf format
│   ├── templates		Slide templates used or considered
│   │   ├── Microbiology\ Breakthrough\ by\ Slidesgo.pptx
│   │   └── TS101967975.potx
│   └── tree.txt		The command line generated version of this document tree.
├── sample.ipynb		The sample notebook provided by Johannes Kepler University
├── sample_NR.AhR_modified_metric.ipynb		Generates Precision-Recall Curves for the
│				the Random Forest Models as developed from sample.ipynb
├── single_task_dnn_base.ipynb		The extension of MVP.ipynb to all targets
└── src		Python source code
    ├── __init__.py
    ├── explore.py
    ├── helper_functions.py		Functions for loading data files, loading/saving
    │				models and model performance dataframes
    ├── interpret.py
    ├── metrics.py		Functions for calculating DNN training metrics
    ├── model.py
    ├── obtain.py
    ├── scrub.py
    └── utils
        ├── __init__.py
        ├── load_or_make.py
        └── make_dataset.py

```



