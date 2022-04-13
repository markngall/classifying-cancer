# classifying-cancer
Classifying tumours as benign or malignant using a wide array of different supervised learning algorithms.

A project for the Machine Learning and Data Mining module at the University of Sydney.

The dataset contains numerical features computed from images of tumours. Classes 1 and 2 refer to benign and malignant tumours respectively. Preprocessing included normalisation and imputing missing values. 

Algorithms investigated:
- K-Nearest Neighbour
- Logistic Regression
- Naive Bayes
- Decision Tree
- Linear Support Vector Machine (SVM)
- Ensembles (including AdaBoost, Gradient Boosting and Random Forest)

For Linear SVM and Random Forest, a grid search with stratified 10-fold cross-validation was used for hyperparameter tuning and the generalisation performance was evaluated on an unseen test set. Hyperparameter tuning was not performed for the other algorithms. They were simply evaluated using stratified 10-fold cross-validation.

The code is run using three command line arguments, the first of which is the path to the data. The second is an abbreviation for the algorithm (or P for preprocessing). The third is an optional path to a file containing parameter values (for algorithms that require them, such as K-Nearest Neighbour). Note that the algorithms assume preprocessed data (i.e. proc_data.csv).
