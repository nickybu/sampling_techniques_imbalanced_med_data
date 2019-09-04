import math, operator, random
import numpy as np
import pandas as pd
import math
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import itertools
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

# ---- KNN for SMOTE ----

# Compute square root of the sum of the squared differences between two arrays of numbers
def euclideanDistance(sample, sample_set):
    distance = 0
    for x in range(len(sample)):
        distance += pow((sample[x] - sample_set[x]), 2)
    return math.sqrt(distance)

# Returns k most similar neighbours from the sample set for a given sample
def getKNeighbours(sample_set, sample, k):
    distances = []
    for x in range(len(sample_set)):
        dist = euclideanDistance(sample, sample_set[x])
        distances.append((x, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k+1):
        if(distances[x][0] <> sample.name):
            neighbours.append(distances[x])
    return neighbours

# ---- SMOTE ----

# T: minority class samples
# N: amount of SMOTE N%
# k: number of nearest neighbours
# Output: (N/100) * T synthetic minority class samples
def smote(T, N, k):
    # Convert T to DataFrame
    T_df = pd.DataFrame(T)
    # If N < 100% randomise minority class samples as only a random % of them will be SMOTEd
    if(N < 100):
        # Choose random N% from T
        T_df = T_df.sample(frac=N/100.0)
        N = 100
    # The amount of SMOTE is assumed to be in integral multiples of 100
    N = N/100
    # Number of attributes
    num_attrs = len(T_df.columns)
    # Keeps a count of number of synthetic samples generated
    num_synthetic_samples_generated = 0
    # Store synthetic examples
    synthetic = np.zeros((N * len(T_df), num_attrs))
    # Compute k nearest neighbors for each minority class sample
    for index, sample in T_df.iterrows():
        # Compute k nearest neighbours for i and save indices to nnarray
        nnarray = getKNeighbours(T, sample, k)
        N_temp = N
        while(N_temp != 0):
            # Choose random number between 1 and k, i.e. one of the k nearest neighbours of i
            nn = random.randint(0,k-1)
            for attr in range(num_attrs):
                # Compute difference
                T_row = nnarray[nn][0]
                dif = T[T_row][attr] - sample.values[attr]
                # Compute gap
                gap = np.random.random(1)[0]
                synthetic[num_synthetic_samples_generated][attr] = sample.values[attr] + gap * dif
            num_synthetic_samples_generated += 1
            N_temp = N_temp - 1
    return synthetic

# ---- TOMEK ----

# Calculate closest element per Euclidean distance 
import math
def get_dist(sample_1, sample_2):
    dist = 0
    for i in range(len(sample_1)):
        dist += pow(sample_1[i] - sample_2[i], 2) # Calculates the euclidean distance
    return math.sqrt(dist)

# Finds the TOMEK link for specified data point
def tomek_links(loc, x_maj, x_min):
    tomek_link = [999999,0,loc] # Initialises list with high euclidean distance for comparison
    for i in range(0,len(x_maj)):
        if get_dist(x_maj[i,:], x_min[loc,:]) < tomek_link[0]:
            tomek_link[0] = get_dist(x_maj[i,:], x_min[loc,:])
            tomek_link[1] = i
    return tomek_link

# Change label 0 and 1 depending on which class 0 and 1 belongs to
def tomek(x_majority, x_minority, y_majority, y_minority, num_to_remove, classes, classes_to_remove):
    # Creates a list of random unrepeated integers in the minority class, with a size being the number of values to remove in majority class
    vals_to_remove = random.sample(range(len(x_minority)), num_to_remove)
    # Finds the TOMEK majority class link for each data point in the minority class then removes the majority point from its separate dataset
    if classes_to_remove == 1:
        for i in range(num_to_remove):
            val_to_remove = random.randint(0,len(x_minority))-1
            tomek_link = tomek_links(val_to_remove, x_majority, x_minority)
            x_majority = np.delete(x_majority, tomek_link[1], axis=0)
            y_majority = np.delete(y_majority, tomek_link[1], axis=0)
    elif classes_to_remove == 2:
        for i in range(num_to_remove):
            val_to_remove = random.randint(0,len(x_minority))-1
            tomek_link = tomek_links(val_to_remove, x_majority, x_minority)
            x_majority = np.delete(x_majority, tomek_link[1], axis=0)
            y_majority = np.delete(y_majority, tomek_link[1], axis=0)
            x_minority = np.delete(x_minority, tomek_link[2], axis=0)
            y_minority = np.delete(y_minority, tomek_link[2], axis=0)
    
    # Commented out for later use
    """
    # Plot datapoints after applying TOMEK to dataset
    fig2 = plt.figure(2)
    plt.title('After')
    plt.scatter([x_maj[class_x].values], [x_maj[class_y].values], color='g', marker='^', label=label_maj)
    plt.scatter([x_min[class_x].values], [x_min[class_y].values], color='r', marker='*', label=label_min)
    plt.xlabel(class_x)
    plt.ylabel(class_y)
    plt.legend(loc=1)
    plt.show()
    """
    y_majority = y_majority.reshape((len(y_majority), 1))
    updated_majority = np.concatenate((x_majority, y_majority), axis=1)
    y_minority = y_minority.reshape((len(y_minority), 1))
    updated_minority = np.concatenate((x_minority, y_minority), axis=1)
    dataset = np.concatenate((updated_majority, updated_minority), axis=0)
    
    return dataset # Needs shuffling before use

# ---- SMOTE + Tomek ----

# dataset: full dataset with all column headers
# smote_per: amount of SMOTE N% (determined by applying only SMOTE on imbalanced dataset)
# k: number of nearest neighbours (determined by using KNN on imbalanced dataset)
# classes: name of class, i.e. severity, diagnosis
# class_x: column name of first data column to compare
# class_y: column name of second data column to compare
# label_0: class label that corresponds to class 0 in the class column
# label_1: class label that corresponds to class 1 in the class column
def smote_tomek(x_majority, x_minority, y_majority, y_minority, majority_class, minority_class, smote_per, k, classes):
#     x_new_samples = smote(x_minority.values, smote_per, k)
    x_new_samples = smote(x_minority, smote_per, k)
    updated_x_minority = np.concatenate((x_minority, x_new_samples), axis=0)
#     updated_x_minority = preprocessing.normalize(updated_x_minority)

    # Update y following SMOTE
    smote_y = np.full((len(x_new_samples)), minority_class)
    updated_y_minority = np.concatenate([y_minority, smote_y], axis=0)
    updated_y_minority = updated_y_minority.reshape((len(updated_y_minority), 1))
    #smoted_dataset = np.concatenate((x_smote, y_smote), 1)
    
    num_to_remove = len(x_new_samples) / 2
    if num_to_remove > len(x_minority):
        num_to_remove = len(x_minority)/2
#     tomeked_dataset = tomek(x_majority.values, updated_x_minority, y_majority.values, #updated_y_minority, num_to_remove, classes, 2)
    tomeked_dataset = tomek(x_majority, updated_x_minority, y_majority, updated_y_minority, num_to_remove, classes, 2)

    return tomeked_dataset
    
# ---- KNN: Hyperparameter Setting ----

# Find best K value given pre-set range
def knn_set_hyper_params(x_train, y_train, x_test, y_test, x, y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score
    
    k = [1,3,5,7,9]
    cv_best = -1
    k_best = 0
    cv_folds = 10

    for i in k:
        # Uses minkowski with p=2, i.e. euclidean
        model = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
        model.fit(x_train, y_train)

        # Predict on testing data
        y_pred = model.predict(x_test)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        print ("Accuracy K= %d : %f" %(i, acc_score))

        # k cross fold validation
        scores = cross_val_score(model, x, y, cv=cv_folds)
        print("K Fold Cross Validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        if(scores.mean() > cv_best):
            cv_best = scores.mean()
            k_best = i
    return k_best

# ---- SVM: Hyperparameter Setting ----

# Find best Gamma and C values given pre-set range
def svm_set_hyper_params(x_train, y_train, x_test, y_test, print_scores):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    
    for score in scores:
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10, scoring='%s_macro' % score)
        clf.fit(x_train, y_train)   
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']  
        y_true, y_pred = y_test, clf.predict(x_test)
        
        if(print_scores):
            print("# Tuning hyper-parameters for %s" % score)
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print("Grid scores on development set:")
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))      
            print("Detailed classification report:")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print(classification_report(y_true, y_pred))
    return clf.best_params_

# ---- Show Metrics ----

# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Show Precision/Recall, F1 Score, Cross-entropy Loss, Confusion Matrix
def show_metrics(y_test, y_pred, class_labels, pos_label):
    # ROC AUC score
    ROCAUC = metrics.roc_auc_score(y_test, y_pred)
    print 'ROC AUC Score: ', ROCAUC
    
    # Precision/Recall
    # Precision: result relevancy
    # Recall: measure of how many truly relevant results are returned
    # High scores indicate accurate results (high precision) and a majority of all positive results (high recall)
    precision_score = metrics.precision_score(y_test, y_pred, average='binary', pos_label=pos_label)
    print 'Precision Score: ', precision_score

    recall_score = metrics.recall_score(y_test, y_pred, average='binary', pos_label=pos_label)
    print 'Recall Score: ', recall_score

    average_precision_recall_score = metrics.average_precision_score(y_test, y_pred)
    print 'Average precision-recall score: {0:0.3f}'.format(average_precision_recall_score)

    # Precision-Recall curve
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision_recall_score))

    # F1 Score
    f1_score = metrics.f1_score(y_test, y_pred)
    print 'F1 Score: {0:0.3f}'.format(f1_score)

    # Cross-entropy loss
    cross_entropy_loss = metrics.log_loss(y_test, y_pred)
    print 'Cross-entropy Loss: {0:0.3f}'.format(cross_entropy_loss)

    # Confusion Matrix
    # Diagonal: number of samples for which predicted label is true
    # Off-diagonal: mislabeled by classifier

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_labels,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
