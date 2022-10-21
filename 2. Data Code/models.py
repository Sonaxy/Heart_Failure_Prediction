# https://www.askpython.com/python/examples/k-nearest-neighbors-from-scratch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import mode


heart = pd.read_csv("heart.csv")


# Xg = heart.drop(columns = ['HeartDisease'] ).copy()
# Yg = heart['HeartDisease']
#print(X,"\n\n\n")
#print(Y,'\n\n\n')

# Removing 20% of data for testing
# X_train, X_test, Y_train, Y_test = train_test_split(Xg, Yg, test_size = 0.2, shuffle = True, random_state = 5 )
# splitting the leftover training data into  75% training and 25% validation data
# X_train, X_valid, Y_train, Y_valid = train_test_split(Xg,Yg, test_size = 0.25, shuffle = True, random_state = 5)

def euclidean(l1, l2):
        distance = np.sqrt(np.sum(l1-l2)**2)
        return distance 

def metric(actual, predicted):
    true_positive = [1 for act,pre in zip(actual,predicted) if act == pre == 1 ]
    true_positive = sum(true_positive)
    true_negative = [1 for act,pre in zip(actual,predicted) if act == pre == 0]
    true_negative = sum(true_negative)
    false_positive =[1 for act,pre in zip(actual,predicted) if (act == 0 and pre == 1)]
    false_positive = sum(false_positive)
    false_negative = [1 for act,pre in zip(actual,predicted) if (act == 1 and pre == 0)]
    false_negative = sum(false_negative)
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    confusion_matrix = np.array([[true_positive, false_positive] , [false_negative, true_negative]])
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = 2 * (precision * recall)/(precision + recall)
    return confusion_matrix, accuracy, precision, recall, f1 
    

def train(X_train,Y_train,x_input,k):
    predicted_labels = []

    for predictors in x_input:

        # store the distances 
        dist_arr = []
        for row in range(len(X_train)):
            distances = euclidean(np.array(X_train[row,:]), predictors)
            dist_arr.append(distances)
        dist_arr = np.array(dist_arr)
        # array of k nearest neighbours
        kdistances = np.argsort(dist_arr)[:k]

        
        labels = Y_train[kdistances]
        
        
        # majority 

        nearest_label =  mode(labels).mode[0]
        predicted_labels.append(nearest_label)
    # print(predicted_labels)    
    return predicted_labels
        # label = np.mode(labels)
        # print(label)
        
def accuracy(Y_train,Y_pred):
    acc = np.sum(np.equal(Y_train, Y_pred)) / len(Y_train)
    return acc

def knn():
    
    # converting catogerical values into numerical values 
    #https://www.geeksforgeeks.org/how-to-convert-categorical-variable-to-numeric-in-pandas/
    data = heart.copy()
    dummies = pd.get_dummies(data,columns=['Sex', 'ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope'])
    merged = pd.concat([data, dummies], axis='columns')

    # Dropping duplicate columns after concate
    #https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    mergedDup = merged.loc[:,~merged.columns.duplicated()]
    X = mergedDup.drop(columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope'])
    # Removing 20% of data for testing
    Xg = X.drop(columns = ['HeartDisease'] ).copy()
    Yg = X['HeartDisease']
    #Y = Yg.copy()
    X_train, X_test, Y_train, Y_test = train_test_split(Xg,Yg , test_size = 0.2, shuffle = True, random_state = 5 )
# splitting the leftover training data into  75% training and 25% validation data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size = 0.25, shuffle = True, random_state = 5)
    #Xk = X.drop(columns = ['Sex','ChestPainType','RestingECG'])
    # z= pd.concat([X_train,Y_train])
    Xg.to_csv('file_name.csv')
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    # Training knn on train set
    Y_pred_train = train(X_train,Y_train,X_train,4)

    # using validation set to find best k value
    Y_pred_valid = train(X_train, Y_train, X_valid,5)
    
    # Print training results
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_train,Y_pred_train)
    print("TRAINING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    # print validation results
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_train,Y_pred_valid)
    print("VALIDATION:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")\

    # Appending traning and validation sets
    X_train_valid = np.concatenate((X_train, X_valid))
    Y_train_valid = np.concatenate((Y_train,Y_valid))
    Y_pred_train_valid = train(X_train_valid, Y_train_valid, X_train_valid,5)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_train_valid,Y_pred_train_valid)
    print("MERGED DATA:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

    # running knn on test set
    Y_test_pred = train(X_train_valid, Y_train_valid, X_test,5)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test,Y_test_pred)
    print("TESTING:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

    


    

if __name__ == '__main__':
    knn()   



def navie_bayes():
    pass

def logistic_regression():
    pass

def decision_tree():
    pass

# 0.5303867403314917 - 7
#  0.5360824742268041 -5
# 0.5303867403314917 - 3
# 