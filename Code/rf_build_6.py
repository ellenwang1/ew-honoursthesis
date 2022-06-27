from sklearn.metrics import accuracy_score

def best_number_of_trees(dataset, Y_train):
    accuracy_list = []
    error_list = []
    no_trees = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
    for i in no_trees:
        from sklearn.ensemble import RandomForestClassifier
        #max features, min sample split 
        clf = RandomForestClassifier(n_estimators = i, max_features = 'sqrt', min_samples_split = 2, criterion = 'gini', oob_score = True, bootstrap = True, random_state = 30)
        clf.fit(dataset, Y_train)
        y_pred = (clf.oob_decision_function_[:,1] >= 0.50).astype(bool)
        accuracy = accuracy_score(Y_train, y_pred)
        accuracy_list.append(accuracy)
    for i in accuracy_list:
        error_list.append(1-i)
    return accuracy_list, error_list