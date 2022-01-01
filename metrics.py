
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    
    return TP, FP, TN, FN
  
    
def get_metrics(TP, FP, TN, FN):  
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    fpr = (FP)/(FP+TN)
    tpr = (TP)/(TP+FN)
    recall = (TP)/(TP+FN)
    precision = (TP)/(TP+FP)
    
    return accuracy, fpr, tpr, recall, precision