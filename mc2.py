import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve, recall_score, confusion_matrix, roc_curve, auc, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from copy import copy
# from sklearn.stats.outliers_influence import variance_inflation_factor

class BinaryMisclassificationAccuracy(object):
    
    def __init__(self, dataframe, model_dict, target_class, split_proportion, preproc = False, flip_proportion = .05):
        
        self.original = dataframe.dropna()
        self.dataframe = dataframe.dropna()
        
        self.model_dict = model_dict
        self.model_names = list(model_dict.keys())
        self.model_list = list(model_dict.values())
        
        self.outcomes_dict = dict([(key, []) for key in self.model_names])
        
        self.target = target_class
        self.split_proportion = split_proportion
        self.classes = dataframe[target_class].unique()
        self.size = len(dataframe[target_class])
        
        if flip_proportion == .05:
            self.flip_proportion = np.arange(0,.95,flip_proportion)
            
        else:
            pass
        
        if self.size >= 2:
            class_sizes = [len(dataframe.loc[dataframe[target_class] == tar]) for tar in self.classes]
            self.proportion = [int(size)*1.0/(len(dataframe)*1.0) for size in class_sizes]
        else:
            raise ValueError('Class must have at least 2 sizes')
            
        self.imbalance = dict(zip(self.classes, self.proportion))
        
        col_list = self.dataframe.columns.tolist()
        col_list.remove(target_class)
        
        X = self.dataframe[col_list]
        y = self.dataframe[target_class]
        
        if preproc:
            self.X = scale(X)
            self.y = y
        
        else:
            self.X = X
            self.y = y
            
            
            
    def misclassify(self):
#         if len(self.X) > 5:
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.split_proportion, random_state = 5)
            self.y_train = copy(y_train)
            self.zero_indices = []
            train_list = self.y_train.tolist()

            for i in range(len(self.y_train)):

                if train_list[i] == 0:
                    self.zero_indices.append(i)
                else:
                    pass

            self.zero_indices = np.asarray(self.zero_indices)
            y_len = len(y_train)

            for i in range(len(self.model_names)):

                model_name = self.model_names[i]
                print('Beginning to fit data to: %s' % model_name)

                y_train = self.y_train

                prop_list = self.flip_proportion
                name_rep = len(self.proportion)
                names = [model_name]*name_rep

                accuracy_list = []
                roc_auc_list = []
                pr_auc_list = []
                lloss_list = []
                brier_list = []
                non_agg_dict = dict([(key,None) for key in self.flip_proportion])

                for proportion in self.flip_proportion:

                    model = copy(self.model_dict[model_name])
                    print("Proportion of mislabelled 0's: %s" % proportion)
                    y_train = copy(self.y_train)

                    y_len = len(y_train)
                    number_flipped = int(proportion*y_len)


                    if (number_flipped <= len(self.zero_indices)) and (len(y_train.unique().tolist()) != 1):
    #                     print(number_flipped,len(self.zero_indices))
                        flip_selection = np.random.choice(self.zero_indices, number_flipped, replace = False)
                        np.put(y_train, flip_selection, 1)
                        model.fit(X_train, y_train)
                        predicted_values = model.predict(X_test)
                        predicted_probs = model.predict_proba(X_test)[:,1]

                        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, predicted_probs, pos_label = 1)
                        precision, recall, thresholds_pr = precision_recall_curve(y_test, predicted_probs)

                        lloss = log_loss(y_test, predicted_probs)
                        brier = brier_score_loss(y_test, predicted_probs)
                        auc_roc = auc(fpr_roc, tpr_roc)
                        auc_pr = auc(recall, precision)

                        roc_auc_list.append(auc_roc)
                        pr_auc_list.append(auc_pr)
                        lloss_list.append(lloss)
                        brier_list.append(brier)

                        non_agg_set = {'fpr': fpr_roc, 'tpr': tpr_roc, 'thresholds_roc': thresholds_roc
                                          ,'precision': precision, 'recall': recall, 'thresholds_pr': thresholds_pr}
                        non_agg_dict[proportion] = non_agg_set

                    else:
                        break
                agg_metrics = {'misclassified_prop': prop_list, 'roc_auc': roc_auc_list
                                  , 'pr_auc': pr_auc_list, 'brier_loss': brier_list
                                  , 'log_loss': lloss_list}

                self.outcomes_dict[model_name].append(agg_metrics)

                non_agg_dict_clean = copy(non_agg_dict)

                for key in non_agg_dict_clean:

                    if non_agg_dict_clean[key] is None:

                        del non_agg_dict[key]

                    else:

                        pass

                self.outcomes_dict[model_name].append(non_agg_dict)
        except Exception as e:
            pass
#         else:   
#             pass
