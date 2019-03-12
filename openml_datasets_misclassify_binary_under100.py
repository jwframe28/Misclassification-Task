import openml
import time

openml_list = openml.datasets.list_datasets()
keys = [3, 13, 15, 24, 29, 31, 37, 43, 49, 311, 316, 336, 337, 4329, 4534]

outcomes = []

# test = keys[0:3]

num_d = 0
for i in range(len(keys)):
    try:
        print(i)
        if num_d <= 200 and (openml_list[keys[i]]['NumberOfClasses'] == 2) and (openml_list[keys[i]]['NumberOfInstances'] < 100000 and openml_list[keys[i]]['NumberOfInstances'] > 200):
            rf = RandomForestClassifier(random_state = 0, n_estimators = 50)
            model_dict1 = {'rf':rf}
            print('getting tasks')
            
            start = time.time()
                        dtasks = openml.tasks.get_task(keys[i])
            print('pulled tasks')
            X_raw, y = dtasks.get_X_and_y()
            df = pd.DataFrame(X_raw)
            df['y'] = y
            bma1 = BinaryMisclassificationAccuracy(df, model_dict1, 'y', split_proportion = .25)
            try:
                bma1.misclassify()
                outcomes.append({'name':openml_list[keys[i]]['name'],'outcomes': bma1,'NumberOfClasses':openml_list[keys[i]]['NumberOfClasses']
                             ,'NumberOfFeatures':openml_list[keys[i]]['NumberOfFeatures'],'NumberOfInstances':openml_list[keys[i]]['NumberOfInstances']
                             ,'NumberOfInstancesWithMissingValues':openml_list[keys[i]]['NumberOfInstancesWithMissingValues']
                             ,'NumberOfNumericFeatures':openml_list[keys[i]]['NumberOfNumericFeatures']
                             ,'NumberOfSymbolicFeatures':openml_list[keys[i]]['NumberOfSymbolicFeatures']})
                num_d = len(outcomes)
            except:
                pass
    except:
        pass
