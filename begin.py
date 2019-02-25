import openml

openml_list = openml.datasets.list_datasets()
keys = list(openml_list.keys())

outcomes = []

# test = keys[0:3]
for i in range(len(keys)):
# for i in range(len(keys)-(len(keys)-5)):
    try:
        print(i)
        if openml_list[keys[i]]['NumberOfClasses'] == 2:
            rf = RandomForestClassifier(random_state = 0, n_estimators = 50)
            model_dict1 = {'rf':rf}
            print('getting tasks')
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
            except:
                pass
    except:
        pass #what are all of these errors?, some things are appended empty
