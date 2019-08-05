import os
import json

# make dictionary with parameters specific to all datasets
data = {}
data['isic_2017'] = {
  'classes': ['melanoma', 'nevus_sk'],
  'orig_path': 'ISIC-2017',
  'dataset_path': 'dataset_ISIC',
  'output_path': 'output_ISIC'
  }

# list with all datasets; so certain paths can be added automatically to the data dictionary
datasets = ['isic_2017']
for dataset in datasets:
    # path to original dataset, use parent path so data is not in same repo as code
    parent_path = os.path.dirname(os.getcwd())
    orig_data_path = os.path.join(parent_path, 'Data/{}'.format(data[dataset]['orig_path']))
    data[dataset]['orig_data_path'] = orig_data_path

    # base path after splitting data
    dataset_path = os.path.join(parent_path, 'datasets/{}'.format(data[dataset]['dataset_path']))
    data[dataset]['dataset_path'] = dataset_path

    # path to save plots
    plot_path = os.path.join(parent_path, 'outputs/{}/plots'.format(data[dataset]['output_path']))
    data[dataset]['plot_path'] = plot_path

    # path to save trained model
    model_savepath = os.path.join(parent_path, 'outputs/{}/models'.format(data[dataset]['output_path']))
    data[dataset]['model_savepath'] = model_savepath

    # data split paths
    trainingpath = os.path.join(dataset_path, 'training')
    data[dataset]['trainingpath'] = trainingpath
    validationpath = os.path.join(dataset_path, 'validation')
    data[dataset]['validationpath'] = validationpath
    testpath = os.path.join(dataset_path, 'test')
    data[dataset]['testpath'] = testpath

# create json configuration file
with open('config.json', 'w') as f:
    json.dump(data, f)
