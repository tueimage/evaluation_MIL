def check_key(config_keys, real_keys):
    for key in real_keys:
        assert key in config_keys, f'Define "{key}" in the configuration file.'
        
def check_type(config_keys, real_keys, config, real):
    for key in real_keys:
        assert type(config[key]) in real[key]


def verify_input(config):
    print('Verifying input file...')
      
    vars_template_main = {
        'dataset': 'str',
        'mode': 'str',
        'project_path': 'str',
        'gpu': 'str'   
    }
    
    vars_template_prepare = {
        'skip_processing_labels': 'bool',
        'image_path': 'str',
        'test_image_path': 'str',
        'labels_path': 'str',
        'test_labels_path': 'str',
        'localization_labels_path': 'str',

        'class_name': 'str',
        'subset_size': 'float, int',
        'number_of_subsets': 'int',
    }
    
    vars_template_train = {
        'nr_epochs': 'int',
        'lr': 'float, int',
        'reg_weight': 'float, int',
        'pooling_operator': 'str',
        'mura_interpolation': 'bool',
        'prediction_results_path': 'str',
        'trained_models_path': 'str',
        'skip_processing_labels': 'bool',
    }
    
    vars_template_test = {
        'results_path': 'str',
        'prediction_results_path': 'str',
        'trained_models_path': 'str',
        'model_name': 'str',
    }
    
    config_keys   = list(config.keys())
    template_keys = list(vars_template_main.keys())
    
    check_key(config_keys, template_keys)
    
    if config['mode'] == 'prepare':
        template_keys = list(vars_template_prepare.keys())
        check_key(config_keys, template_keys)
    
    if config['mode'] == 'train':
        template_keys = list(vars_template_train.keys())
        check_key(config_keys, template_keys)
        
    if config['mode'] == 'test':
        template_keys = list(vars_template_test.keys())
        check_key(config_keys, template_keys)
    