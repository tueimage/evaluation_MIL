import yaml
import argparse

from stability.stability_all_classifiers.multi_classifier_stability import stability_all_classifiers, \
    stability_all_classifiers_instance_level, stability_all_classifiers_instance_level_pascal


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)
xray_dataset = config['use_xray_dataset']
use_pascal_dataset = config['use_pascal_dataset']

set_name1 ='shoulder_test_set_CV0_4shoulder_0.95.npy'
set_name2 = 'shoulder_test_set_CV0_0shoulder_0.95.npy'
set_name3 ='shoulder_test_set_CV0_3shoulder_0.95.npy'
set_name4 = 'shoulder_test_set_CV0_2shoulder_0.95.npy'
set_name5 ='shoulder_test_set_CV0_1shoulder_0.95.npy'


classifiers = [set_name1, set_name2, set_name3, set_name4, set_name5]

if xray_dataset:
    stability_all_classifiers(config, classifiers, only_segmentation_images=False, only_positive_images=True,
                              visualize_per_image= False)
    stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images=True, only_positive_images=False)


else:
    stability_all_classifiers(config, classifiers, only_segmentation_images=False, only_positive_images=True,
                              visualize_per_image=False)
    if use_pascal_dataset:
        stability_all_classifiers_instance_level_pascal(config, classifiers)
