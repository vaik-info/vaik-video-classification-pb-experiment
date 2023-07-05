import argparse
import os
import glob
import json
import collections
import itertools
from sklearn import metrics

def calc_acc(json_dict_list, classes):
    pred_list = []
    gt_list = []
    for json_dict in json_dict_list:
        classes_score = {}
        for inf_json_dict in json_dict['inf']:
            for label, score in zip(inf_json_dict['label'], inf_json_dict['score']):
                classes_score[label] = score
        inf_class_label = max(classes_score.items(), key=lambda x: x[1])[0]
        pred_list.append(classes.index(inf_class_label))
        gt_list.append(classes.index(json_dict['answer']))
    print(metrics.classification_report(gt_list, pred_list, target_names=classes, digits=4))


def main(input_json_dir_path, input_classes_path):
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    calc_acc(json_dict_list, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default='~/.vaik-video-classification-pb-experiment/test_inf')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt')
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)

    main(**args.__dict__)