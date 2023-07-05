import argparse
import os
import glob
import json
import tqdm
import time
import imageio
from tqdm import tqdm
from vaik_video_classification_pb_inference.pb_model import PbModel

def main(input_saved_model_dir_path, input_classes_path, input_video_dir_path, output_json_dir_path):
    os.makedirs(output_json_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = PbModel(input_saved_model_dir_path, classes)

    types = ('*.avi', '*.mp4')
    video_path_list = []
    for files in types:
        video_path_list.extend(glob.glob(os.path.join(input_video_dir_path, '*', files), recursive=True))
    total_inference_time = 0
    for video_path in tqdm(video_path_list):
        # read
        video = imageio.get_reader(video_path,  'ffmpeg')
        # inference
        start = time.time()
        output, raw_pred = model.inference([frame for frame in video][::4])
        end = time.time()
        total_inference_time += (end - start)
        # dump
        output_json_path = os.path.join(output_json_dir_path, os.path.splitext(os.path.basename(video_path))[0]+'.json')

        output_dict = {}
        output_dict['inf'] = output
        output_dict['answer'] = os.path.basename(os.path.dirname(video_path))
        output_dict['video_path'] = video_path
        with open(output_json_path, 'w') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(video_path_list)/total_inference_time}[videos/sec]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_dir_path', type=str, default='~/.video-classification-pb-trainer/output_model/2023-07-05-12-06-57/step-1000_batch-8_epoch-5_loss_0.1981_sparse_categorical_accuracy_0.9427_val_loss_1.1117_val_sparse_categorical_accuracy_0.7121')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-utc101-video-classification-dataset/ucf101_labels.txt')
    parser.add_argument('--input_video_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset/train')
    parser.add_argument('--output_json_dir_path', type=str, default='~/.vaik-video-classification-pb-experiment/train_inf')
    args = parser.parse_args()

    args.input_saved_model_dir_path = os.path.expanduser(args.input_saved_model_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_video_dir_path = os.path.expanduser(args.input_video_dir_path)
    args.output_json_dir_path = os.path.expanduser(args.output_json_dir_path)

    main(**args.__dict__)