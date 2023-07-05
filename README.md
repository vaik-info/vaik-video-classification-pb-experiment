# vaik-video-classification-pb-experiment
Create json file by video classification model. Calc Top 3 ACC.

## Install

```shell
pip install -r requirements.txt
```

## Usage

### Create json file

```shell
python inference.py --input_saved_model_dir_path '~/.vaik-video-classification-pb-trainer/output_model/2023-07-05-10-34-29/step-1000_batch-8_epoch-10_loss_0.5479_sparse_categorical_accuracy_0.8443_val_loss_1.7989_val_sparse_categorical_accuracy_0.5060' \
                --input_classes_path '~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt' \
                --input_image_dir_path '~/.vaik-utc101-video-classification-dataset/test' \
                --output_json_dir_path '~/.vaik-video-classification-pb-experiment/test_inf'
```

- input_video_dir_path
    - example

```shell
.
.
├── test
│   ├── ApplyEyeMakeup
│   │   ├── ApplyEyeMakeup_1007.mp4
│   │   ├── ApplyEyeMakeup_1014.mp4
│   │   ├── ApplyEyeMakeup_1138.mp4
・・・
│   ├── ApplyLipstick
│   │   ├── ApplyLipstick_1161.mp4
│   │   ├── ApplyLipstick_1235.mp4
│   │   ├── ApplyLipstick_1342.mp4
・・・
```

#### Output
- output_json_dir_path
    - example

```json
{
  "answer": "ApplyEyeMakeup",
  "inf": [
    {
      "end_frame": 16,
      "label": [
        "ApplyLipstick",
        "BlowingCandles",
        "BoxingSpeedBag",
        "CuttingInKitchen",
        "BlowDryHair",
        "ApplyEyeMakeup",
        "BrushingTeeth",
        "Archery",
        "Bowling",
        "BabyCrawling",
        "BoxingPunchingBag",
        "Billiards",
        "CleanAndJerk",
        "BenchPress",
        "Biking",
        "Basketball",
        "CricketShot",
        "CliffDiving",
        "BodyWeightSquats",
        "BalanceBeam",
        "BasketballDunk",
        "BreastStroke",
        "BandMarching",
        "CricketBowling",
        "BaseballPitch"
      ],
      "score": [
        5.99818229675293,
        5.331709861755371,
        4.273311138153076,
        4.086155891418457,
        3.075073480606079,
        2.8466880321502686,
        2.111257314682007,
        1.962202548980713,
        1.843916893005371,
        1.5590394735336304,
        1.408070683479309,
        0.3319084644317627,
        -0.5513471961021423,
        -0.6263744235038757,
        -1.3993675708770752,
        -2.122079372406006,
        -2.80260968208313,
        -2.889033555984497,
        -3.2077112197875977,
        -3.549764394760132,
        -3.8684756755828857,
        -4.0212297439575195,
        -5.728559970855713,
        -5.763431072235107,
        -7.228459358215332
      ],
      "start_frame": 0
    },
・・・
  ],
  "video_path": "~/.vaik-utc101-video-classification-dataset/test/ApplyEyeMakeup/ApplyEyeMakeup_142.avi"
}
```
-----

### Calc ACC

```shell
python calc_top3_acc.py --input_json_dir_path '~/.vaik-video-classification-pb-experiment/test_inf' \
                --input_classes_path '~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt'
```

#### Output

``` text
                   precision    recall  f1-score   support

   ApplyEyeMakeup     0.5000    0.0227    0.0435        44
    ApplyLipstick     0.0000    0.0000    0.0000        32
          Archery     0.1818    0.0488    0.0769        41
     BabyCrawling     0.1739    0.1143    0.1379        35
      BalanceBeam     0.4706    0.5161    0.4923        31
     BandMarching     0.7500    0.3488    0.4762        43
    BaseballPitch     0.7000    0.1628    0.2642        43
       Basketball     0.6216    0.6571    0.6389        35
   BasketballDunk     0.8261    0.5135    0.6333        37
       BenchPress     0.8750    0.1458    0.2500        48
           Biking     0.4211    0.2105    0.2807        38
        Billiards     1.0000    0.4250    0.5965        40
      BlowDryHair     0.3333    0.2368    0.2769        38
   BlowingCandles     0.1029    0.6364    0.1772        33
 BodyWeightSquats     0.0000    0.0000    0.0000        30
          Bowling     0.4038    0.4884    0.4421        43
BoxingPunchingBag     0.8750    0.1429    0.2456        49
   BoxingSpeedBag     0.5000    0.0270    0.0513        37
     BreastStroke     0.8333    0.1786    0.2941        28
    BrushingTeeth     0.1667    0.0556    0.0833        36
     CleanAndJerk     0.5000    0.0303    0.0571        33
      CliffDiving     0.0816    0.7949    0.1480        39
   CricketBowling     0.4444    0.2222    0.2963        36
      CricketShot     0.6154    0.1633    0.2581        49
 CuttingInKitchen     0.1905    0.1212    0.1481        33

         accuracy                         0.2492       951
        macro avg     0.4627    0.2505    0.2547       951
     weighted avg     0.4827    0.2492    0.2584       951

```