# aqc-textilesam
AQC Segment Anything Model for Textiles

# Installation
1. Install environment from Pipfile:
```
pipenv install
```
2. Activate the environment:
```
pipenv shell
```
# Get started
Download the [model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it at e.g (weights/sam_vit_b_01ec64.pth)

# Download train, validation and test datasets
Download images from COCO files using the download script:
```
python download --coco_file_dir path/to/coco/file
```

# Train TextileSAM
To launch the training to fine-tune the SAM using the downloaded images launch:
```
python textile_sam_train --root_path root/path/to/dataset \
--train_annotation_path path/to/train/coco/file \
--valid_annotation_path path/to/validation/coco/file \
```

There are other parameters that could be change as: number of epochs, batch size and others, please fill free to look inside the file `textile_sam_train.py`.

# Convert the model
To be able to charge the trained model using SAM, it is needed to change the format of the checkpoint. It is needed to move the best trained model to the directory weights.
```
python convert_ckpt.py
```

# Run inference
To generate all anomaly maps and masks for evaluation run:
```
python textile_sam_inference.py --test_annotation_path path/to/test/coco/file
```

# Evaluate model using mvtec evaluation script
Launch the following command:
```
  python evaluate_experiment.py --dataset_base_dir 'path/to/dataset/' \
                                --anomaly_maps_dir 'path/to/anomaly_maps/' \
                                --output_dir 'metrics/'
                                --pro_integration_limit 0.3
```

After running `evaluate_experiment.py` or `evaluate_multiple_experiments.py`,
the script `print_metrics.py` can be used to visualize all computed metrics in a
table. It requires only a single user argument:

- `metrics_folder`: The output directory specified in
  `evaluate_experiment.py` or `evaluate_multiple_experiments.py`.

for a more detailed guide please read `mvtec_evaluation/README_EVAL.md`.
