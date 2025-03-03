from box import Box

config = {
    # "num_devices": 2,
    "batch_size": 4,
    "num_workers": 4,
    "out_dir": "total_model/monuseg",
    "opt": {
        "num_epochs": 100,
        "learning_rate": 2e-4, #1e-4
        "weight_decay": 1e-2, #1e-2,
        "precision": 32, # "16-mixed"
        "steps":  [37*50, 37 * 70],
        "warmup_steps": 37*10,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "image_encoder/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,

        "extra_encoder": None,
        "extra_type": "plus",
        # "extra_encoder": "effunet",
        # "extra_type": "fusion",
        "extra_checkpoint": "extra_encoder/efficient_supervised_best_encoder_monuseg.pth",
    },
    "loss": {
        "focal_cof": 0.125,
        "dice_cof": 0.875,
        "ce_cof": 0.,
        "iou_cof": 0.00,
    },
    "dataset": {
        "dataset_root": "dataset/monuseg",
        "dataset_csv_path": "dataset_cfg/experiment_monuseg.csv",
        "data_ext": ".png",
        "val_fold_id": 0,
        "num_classes": 2,

        "ignored_classes": None,
        "ignored_classes_metric": 1, # if we do not count background, set to 1 (bg class)
        "image_hw": (1024, 1024), # default is 1024, 1024

        "feature_input": False, # or "True" for *.pt features
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std": (0.229, 0.224, 0.225),
    }
}

cfg = Box(config)