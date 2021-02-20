# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Task 1: Speech Command

## Preparing the dataset
Use the `process_speech_commands_data.py` script under <NEMO_ROOT>/scripts in order to prepare the dataset.

```sh
python <NEMO_ROOT>/scripts/process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset> \
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --rebalance \
    --log
```

## Train to convergence
```sh
python speech_to_label.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath=["<path to val manifest>","<path to test manifest>"] \
    trainer.gpus=2 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-v1" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-v1" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6
```


# Task 2: Voice Activity Detection

## Preparing the dataset
Use the `process_vad_data.py` script under <NEMO_ROOT>/scripts in order to prepare the dataset.

```sh
python process_vad_data.py \
    --out_dir=<output path to where the generated manifest should be stored> \
    --speech_data_root=<path where the speech data are stored> \
    --background_data_root=<path where the background data are stored> \
    --rebalance_method=<'under' or 'over' of 'fixed'> \
    --log
    (Optional --demo (for demonstration in tutorial). If you want to use your own background noise data, make sure to delete --demo)
```

## Train to convergence
```sh
python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "conf">
    --config-name=<name of config without .yaml e.g. "matchboxnet_3x1x64_vad"> \
    model.train_ds.manifest_filepath="<path to train manifest>" \
    model.validation_ds.manifest_filepath=["<path to val manifest>","<path to test manifest>"] \
    trainer.gpus=2 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-vad" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-vad" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6
```

# Optional: Use tarred dataset to speed up data loading. Apply to both tasks.
## Prepare tarred dataset. 
   Prepare ONE manifest that contains all training data you would like to include. Validation should use non-tarred dataset.
   Note that it's possible that tarred datasets impacts validation scores because it drop values in order to have same amount of files per tarfile; 
   Scores might be off since some data is missing. 

   Use the `convert_to_tarred_audio_dataset.py` script under <NEMO_ROOT>/scripts in order to prepare tarred audio dataset.
   For details, please see TarredAudioToClassificationLabelDataset in <NEMO_ROOT>/nemo/collections/asr/data/audio_to_label.py

python speech_to_label.py \
    --config-path=<path to dir of configs e.g. "conf">
    --config-name=<name of config without .yaml e.g. "matchboxnet_3x1x64_vad"> \
    model.train_ds.manifest_filepath=<path to train tarred_audio_manifest.json> \
    model.train_ds.is_tarred=True \
    model.train_ds.tarred_audio_filepaths=<path to train tarred audio dataset e.g. audio_{0..2}.tar> \
    +model.train_ds.num_worker=<num_shards used generating tarred dataset> \
    model.validation_ds.manifest_filepath=<path to validation audio_manifest.json>\
    trainer.gpus=2 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=200 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-vad" \
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-vad" \
    +trainer.precision=16 \
    +trainer.amp_level=O1  # needed if using PyTorch < 1.6

"""

# for speech commands recognitioin, first run `process_speech_commands_data.py` to download data and prepare manifest files
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="/Users/xujinghua/NeMo/examples/asr/conf", config_name="matchboxnet_3x1x64_v1")
def main(cfg):

    print(OmegaConf.to_yaml(cfg))

    # update config with manifest file paths
    cfg.model.train_ds.manifest_filepath = '/Users/xujinghua/google_speech_recognition_v1/train_manifest.json'
    cfg.model.validation_ds.manifest_filepath = '/Users/xujinghua/google_speech_recognition_v1/validation_manifest.json'
    cfg.model.test_ds.manifest_filepath = '/Users/xujinghua/google_speech_recognition_v1/test_manifest.json'
    
    # print(OmegaConf.to_yaml(cfg))
    
    # Preserve some useful parameters
    # labels = cfg.model.labels
    # sample_rate = cfg.sample_rate

    # for label in labels:
        # print(label)

        # print(sample_rate)

    '''
    cfg.trainer.gpus=2 
    cfg.trainer.accelerator="ddp" 
    cfg.trainer.max_epochs=200 
    cfg.trainer.precision=16 
    
    exp_manager.create_wandb_logger=True 
    exp_manager.wandb_logger_kwargs.name="MatchboxNet-3x1x64-v1" 
    exp_manager.wandb_logger_kwargs.project="MatchboxNet-v1" 
    '''

    # colab tutorial default setting
    # Lets modify some trainer configs for this demo
    # Checks if we have GPU available and uses it
    # cuda = 1 if torch.cuda.is_available() else 0
    # cfg.trainer.gpus = cuda

    # Reduces maximum number of epochs to 5 for quick demonstration
    # cfg.trainer.max_epochs = 5

    # Remove distributed training flags
    # cfg.trainer.accelerator = None
            
    # setting up trainer
    # trainer = pl.Trainer(**cfg.trainer)

    # fast training with flags

    # cfg.trainer.max_epochs=200 

    # all flags to speed up the process
    # trainer = pl.Trainer(**cfg.trainer, amp_level='O1', precision=16, gpus=2, num_nodes=2, accelerator='ddp')

    # Trainer with a distributed backend:
    # trainer = Trainer(gpus=2, num_nodes=2, accelerator='ddp')

    # setting up experiment manager
    # The exp_dir provides a path to the current experiment for easy access
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))


    # build a MatchboxNet model
    asr_model = EncDecClassificationModel(cfg=cfg.model, trainer=trainer)
    # train a MatchboxNet model
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(gpus=gpu)
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)
    


if __name__ == '__main__':
    # config = OmegaConf.load(config_path)
    # config_path="/Users/xujinghua/NeMo/examples/asr/conf"
    main()  # noqa pylint: disable=no-value-for-parameter
