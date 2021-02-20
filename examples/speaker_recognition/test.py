# Experiment: restore model and Extract Speaker Embeddings

import os

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ExtractSpeakerEmbeddingsModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import torch
import nemo.collections.asr as nemo_asr

from omegaconf import OmegaConf

############################################################
## !head -5 {validation_manifest} > embeddings_manifest.json
############################################################

os.environ["OMP_NUM_THREADS"] = '1'


# restore model 
model_path = '/Users/xujinghua/NeMo/nemo_experiments/SpeakerNet/spkr.nemo'
restored_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)

cuda = 1 if torch.cuda.is_available() else 0


# extract embeddings
test_config = OmegaConf.create(dict(
    manifest_filepath = '/Users/xujinghua/NeMo/embeddings_manifest.json',
    sample_rate = 16000,
    labels = None,
    batch_size = 1,
    shuffle = False,
    time_length = 8,
    embedding_dir='/Users/xujinghua/NeMo'
))

print(OmegaConf.to_yaml(test_config))
verification_model.setup_test_data(test_config)

trainer = pl.Trainer(gpus=cuda,accelerator=None)
test_result = trainer.test(verification_model, verbose = True)
# , ckpt_path='/Users/xujinghua/NeMo/embeddings_manifest.json')

print(test_result)

for k, v in test_result:
    print('--------------')
    print(k)
    print(v)
    print('--------------')
    