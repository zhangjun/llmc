import os
import random
import shutil

import numpy as np
import torch
from loguru import logger


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_config(config):
    if config.get('sparse', False):
        logger.info('Use sparsificatino method')
    else:

        def check_weight_setting(weight_setting):
            if weight_setting.granularity == 'per_group':
                assert weight_setting.group_size > 0
            elif weight_setting.granularity == 'per_head':
                assert weight_setting.head_num > 0

        if config.quant.weight.get('granularity', False):
            weight_setting = config.quant.weight
            check_weight_setting(weight_setting)
        if config.quant.weight.get('w_1', False):
            weight_setting = config.quant.weight.w_1
            check_weight_setting(weight_setting)
        if config.quant.weight.get('w_2', False):
            weight_setting = config.quant.weight.w_2
            check_weight_setting(weight_setting)
    if 'eval' in config and 'fake_quant' in config.eval.eval_pos:
        if 'save' in config:
            assert not config.save.get(
                'save_quant', False
            ), 'Fake_quant—eval and save_quant conflict now.'
            assert not (
                config.save.get('save_fake', False)
                and config.save.get('save_quant', False)
            ), 'Saving fake quant and saving real quant conflict now.'
    if config.model.get('tokenizer_mode', False):
        assert (
            config.model.tokenizer_mode == 'slow'
            or config.model.tokenizer_mode == 'fast'
        ), 'Tokenizer_mode should be slow or fast.'
        logger.info(f'Tokenizer_mode is set to {config.model.tokenizer_mode}.')
    else:
        config.model.tokenizer_mode = 'slow'
        logger.info('Tokenizer_mode is set to slow.')

    if 'calib' in config and not config.calib.get('type', False):
        config.calib.type = 'txt'
    if 'eval' in config and not config.eval.get('type', False):
        config.eval.type = 'ppl'


def mkdirs(path):
    os.makedirs(path,  exist_ok=True)


def copy_files(source_dir, target_dir, substring):
    for filename in os.listdir(source_dir):
        if substring in filename:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.copy(source_file, target_file)
            logger.info(f'Copied {filename} to {target_dir}')


def print_important_package_version():
    from importlib.metadata import version
    logger.info(f"torch : {version('torch')}")
    logger.info(f"transformers : {version('transformers')}")
    logger.info(f"tokenizers : {version('tokenizers')}")
    logger.info(f"huggingface-hub : {version('huggingface-hub')}")
    logger.info(f"datasets : {version('datasets')}")
