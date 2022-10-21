# Change dataloader multiprocess start method to anything not fork
import functools
import glob
import logging
import os
import sys
import time
import inspect

import numpy as np
from MinkowskiEngine.MinkowskiInterpolation import MinkowskiInterpolation
from MinkowskiEngine.MinkowskiPooling import MinkowskiMaxPooling
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
sys.path.append(os.path.dirname(inspect.getfile(SparseTensor)))
from MinkowskiTensor import SparseTensorQuantizationMode
from MinkowskiEngine.MinkowskiTensorField import TensorField
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import random
import string
import lib.transforms as t

# Torch packages
import torch

# Train deps
from config.config import get_config

from lib.utils import load_state_with_same_shape, count_parameters, visualize_results
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.voxelizer import Voxelizer

from models import load_model, load_wrapper

import MinkowskiEngine as ME



# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def randStr(chars = string.ascii_lowercase + string.digits, N=10):
    return ''.join(random.choice(chars) for _ in range(N))


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def inspect_layer_output(model, layer_path, storage_dict: dict, name: str = None, unsqueeze: bool = True, index_to_inspect: int = None, skip_if_no_grad: bool = True):
    if name is None:
        name = layer_path

    def hook(model, input, output):
        if skip_if_no_grad and not torch.is_grad_enabled():
            return

        if index_to_inspect is not None:
            output = output[index_to_inspect]
        if unsqueeze:
            output = output.unsqueeze(0)
        if name in storage_dict:
            storage_dict[name] = torch.cat((storage_dict[name], output), dim=0)
        else:
            storage_dict[name] = output

    return rgetattr(model, layer_path).register_forward_hook(hook)


class CleanCacheCallback(Callback):

    def training_step_end(self, trainer):
        torch.cuda.empty_cache()

    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def validation_step_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


def main():
    config = get_config()

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    DatasetClass = load_dataset(config.dataset)

    logging.info('===> Initializing dataloader')

    data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        num_workers=config.num_workers,
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints,
        collate_function=t.cfl_collate_fn_factory)

    if data_loader.dataset.NUM_IN_CHANNEL is not None:
        num_in_channel = data_loader.dataset.NUM_IN_CHANNEL
    else:
        num_in_channel = 3  # RGB color

    num_labels = data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')
    NetClass = load_model(config.model)

    if config.wrapper_type == 'None':
        model = NetClass(num_in_channel, num_labels, config)
        logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model)))
    else:
        wrapper = load_wrapper(config.wrapper_type)
        model = wrapper(NetClass, num_in_channel, num_labels, config)

        logging.info('===> Number of trainable parameters: {}: {}'.format(
            wrapper.__name__ + NetClass.__name__, count_parameters(model)))

    # Load weights if available
    if not (config.weights == 'None' or config.weights is None):
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        if config.weights_for_inner_model:
            model.model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                if 'pth' in config.weights:  # CSC version of model state
                    matched_weights = load_state_with_same_shape(model, state['state_dict'], prefix='')
                else:  # Lightning
                    matched_weights = load_state_with_same_shape(model, state['state_dict'], prefix='model.')

                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])

    # Sync bathnorm for multiple GPUs
    if config.num_gpu > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    log_folder = config.log_dir
    num_devices = min(config.num_gpu, torch.cuda.device_count())
    logging.info('Starting training with {} GPUs'.format(num_devices))

    checkpoint_callbacks = [pl.callbacks.ModelCheckpoint(
        dirpath=config.log_dir,
        monitor="val_miou",
        mode='max',
        filename='checkpoint-{val_miou:.2f}-{step}',
        save_top_k=1,
        every_n_epochs=1)]

    # Set wandb project attributes
    wandb_id = randStr()
    version_num = 0
    if config.resume:
        directories = glob.glob(config.resume + '/default/*')
        versions = [int(dir.split('_')[-1]) for dir in directories]
        list_of_ckpts = glob.glob(config.resume + '/*.ckpt')

        if len(list_of_ckpts) > 0:
            version_num = max(versions) if len(versions) > 0 else 0
            ckpt_steps = np.array([int(ckpt.split('=')[1].split('.')[0]) for ckpt in list_of_ckpts])
            latest_ckpt = list_of_ckpts[np.argmax(ckpt_steps)]
            config.resume = latest_ckpt
            state_params = torch.load(config.resume)['hyper_parameters']

            if 'wandb_id' in state_params:
                wandb_id = state_params['wandb_id']
        else:
            config.resume = None
        print('Resuming: ', config.resume)
    config.wandb_id = wandb_id

    # Import the correct trainer module
    if config.use_embedding_loss and config.use_embedding_loss != 'both':
        from lib.train_test.pl_RepresentationTrainer import RepresentationTrainerModule as TrainerModule

        # we only have representation losses here
        checkpoint_callbacks += [pl.callbacks.ModelCheckpoint(
            dirpath=config.log_dir,
            monitor="val_loss",
            mode='min',
            filename='checkpoint-{val_loss:.5f}-{step}',
            save_top_k=1,
            every_n_epochs=1)]
    else:
        if 'Classifier' in config.model:
            from lib.train_test.pl_ClassifierTrainer import ClassifierTrainerModule as TrainerModule
        else:
            from lib.train_test.pl_BaselineTrainer import BaselineTrainerModule as TrainerModule

    # Init loggers
    tensorboard_logger = TensorBoardLogger(log_folder, default_hp_metric=False, log_graph=True, version=version_num)
    run_name = config.model + '-' + config.dataset if config.is_train else config.model + "_test"



    if config.cache:
        config.val_batch_size = 1

        pl_module = TrainerModule(model, config, data_loader.dataset)
        dataloader = pl_module.val_dataloader()

        visualize_path = config.visualize_path
        for scene_idx, data in enumerate(dataloader):
            if config.scene_idx is not None and scene_idx != config.scene_idx:
                continue

            relative_room_folder = data_loader.dataset.data_paths[scene_idx]
            config.visualize_path = os.path.join(visualize_path, relative_room_folder)

            outputs = pl_module.model_step(batch=data, batch_idx=scene_idx, mode='validation')
            outputs = pl_module.eval_step(outputs)

            class_ids = np.arange(pl_module.num_labels)

            label_mapper = lambda t: pl_module.dataset.inverse_label_map[t]
            target = outputs['final_target'].cpu().apply_(label_mapper)
            pred = outputs['final_pred'].cpu().apply_(label_mapper)
            feat = outputs['feature_maps'].F.cpu()
            invalid_parents = target == pl_module.config.ignore_label
            pred[invalid_parents] = pl_module.config.ignore_label

            # visualize_results(coords=outputs['coords'], colors=outputs['colors'], target=target,
            #                   prediction=pred, config=pl_module.config, iteration=pl_module.global_step,
            #                   num_labels=pl_module.num_labels, train_iteration=pl_module.global_step,
            #                   valid_labels=class_ids, save_npy=True,
            #                   scene_name=outputs['scene_name'],
            #                   name_prefix='eval')

            # Get predictions for all voxel centers at a coarser resolution
            pred_tensor = TensorField(features=pred[:, None], coordinates=outputs['coords'])

            fine_coords = data[0]
            # Create 20cm grid
            coarse_coords = torch.floor(fine_coords / 10)
            # coarse_coords = coarse_coords + coarse_coords.min(dim=0, keepdim=True)[0]

            target_tensor = SparseTensor(features=target[:, None].to(torch.float),
                                         coordinates=coarse_coords,
                                         quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            prediction_tensor = SparseTensor(features=pred[:, None].to(torch.float),
                                             coordinates=coarse_coords,
                                             quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            feature_tensor = SparseTensor(features=feat.to(torch.float),
                                             coordinates=coarse_coords,
                                             quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            coarse_coords = prediction_tensor.C
            coarse_pred = (prediction_tensor.F > 0.5).to(torch.int)
            coarse_feat = feature_tensor.F
            coarse_target = (target_tensor.F > 0.5).to(torch.int)

            # Convert to Mitsuba convention
            coarse_coords = torch.index_select(coarse_coords.cpu(), 1, torch.LongTensor([0, 1, 3, 2])) * torch.Tensor([[1, 1, 1, -1]])
            coarse_coords = coarse_coords.to("cuda")
            coarse_coords -= coarse_coords.min(dim=0, keepdim=True)[0]

            visualize_results(coords=coarse_coords,
                              colors=torch.zeros((len(coarse_target), 3), device=coarse_coords.device),
                              target=coarse_target[:, 0],
                              prediction=coarse_pred[:, 0], config=pl_module.config, iteration=pl_module.global_step,
                              num_labels=pl_module.num_labels, train_iteration=pl_module.global_step,
                              valid_labels=class_ids, save_npy=True,
                              scene_name=outputs['scene_name'],
                              name_prefix='coarse')

            # Save the features

            prediction = coarse_feat.detach()
            target = coarse_target[:, 0]
            input_xyz = coarse_coords[:, 1:]
            target_batch = (coarse_coords[:, 0] == 0).detach().cpu()
            target_valid = torch.ne(target, config.ignore_label).detach()
            batch_ids = torch.logical_and(target_batch, target_valid)
            target_nonpred = torch.logical_and(target_batch, ~target_valid)  # type: torch.Tensor

            ptc_nonpred_np = np.hstack(
                (input_xyz[target_nonpred].cpu().numpy(),
                 np.zeros((torch.sum(target_nonpred).item(), 1))))  # type: np.ndarray

            input_xyz_np = input_xyz[batch_ids].cpu().numpy()
            xyzlabel_np_pred = np.hstack((input_xyz_np, prediction[batch_ids.detach().numpy()]))  # type: np.ndarray

            np.save(os.path.join(config.visualize_path, "features.npy"), xyzlabel_np_pred)
            print("Scene done")
        print("Done")





    else:
        # Try a few times to avoid init error based on connection
        # loggers = [tensorboard_logger]
        loggers = []
        while True:
            try:
                wandb_logger = WandbLogger(project="lg_semseg", name=run_name, log_model=False, id=config.wandb_id,
                                           save_dir=config.wandb_logdir)
                loggers += [wandb_logger]
                break
            except:
                print("Retrying WanDB connection...")
                time.sleep(10)

        trainer = Trainer(max_epochs=config.max_epoch, logger=loggers,
                          devices=num_devices, accelerator="gpu", strategy=DDPPlugin(find_unused_parameters=True),
                          num_sanity_val_steps=4, accumulate_grad_batches=1,
                          callbacks=[*checkpoint_callbacks, CleanCacheCallback()],
                          check_val_every_n_epoch=config.val_freq)

        pl_module = TrainerModule(model, config, data_loader.dataset)
        if config.is_train:
            trainer.fit(pl_module, ckpt_path=config.resume)
        else:
            trainer.test(pl_module, ckpt_path=config.resume)


if __name__ == '__main__':
    __spec__ = None
    main()
