from flax.serialization import from_bytes
from ml_collections import ConfigDict, FrozenConfigDict
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax
import jax
import os
import tensorflow_datasets as tfds
from flax.training import common_utils



from train_parallel import create_train_state
from src.transformer.input_pipeline import get_data_from_tfds
from src.utilities.visualisation import plot_predictions, plot_delta, plot_loss, plot_fields, plotgrayscale
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.pressure_preprocesing import *
from src.utilities.de_normalization import Denormalize


def errorcal(config, test, pred):
    print(np.shape(pred))
    error = (jnp.sqrt(jnp.sum((test-pred)**2))) / np.sum(np.shape(test))
    return error
    

def load_state(statefile, state_dict):
    # Load the serialized bytes from the file
    with open(statefile, 'rb') as f:
        bytes_data = f.read()
    # Deserialize the bytes to obtain the model parameters
    state = from_bytes(state_dict, bytes_data)
    return state


@partial(jax.pmap, axis_name='num_devices')
def test_step(state, x: jnp.ndarray, y: jnp.ndarray):
    preds = state.apply_fn({'params': state.params}, x, y, train=False)

    loss = optax.squared_error(preds, y).mean()
    loss = jax.lax.psum(loss, axis_name='num_devices')
    
    return preds, loss



@jax.jit
def masking(encoder, resultant, internal_value, postprocessed_internal_value):
    shape = resultant.shape
    #print(shape)
    encoder = encoder.flatten(order='C')
    mask = (encoder != internal_value)
    #mask1 = jnp.where(mask == False, postprocessed_internal_value, 1)
    #internal_value_idx = jnp.where(mask)[0]
    resultant = resultant.reshape(-1,3).transpose()
    for i in range(shape[-1]):
        array = jnp.copy(resultant[i]) * mask # .at[internal_value_idx].set(postprocessed_internal_value))
        array = jnp.where(array == internal_value, postprocessed_internal_value, array)
        resultant = resultant.at[i].set(array)
    resultant = resultant.transpose()
    resultant = resultant.reshape(shape)
    return resultant


def post_process_state(config: ConfigDict):
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, jax.local_device_count())
    rng_idx = np.random.default_rng(0)


    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=1
    )

    state_dict = create_train_state(rng, FrozenConfigDict(config), lr_scheduler)

    vectorized_masking = jax.vmap(masking, in_axes=(0,0,None,None))
    comm = []
    state = load_state(os.path.join(config.output_dir,'model_state.jax'), state_dict)
    if config.vit.img_size[0] == 256:
        mode = 'test'
    else:
        mode = 'train'
    ds_test = get_data_from_tfds(config=config, mode=mode)
    #ds_test = get_data_from_tfds(config=config, mode='train')
    test_log = []
    idx1 = -1
    idx2 = -1
    for test_batch in tfds.as_numpy(ds_test):
        decoder_test_batch = test_batch['decoder']
                
        if config.internal_geometry.set_internal_value:
            test_batch = set_geometry_internal_value(batch,config.internal_geometry.value)

        if config.pressure_preprocessing.enable :
            test_batch = pressure_preprocessing(test_batch, config)
        label = test_batch.pop('label')
        #b'4535_6_0.2' or b'5220_9_0.1'
        if config.vit.img_size[0] == 128:
            if b'4120_-4_0.2' not in label:
                continue
            else:
                idx = np.where(label == b'4120_-4_0.2')
                print(idx[0])
                idx1 = int(idx[0]/6)
                idx2 = int(idx[0] - (idx1*6))
                print(idx1)
                print(idx2)
        x_test = common_utils.shard(test_batch['encoder'])
        y_test = common_utils.shard(test_batch['decoder'])
                
        #label = test_batch.pop('label')
        #print(label)
        comm.append(label)
        
        preds, test_loss = test_step(state, x_test, y_test)
        test_log.append(test_loss)
        break

    test_loss = np.mean(test_log)

    #comm = np.array(comm).flatten()
    #csv_file = 'my_list256.txt'
    # Writing the list to a CSV file
    #with open(csv_file, 'w') as file:
    #    for item in comm:
    #        file.write(f"{item}\n")


    if config.denormalization.denormalize:
        label_data = process_label(label)
        mach = jnp.array(label_data)
        preds, y_test = Denormalize(config, preds.reshape(config.batch_size,config.vit.img_size[0],config.vit.img_size[1],-1), decoder_test_batch, mach)



   
    #internal_postProcessed_value = config.internal_geometry.postProcessed_value
    if config.internal_geometry.set_postProcessed_internal_value:
        preds = vectorized_masking(x_test, common_utils.shard(preds), config.internal_geometry.value, config.internal_geometry.postProcessed_value)
        y_test = vectorized_masking(x_test, common_utils.shard(y_test), config.internal_geometry.value, config.internal_geometry.postProcessed_value)
    preds = preds.reshape(4, int(config.batch_size/4), config.vit.img_size[0], config.vit.img_size[1], -1)
    y_test = y_test.reshape(4, int(config.batch_size/4), config.vit.img_size[0], config.vit.img_size[1], -1)
    error = errorcal(config, y_test[0][0].flatten(), preds[0][0].flatten())
    print(error)
    pred_data = preds[idx1,idx2,:,:,:].squeeze()
    test_data = y_test[idx1,idx2,:,:,:].squeeze()
    print(np.shape(pred_data))
    plotgrayscale(config, pred_data[:,:,0])
    #plot_predictions(config, pred_data, test_data, 106,0)
    #plot_delta(config, pred_data, test_data, 106, 0)
    #plot_fields(config, pred_data, test_data, 106, 0)



