import jax.numpy as jnp
import typing
import jax
from PIL import Image
import matplotlib.pyplot as plt7


import numpy as np

#TODO test if this works as expected
def set_geometry_internal_value(batch, old_value, new_value):
    encoder_input = batch['encoder']
    decoder_input = batch['decoder']

    new_encoder_input = jnp.where(encoder_input[:, :, :, :] == old_value, new_value, encoder_input[:, :, :, :])
    new_decoder_input = jnp.where(decoder_input[:,:,:,:]== old_value, new_value, decoder_input[:, :, :, :]  )
    
    batch['encoder'] = new_decoder_input
    batch['decoder'] = new_decoder_input

    return batch 


'''
https://en.wikipedia.org/wiki/Pressure_coefficient
'''
@jax.jit
def normalize_pressure_coefficient(decoder_input, mach, geometry_internal_value):
    
    # TODO try using inf or NAN for the inside of the geometry instead of 
    # # Define thermodynamic properties of air at ICAO standard atmosphere
    T0 = 288.15  # [K] Total temperature
    p0 = 101325  # [Pa] Total pressure
    gamma = 1.4  # [-] Ratio of specific heats
    R = 287.058  # [J/(kg*K)] Specific gas constant for dry air
    rho0 = 1.225 # [kg/m^3] air density look at ICA0 standard atmosphere

    # # Normalise pressure by freestream pressure
     
    T = T0 / (1 + 0.5 * (gamma - 1) * mach ** 2)
    p_inf = p0 * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (-gamma / (gamma - 1))
    u_inf = mach * jnp.sqrt(gamma * R * T)

    # since the TfRecord files are normalised by p_inf themselfes we do (p-1)/(0.5*rho0*p_inf*u_inf^2)  
    denominator = (rho0 * u_inf ** 2) / (2 * p_inf) 
    result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value,(decoder_input[:,:,0] - 1) / denominator, geometry_internal_value)
    decoder_input = decoder_input.at[:,:,0].set(result) 
    return decoder_input





@jax.jit
def scale_to_range(decoder_input, new_range, geometry_internal_value):
    # TODO try using inf or NaN for the inside of the geometry instead of 0

    target_min, target_max = new_range
    max_float32 = jnp.finfo(jnp.float32).max
    min_float32 = jnp.finfo(jnp.float32).min

    pressure_field = np.squeeze(decoder_input[:, :, 0])

    pressure_field_min = jnp.copy(pressure_field)
    pressure_field_min = jnp.where(pressure_field_min == geometry_internal_value, max_float32, pressure_field)

    pressure_field_max = jnp.copy(pressure_field)
    pressure_field_max = jnp.where(pressure_field_max == geometry_internal_value, min_float32, pressure_field)


    min = jnp.min(pressure_field_min)
    max = jnp.max(pressure_field_max)

    result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value,((decoder_input[:, :, 0] - min) * (target_max - target_min) / (max - min)) + target_min,geometry_internal_value)
    decoder_input = decoder_input.at[:,:,0].set(result)
    return decoder_input

@jax.jit
def standardize(decoder_input, geometry_internal_value):

    pressure_field_copy = jnp.copy(decoder_input[:,:,0])

    pressure_field_copy = jnp.where(pressure_field_copy == geometry_internal_value, jnp.nan, pressure_field_copy)    

    mean = jnp.nanmean(pressure_field_copy)
    std_deviation = jnp.nanstd(pressure_field_copy)

    result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value,(decoder_input[:, :, 0] - mean) / std_deviation, geometry_internal_value)
    decoder_input = decoder_input.at[:, :, 0].set(result)
    return decoder_input


@jax.jit
def standardize_pressure_and_velocity(decoder_input, geometry_internal_value):

    h , w, c = decoder_input.shape
    
    for i in range(c):

        field_copy = jnp.copy(decoder_input[:,:,i])

        field_copy = jnp.where(field_copy == geometry_internal_value, jnp.nan, field_copy)    

        mean = jnp.nanmean(field_copy)
        std_deviation = jnp.nanstd(field_copy)
        result = jnp.where(decoder_input[:, :, i] != geometry_internal_value,(decoder_input[:, :, i] - mean) / std_deviation, geometry_internal_value)
        decoder_input = decoder_input.at[:, :, i].set(result)
    return decoder_input

@jax.jit
def two_standard_deviation_pressure_and_velocity(decoder_input, geometry_internal_value):

    h , w, c = decoder_input.shape
    
    for i in range(c):

        field_copy = jnp.copy(decoder_input[:,:,i])

        field_copy = jnp.where(field_copy == geometry_internal_value, jnp.nan, field_copy)    

        mean = jnp.nanmean(field_copy)
        std_deviation = jnp.nanstd(field_copy)
        result = jnp.where(decoder_input[:, :, i] != geometry_internal_value,(decoder_input[:, :, i] - mean) / (2 * std_deviation), geometry_internal_value)
        decoder_input = decoder_input.at[:, :, i].set(result)
    return decoder_input



#TODO extract mach from label data to put into the normalize pressure_coefficient function
def process_label(label):
    mach = []

    label = label.tolist()

    for entry in label:
        entry = entry.decode("utf-8")
        #entry = entry[2:-2]
        label_data = entry.split('_')
        mach.append(float(label_data[-1]))
    
    return mach
    
def reverse_standardize(input ,mean, std, geometry_internal_value):
    pressure_field_copy = jnp.copy(input[:,:,0])
    pressure_field_copy = jnp.where(pressure_field_copy == geometry_internal_value, jnp.nan, pressure_field_copy)    
    result = jnp.where(input[:, :, 0] != geometry_internal_value, input[:, :, 0]*std + mean , geometry_internal_value)
    input = input.at[:, :, 0].set(result)
    return input     

#TODO fix this 
def reverse_standardize_batched(batch, preds, mean_std , geometry_internal_value):
    
    encoder = batch['encoder']
    decoder = batch['decoder']

    reversed_preds = np.array(preds.shape)
    reversed_batch = np.array(decoder.shape)
    for i, sample in enumerate(decoder):
        mean, std = mean_std[i]
        reversed_batch[i] = reversed_batch[i].set(reverse_standardize(sample,mean, std, geometry_internal_value))
    
    for i, sample in enumerate(preds):
        mean, std = mean_std[i]
        reversed_preds[i] = reversed_preds[i].set(reverse_standardize(sample, mean, std, geometry_internal_value))
        
    batch['decoder']= reversed_batch


    return batch, reversed_preds 

@jax.jit
def reverse_Pressure_Coefficient(decoder_input, mach,geometry_internal_value):
    T0 = 288.15  # [K] Total temperature
    p0 = 101325  # [Pa] Total pressure
    gamma = 1.4  # [-] Ratio of specific heats
    R = 287.058  # [J/(kg*K)] Specific gas constant for dry air
    rho0 = 1.225 # [kg/m^3] air density look at ICA0 standard atmosphere

    # # Normalise pressure by freestream pressure
     
    T = T0 / (1 + 0.5 * (gamma - 1) * mach ** 2)
    p_inf = p0 * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (-gamma / (gamma - 1))
    u_inf = mach * jnp.sqrt(gamma * R * T)

    # since the TfRecord files are normalised by p_inf themselfes we do (p-1)/(0.5*rho0*p_inf*u_inf^2)  
    nominator = (rho0 * u_inf ** 2) / (2 * p_inf) 
    result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value,(decoder_input[:, :, 0] * nominator) + 1, geometry_internal_value)
    decoder_input = decoder_input.at[:,:,0].set(result) 
    return decoder_input



def pressure_preprocessing(batch, config):

    type = config.pressure_preprocessing.type
    if type == 'coefficient':
        return batch
        #batch['decoder'] = vectorized_normalize_pressure_coefficient(decoder_input, mach, internal_value)


    range = config.pressure_preprocessing.new_range
    geometry_internal_value = config.internal_geometry.value

    

    encoder_input = batch['encoder']
    decoder_input = batch['decoder']
    label = batch['label']
    label_data = process_label(label)


    mach = jnp.array(label_data)
    range_array = jnp.array(range)

    internal_value = geometry_internal_value
 
    vectorized_normalize_pressure_coefficient = jax.vmap(normalize_pressure_coefficient,in_axes=(0,0,None))
    vectorized_reverse_pressure_coefficient = jax.vmap(reverse_Pressure_Coefficient,in_axes=(0,0,None))
    vectorized_range = jax.vmap(scale_to_range,in_axes=(0,None,None))
    vectorized_standardize = jax.vmap(standardize, in_axes=(0,None))
    vectorized_standardize_all = jax.vmap(standardize_pressure_and_velocity, in_axes=(0,None))
    vectorized_2std_all = jax.vmap(two_standard_deviation_pressure_and_velocity, in_axes=(0,None))

    if type == 'standardize':
        batch['decoder'] = vectorized_reverse_pressure_coefficient(decoder_input, mach, internal_value)
        batch['decoder'] = vectorized_standardize(decoder_input, internal_value)
    elif type == 'standardize_all':
        batch['decoder'] = vectorized_reverse_pressure_coefficient(decoder_input, mach, internal_value)
        batch['decoder'] = vectorized_standardize_all(decoder_input, internal_value)  
#        print(np.shape(batch['decoder']))  
#        plt.contourf(batch['decoder'][0, :, :, 2])
#        plt.savefig("contourplot")
    elif type == '2_standard_deviation_normalization':
        batch['decoder'] = vectorized_reverse_pressure_coefficient(decoder_input, mach, internal_value)
        batch['decoder'] = vectorized_2std_all(decoder_input, internal_value)
    elif type == 'range':
        batch['decoder'] = vectorized_reverse_pressure_coefficient(decoder_input, mach, internal_value)
        batch['decoder'] = vectorized_range(decoder_input, range_array, internal_value)
#    elif type == 'coefficient':
#        #print('maybe add nan for internal points')
#        continue
        #batch['decoder'] = vectorized_normalize_pressure_coefficient(decoder_input, mach, internal_value)
    else:
        raise Exception("No proper normalization specified")

    return batch
