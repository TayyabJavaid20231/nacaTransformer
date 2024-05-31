import jax.numpy as jnp
import typing
import jax
from PIL import Image
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
from src.utilities.pressure_preprocesing import *



@jax.jit
def reverse_Pressure_Coefficient(decoder_input, mach, geometry_internal_value):
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
    decoder_input = decoder_input.at[:, :, 0].set(result) 
    return decoder_input


@jax.jit
def standardize_pressure_and_velocity_denormalize(pred, decoder_input, geometry_internal_value):

    h , w, c = decoder_input.shape
    
    for i in range(c):

        field_copy = jnp.copy(decoder_input[:,:,i])

        field_copy = jnp.where(field_copy == geometry_internal_value, jnp.nan, field_copy)

        mean = jnp.nanmean(field_copy)
        std_deviation = jnp.nanstd(field_copy)
        #result_decoder = jnp.where(decoder_input[:, :, i] != geometry_internal_value,(decoder_input[:, :, i] * std_deviation) + mean, geometry_internal_value)
        #decoder_input = decoder_input.at[:, :, i].set(result_decoder)

        result_pred = jnp.where(pred[:, :, i] != geometry_internal_value,(pred[:, :, i] * std_deviation) + mean, geometry_internal_value)
        pred = pred.at[:, :, i].set(result_pred)

    return pred



@jax.jit
def STD2_denormalize(pred, decoder_input, geometry_internal_value):

    h , w, c = decoder_input.shape
    
    for i in range(c):

        field_copy = jnp.copy(decoder_input[:,:,i])

        field_copy = jnp.where(field_copy == geometry_internal_value, jnp.nan, field_copy)

        mean = jnp.nanmean(field_copy)
        std_deviation = jnp.nanstd(field_copy)
        #result_decoder = jnp.where(decoder_input[:, :, i] != geometry_internal_value,(decoder_input[:, :, i] * std_deviation) + mean, geometry_internal_value)
        #decoder_input = decoder_input.at[:, :, i].set(result_decoder)

        result_pred = jnp.where(pred[:, :, i] != geometry_internal_value,(pred[:, :, i] * (2*std_deviation)) + mean, geometry_internal_value)
        pred = pred.at[:, :, i].set(result_pred)

    return pred


@jax.jit
def scale_to_range_denormalize(pred, decoder_input, new_range, geometry_internal_value):

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

    #result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value,((decoder_input[:, :, 0] - min) * (target_max - target_min) / (max - min)) + target_min,geometry_internal_value)
    result = jnp.where(pred[:,:,0] != geometry_internal_value, ((pred[:,:,0]-target_min) * (max - min) / (target_max - target_min)) + min, geometry_internal_value)
    pred = pred.at[:,:,0].set(result)
    return pred










def Denormalize(config: ConfigDict, pred, y_data, mach):


    range = config.pressure_preprocessing.new_range
    range_array = jnp.array(range)


    vectorized_reverse_pressure_coefficient = jax.vmap(reverse_Pressure_Coefficient,in_axes=(0,0,None))
    vectorized_standardize_all_denormalize = jax.vmap(standardize_pressure_and_velocity_denormalize, in_axes=(0,0,None))
    vectorized_STD2_denormalize = jax.vmap(STD2_denormalize, in_axes=(0,0,None))
    vectorized_Range_denormalize = jax.vmap(scale_to_range_denormalize, in_axes=(0,0, None,None))
    #vectorized_standardize_all = jax.vmap(standardize_pressure_and_velocity, in_axes=(0,None))



    if config.pressure_preprocessing.type == 'coefficient':
        y_data = vectorized_reverse_pressure_coefficient(y_data, mach, config.internal_geometry.value)
        pred = vectorized_reverse_pressure_coefficient(pred, mach, config.internal_geometry.value)
    elif config.pressure_preprocessing.type == 'standardize_all':
        y_data = vectorized_reverse_pressure_coefficient(y_data, mach, config.internal_geometry.value)
        pred = vectorized_standardize_all_denormalize(pred, y_data, config.internal_geometry.value)
    elif config.pressure_preprocessing.type == '2_standard_deviation_normalization':
        y_data = vectorized_reverse_pressure_coefficient(y_data, mach, config.internal_geometry.value)
        pred = vectorized_STD2_denormalize(pred, y_data, config.internal_geometry.value)
    elif config.pressure_preprocessing.type == 'range':
        y_data = vectorized_reverse_pressure_coefficient(y_data, mach, config.internal_geometry.value)
        pred = vectorized_Range_denormalize(pred, y_data, range_array, config.internal_geometry.value)

        

    return pred, y_data
                







