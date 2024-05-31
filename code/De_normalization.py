import jax.numpy as jnp
import typing
import jax
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np


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


#TODO fix this 
def reverse_std_normalization(batch, preds, mean_std , geometry_internal_value):
    
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







def Denormalize(config: ConfigDict, x_data, y_data, mach):
    vectorized_reverse_pressure_coefficient = jax.vmap(reverse_Pressure_Coefficient,in_axes=(0,0,None))
    if config.pressure_preprocessing.type == 'coefficient':
        x_data = vectorized_reverse_pressure_coefficient(x_data, mach, config.internal_geometry.value)
        y_data = vectorized_reverse_pressure_coefficient(y_data, mach, config.internal_geometry.value)

    return x_data, y_data
                







