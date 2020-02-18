

TRAINDIR = '/oasis/scratch/comet/ankitesh/temp_project/PrepData/'
path = '/home/ankitesh/CBrain_project/CBRAIN-CAM/cbrain/'
path_hyam = 'hyam_hybm.pkl'
BATCH_SIZE = 1024
hf = open(path+path_hyam,'rb')

TRAINFILE_RH = 'CI_RH_M4K_NORM_train_shuffle.nc'
NORMFILE_RH = 'CI_RH_M4K_NORM_norm.nc'
VALIDFILE_RH = 'CI_RH_M4K_NORM_valid.nc'

import xarray as xr
ds = xr.open_dataset(TRAINDIR+TRAINFILE_RH)


import sys
sys.path.insert(1,"/home/ankitesh/miniconda3/envs/CbrainCustomLayer/lib/python3.6/site-packages") #work around for h5py
# from cbrain.imports import *
from cbrain.cam_constants import *
from cbrain.utils import *
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
# import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as imag
import scipy.integrate as sin
# import cartopy.crs as ccrs
import matplotlib.ticker as mticker
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
from climate_invariant import *


TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'

scale_dict = load_pickle('/home/ankitesh/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')
in_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']


train_gen = DataGenerator(
    data_fn = TRAINDIR+TRAINFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=False
)

valid_gen = DataGenerator(
    data_fn = TRAINDIR+VALIDFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=False
)

hyam,hybm = pickle.load(hf)
scale_dict_RH = load_pickle('/home/ankitesh/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')
in_vars_RH = ['RH','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars_RH = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']
scale_dict_RH['RH'] = 0.01*L_S/G, # Arbitrary 0.1 factor as specific humidity is generally below 2%



train_gen_RH = DataGenerator(
    data_fn = TRAINDIR+TRAINFILE_RH,
    input_vars = in_vars_RH,
    output_vars = out_vars_RH,
    norm_fn = TRAINDIR+NORMFILE_RH,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_RH,
    batch_size=1024,
    shuffle=True,
    normalize_flag=False
)

from tensorflow.keras import initializers
from tensorflow.keras import layers

class CustomBatchNormalization(layers.Layer):
    def __init__(self, momentum=0.99, epsilon=1e-3,beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros',
                 moving_range_initializer='ones',**kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_range_initializer = (
            initializers.get(moving_range_initializer))
        
        super().__init__(**kwargs)
    
    def build(self,input_shape):
        dim = input_shape[-1]
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                             name='gamma',
                             initializer=self.gamma_initializer,trainable=True)
        self.beta = self.add_weight(shape=shape,
                            name='beta',
                            initializer=self.beta_initializer,
                                   trainable=True)
        
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        
        self.moving_range = self.add_weight(
            shape=shape,
            name='moving_range',
            initializer=self.moving_range_initializer,
            trainable=False)

    #@tf.function
    def call(self, inputs, training=None):
        input_shape = inputs.shape
        
        if not training:
            scaled = (inputs-self.moving_mean)/(self.moving_range+self.epsilon)
            return self.gamma*scaled + self.beta
        
        mean = tf.math.reduce_mean(inputs,axis=0)
        maxr = tf.math.reduce_max(inputs,axis=0)
        minr = tf.math.reduce_min(inputs,axis=0)
        
        range_diff = tf.math.subtract(maxr,minr)
        self.moving_mean = tf.math.add(self.momentum*self.moving_mean, (1-self.momentum)*mean)
        self.moving_range = tf.math.add(self.momentum*self.moving_range,(1-self.momentum)*range_diff)
        scaled = tf.math.divide(tf.math.subtract(inputs,mean),(range_diff+self.epsilon))
        return tf.math.add(tf.math.multiply(self.gamma,scaled),self.beta)
    
    def get_config(self):
        config = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_range_initializer':
                initializers.serialize(self.moving_range_initializer)
        }
        base_config = super(CustomBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape





inp = Input(shape=(64,))
batch_norm_1 = CustomBatchNormalization(dynamic=True)(inp)
inpRH = QV2RH(inp_subQ=train_gen.input_transform.sub, 
              inp_divQ=train_gen.input_transform.div, 
              inp_subRH=train_gen_RH.input_transform.sub, 
              inp_divRH=train_gen_RH.input_transform.div, 
              hyam=hyam, hybm=hybm)(batch_norm_1)
batch_norm_2 = CustomBatchNormalization(dynamic=True)(inpRH)
densout = Dense(128, activation='linear')(batch_norm_2)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    batch_norm_i =CustomBatchNormalization(dynamic=True)(densout)
    densout = Dense(128, activation='linear')(batch_norm_i)
    densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_out = CustomBatchNormalization(dynamic=True)(densout)
out = Dense(64, activation='linear')(batch_norm_out)
Input_RH_CBN = tf.keras.models.Model(inp, out)



Input_RH_CBN.summary()


path_HDF5 = '/oasis/scratch/comet/ankitesh/temp_project/models/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save_RH = ModelCheckpoint(path_HDF5+'CI01_RH_CBN.hdf5',save_best_only=True, monitor='val_loss', mode='min')


Input_RH_CBN.compile(tf.keras.optimizers.Adam(), loss=mse,experimental_run_tf_function=False)
#Inp_RH_CBN.load_weights(path_HDF5+'CI01_RH_CBN.hdf5')
Nep = 10
Input_RH_CBN.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen,\
              callbacks=[earlyStopping, mcp_save_RH])

















