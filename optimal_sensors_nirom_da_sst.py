"""
Created on Wed Mar 23 20:50:40 2022

@author: suraj
"""
import random
random.seed(10)

import numpy as np
np.random.seed(10)

import tensorflow as tf
# tf.random.set_seed(0)

from numpy import linalg as LA
from scipy import linalg

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.animation as animation

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from tensorflow import keras
from tensorflow.keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
    
import warnings
warnings.filterwarnings('ignore')

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)

#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import h5py
from tqdm import tqdm as tqdm

import argparse
import sys
import yaml
import os

#%%
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

def lstm_data_gen(training_set, lookback):
    m, n = training_set.shape
    ytrain = training_set[lookback:,:]
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        xtrain[i] = training_set[i:i+lookback]
    return xtrain, ytrain

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def lstm_model(lookback=4, n_layers=3, n_cells=40, act_func=3,
               initializer=2, optimizer=1, lr_linear=1.5, type_lstm=1):
    
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

    act_func_dict = {1:'tanh',2:'relu',3:tf.keras.layers.LeakyReLU(alpha=0.1)}
    initializer_dict = {1:'uniform',2:'glorot_normal',3:'random_normal'}
    optimizer_dict = {1:'adam',2:'rmsprop',3:'SGD'}
    
    input_modes = Input(shape=(lookback,nr))
        
    a = LSTM(n_cells, return_sequences=True)(input_modes)
    
    if type_lstm == 1:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            a = Add()([a,x]) # main1 + skip1
    
    elif type_lstm == 2:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1             
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(x) # main1            
            a = Add()([a,x]) # main1 + skip1
    
    elif type_lstm == 3:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            a = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # skip1            
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(x) # main1            
            a = Add()([a,x]) # main1 + skip1
            
    elif type_lstm == 4:
        for k in range(n_layers):
            x = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # main1 
            a = LSTM(n_cells, return_sequences=True,activation=act_func_dict[act_func], kernel_initializer=initializer_dict[initializer])(a) # skip1            
            a = Add()([a,x]) # main1 + skip1
    
    x = LSTM(n_cells, return_sequences=False)(a)
    x = Dense(nr, activation='linear')(x)
    model = Model(inputs=[input_modes], outputs=x)
    
    lr = 10**(-2.0*lr_linear)
    
    if optimizer == 1:
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 2:
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 3:
        opt = keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 4:
        opt = keras.optimizers.Adadelta(learning_rate=lr)
    elif optimizer == 5:
        opt = keras.optimizers.Adamax(learning_rate=lr)

    model.compile(loss='mean_squared_error', 
                  optimizer=opt, metrics=[coeff_determination])
    
    return model

#%%
f = h5py.File('../../sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

fig,axs = plt.subplots(1,1, figsize=(10,8))
current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)
cs = axs.imshow(sst2[0,:,:],cmap='RdYlBu')
# axs.grid()
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.4)
fig.tight_layout()
plt.show()    

#%%
sst_no_nan = np.nan_to_num(sst)
sst = sst.T

num_samples = sst.shape[1]

for i in range(num_samples):
    nan_array = np.isnan(sst[:,i])
    not_nan_array = ~ nan_array
    array2 = sst[:,i][not_nan_array]
    # print(i, array2.shape[0])
    if i == 0:
        num_points = array2.shape[0]
        sst_masked = np.zeros((array2.shape[0],num_samples))
    sst_masked[:,i] = array2

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--npe', default=40, type=int, help="number of ensembles")
parser.add_argument('--lambda_', default=1.5, type=np.float64, help="inflation factor")
args = parser.parse_args()

npe = args.npe
lambda_ = args.lambda_

nr = 4
me = 300
nopt = 200
nes = sst_masked.shape[0]
ns_train_rom = 1000
lookback = 4
epochs = 200
batch_size = 32
test_size = 0.2
mu = 0.0
sd2_ic = 1.0
sd1_ic = np.sqrt(sd2_ic)
sd2_obs = 1.0
sd1_obs = np.sqrt(sd2_obs)
sd2_obs_modes = 0.01
sd1_obs_modes = np.sqrt(sd2_obs_modes)
fic = 2
fob = 1 # reconstructed data from sparse sensors
nf = 3 # frequency of observation

nt = num_samples
t = np.linspace(1, nt, nt)

ns_train = 1000
ns_test = nt - ns_train
nb = int(ns_test/nf) # number of observation time
t_test = np.linspace(1, ns_test, ns_test)

directory = f'results_{npe}_{lambda_}'
if not os.path.exists(directory):
    os.makedirs(directory)

#%%
sst_avg = np.mean(sst_masked[:,:ns_train_rom], axis=1, keepdims=True)
sst_fluc_train = sst_masked[:,:ns_train_rom] - sst_avg
sst_fluc = sst_masked - sst_avg

PHIw, L, RIC  = POD(sst_fluc_train, nr)     
L_per = np.cumsum(L, axis=0)*100/np.sum(L)

#%%
# k = np.linspace(1,ns,ns)
# fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
# axs.loglog(k,L_per, lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
# axs.set_xlim([1,ns])
# axs.axvspan(0, nr, alpha=0.2, color='red')
# fig.tight_layout()
# plt.show()

at = PODproj(sst_fluc_train, PHIw)
at_signs = np.sign(at[0,:]).reshape([1,-1])
at = at/at_signs
PHIw = PHIw/at_signs

oin = sorted(random.sample(range(nes), me)) 

sst_obs = np.zeros_like(sst_fluc)
sst_obs_noise = np.zeros_like(sst_fluc)
obs_noise = np.zeros_like(sst_fluc)

obs_noise_sampled = np.random.normal(mu,sd1_obs,[me,nt])
sst_obs[oin,:] = sst_fluc[oin,:] 
sst_obs_noise[oin,:] = sst_fluc[oin,:] + obs_noise_sampled

obs_noise[oin,:] = obs_noise_sampled

atrue = PODproj(sst_fluc, PHIw)
at_obs = PODproj(sst_obs, PHIw)

# sst_obs = sst_fluc[oin,:]

# PHIw_obs, L_obs, RIC_obs = POD(sst_obs, nr)     

# at_obs = PODproj(sst_obs, PHIw_obs)
# at_obs_signs = np.sign(at_obs[0,:]).reshape([1,-1])
# at_obs = at_obs/at_obs_signs
# PHIw_obs = PHIw_obs/at_obs_signs

at = at
atrue = atrue

#%%
PHIw_opt, L_opt, RIC_opt  = POD(sst_fluc_train, nopt) 
at_opt = PODproj(sst_fluc_train, PHIw_opt)
at_opt_signs = np.sign(at_opt[0,:]).reshape([1,-1])
at_opt = at_opt/at_opt_signs
PHIw_opt = PHIw_opt/at_opt_signs

Q, R, pivot = linalg.qr(PHIw_opt.T, mode='economic', pivoting=True)

sensors = pivot[:nopt]
roin = np.linspace(0, nopt-1, nopt, dtype=int)
# observation operator
C = np.zeros((nopt,nes))
C[roin,sensors] = 1.0

# sensors = pivot
# roin = np.linspace(0, nes-1, nes, dtype=int)
# # observation operator
# C = np.zeros((nes,nes))
# C[roin,sensors] = 1.0


print('Checking QR for p = r = nopt case', flush = True)
print(np.allclose(PHIw_opt.T @ C.T[:,:nopt], np.dot(Q, R[:,:nopt])), flush = True)

# print('Checking full QR for p = r = nopt case', flush = True)
# print(np.allclose(PHIw_opt.T @ C.T[:,:], np.dot(Q, R[:,:])), flush = True)


#%%
measurements = sst_fluc[sensors,:] + 0.0*np.random.normal(mu,sd1_obs_modes,[nopt,nt])
am = np.linalg.pinv(C @ PHIw_opt) @ measurements
am = am.T

lin_rec_fluc = PODrec(am,PHIw_opt)
at_obs_lin_rec = PODproj(lin_rec_fluc, PHIw)

#%%
ufom = sst_masked

utrue = PODrec(atrue,PHIw)
utrue = utrue + sst_avg

u_lin_rec = PODrec(am[:,:nr],PHIw)
u_lin_rec = u_lin_rec + sst_avg

rmse_fom_true = np.linalg.norm(ufom - utrue, axis=0)/np.sqrt(nes)
rmse_fom_lin_rec = np.linalg.norm(ufom - u_lin_rec, axis=0)/np.sqrt(nes)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,4),sharex=True)
ax.plot(t, rmse_fom_true, label='FOM-True')
ax.plot(t, rmse_fom_lin_rec, label='FOM-Linear Rec')
ax.legend()
fig.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot = 200
for i in range(4):
    
    ax[i].plot(t[:ns_plot],at[:ns_plot,i], lw=3)
    ax[i].plot(t[:ns_plot],at_opt[:ns_plot,i], 'k--', lw=2)
    # ax[i].plot(am[:ns_plot,i],'gs', fillstyle='none')
    ax[i].plot(t[:ns_plot],at_obs_lin_rec[:ns_plot,i],'ro', fillstyle='none')

fig.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot = 1100
for i in range(nr):
    # ax[i].plot(at[:,i], lw=3)
    ax[i].plot(t[ns_train:ns_plot],atrue[ns_train:ns_plot,i], 'k', lw=2)
    ax[i].plot(t[ns_train:ns_plot],at_obs_lin_rec[ns_train:ns_plot,i],'go',fillstyle='none',ms=8)
    # ax[i].axvspan(0, ns, color='gray', alpha=0.3)    

fig.tight_layout()
plt.show()
    
#%%
sc = MinMaxScaler(feature_range=(-1,1))
sc.fit(at)
training_set = sc.transform(at)

atrain_max = np.max(at, axis=0, keepdims=True)
atrain_min = np.min(at, axis=0, keepdims=True)

print('#-------------- Loading the trained model --------------#')
model_name = '../best_model_noaa.h5'
model = load_model(model_name, custom_objects={'coeff_determination': coeff_determination})
model.summary()

#%%
xtest = np.zeros((npe,1,lookback,nr))
ue = np.zeros((nr,npe,ns_test))
ua = np.zeros((nr,ns_test))
uu = np.zeros((nr,ns_test))

me_modes = nr
freq = int(nr/me_modes)
oin = [freq*i-1 for i in range(1,me_modes+1)]
roin = np.linspace(0, me_modes-1, me_modes, dtype=int)
print(oin)

# observation operator
dh = np.zeros((me_modes,nr))
dh[roin,oin] = 1.0

# virtual observations for twin experiment
oib = [nf*k for k in range(nb+1)]

atrue_sc = sc.transform(atrue)

if fob == 1:    
    uobsfull = sc.transform(at_obs_lin_rec[ns_train:,:]).T

z = np.zeros((me_modes,nb+1))
z[:,:] = uobsfull[oin,:][:,oib]

#%%
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot = 1500
ns_plot_obs = int(ns_plot/nf)
for i in range(nr):
    ax[i].plot(t_test[:ns_plot], atrue_sc[ns_train:ns_train+ns_plot,i],'k-',label='True')
    # ax[i].plot(t_test[:ns_plot], uobsfull.T[:ns_plot,i],'go',fillstyle='none',ms=8,label='Noisy observatios')
    ax[i].plot(t_test[oib][:ns_plot_obs], z.T[:ns_plot_obs,i],'bo',fillstyle='none',ms=6,label='Temporally sparse')

ax[0].legend()
fig.tight_layout()
plt.show()

#%%
# ic_snapshots = np.random.randint(num_samples-lookback, size=npe)
ic_snapshots = np.zeros(npe, dtype=int)

testing_set = np.copy(atrue[ns_train:,:])
testing_set = sc.transform(testing_set)

if fic == 1:
    for ne in range(npe):
        print(ne, ic_snapshots[ne])
        nsnap = ic_snapshots[ne]
        ue[:,ne,:lookback] = testing_set[nsnap:nsnap+lookback,:] + np.random.normal(mu,sd1_ic,[lookback,nr])
        xtest[ne,0,:,:] = ue[:,ne,:lookback]
        
elif fic == 2:
    for ne in range(npe):
        for k in range(lookback):    
            print(ne, k)
            us = sst_fluc[:,ns_train+k] + np.random.normal(mu,sd1_ic,[nes])
            us = np.reshape(us,[-1,1])
            ak = PODproj(us, PHIw)
            ue[:,ne,k] = sc.transform(ak)
            xtest[ne,0,k,:] = ue[:,ne,k]            
    
ua[:,:lookback] = np.average(ue[:,:,:lookback], axis=1)
uu[:,:lookback] = np.std(ue[:,:,:lookback], axis=1)

#%%
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot = lookback
for i in range(nr):
    ax[i].plot(t_test[:ns_plot], testing_set[:ns_plot,i], 'k', lw=2, label='True')
    ax[i].plot(t_test[:ns_plot], ua.T[:ns_plot,i],'r--', lw=2, label='Pred')
    yp = ua.T[:ns_plot,i] + uu.T[:ns_plot,i]
    yn = ua.T[:ns_plot,i] - uu.T[:ns_plot,i]
    ax[i].fill_between(t_test[:ns_plot], yp,yn,'k', alpha=0.5, label='Pred')
    # ax[i].axvspan(0, ns, color='gray', alpha=0.3)    

ax[0].legend()
fig.tight_layout()
plt.show()

#%%
print('#-------------- Starting data assimilation --------------#')
kobs = 2
for k in range(lookback,ns_test):
    for ne in range(npe):
        ue[:,ne,k] = model.predict(xtest[ne])
        xtest[ne,0,:-1,:] = xtest[ne,0,1:,:]
        xtest[ne,0,-1,:] = ue[:,ne,k]
    
    ua[:,k] = np.average(ue[:,:,k], axis=1)    
    uu[:,k] = np.std(ue[:,:,k], axis=1)
    
    if k % nf == 0:
        print(k, kobs)
        # compute mean of the forecast fields
        uf = np.average(ue[:,:,k], axis=1)   
        
        # compute Af dat
        Af = ue[:,:,k] - uf.reshape(-1,1)
        
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        diag = np.arange(nr)
        cc[diag,diag] = cc[diag,diag] + sd2_obs_modes
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
                
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        ua[:,k] = np.average(ue[:,:,k], axis=1)    
        uu[:,k] = np.std(ue[:,:,k], axis=1)
    
        # multiplicative inflation: set lambda=1.0 for no inflation
        ue[:,:,k] = ua[:,k].reshape(-1,1) + lambda_*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        for ne in range(npe):
            xtest[ne,0,-1,:] = ue[:,ne,k]
            
        kobs += 1
        
#%%
xtest = np.zeros((1,lookback,nr))
ypred = np.zeros((ns_test,nr))

# create input at t = 0 for the model testing
for i in range(lookback):
    xtest[0,i,:] = testing_set[i]
    ypred[i] = testing_set[i]

# predict results recursively using the model
for i in range(lookback,ns_test):
    ypred[i] = model.predict([xtest])
    xtest[0,:-1,:] = xtest[0,1:,:]
    xtest[0,lookback-1,:] = ypred[i]

#%%
uobsfull = uobsfull.T
ua = ua.T
uu = uu.T

yp = ua + 1*uu
yn = ua - 1*uu
    
#%%
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot = 200
for i in range(nr):
    ax[i].plot(t_test[:ns_plot], testing_set[:ns_plot,i], 'k', lw=3, label='True')
    # ax[i].plot(t[:ns_plot], uobsfull[:ns_plot,i],'ko',fillstyle='none',ms=8,label='Noisy observations')
    ax[i].plot(t_test[:ns_plot], ypred[:ns_plot,i],'b-', lw=3, label='Pred')
    ax[i].plot(t_test[:ns_plot], ua[:ns_plot,i],'r-', lw=3, label='Pred-DA')
    ax[i].fill_between(t_test[:ns_plot], yp[:ns_plot,i], yn[:ns_plot,i],color='r', alpha=0.2, label='SD-Band')    
    # ax[i].axvspan(0, ns, color='gray', alpha=0.3)    
    
ax[0].legend()
fig.tight_layout()
plt.show()
    
#%%
atest = atrue[ns_train:,:]
apred = sc.inverse_transform(ypred)
apred_da = sc.inverse_transform(ua)

yp_usc = sc.inverse_transform(yp)
yn_usc = sc.inverse_transform(yn)
    
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
ax = ax.flat
ns_plot_st = 0
ns_plot = ns_plot_st + 200
for i in range(nr):
    ax[i].plot(t_test[ns_plot_st:ns_plot],atest[ns_plot_st:ns_plot,i], 'k', lw=2, label='True')
    ax[i].plot(t_test[ns_plot_st:ns_plot],apred[ns_plot_st:ns_plot,i],'b-', lw=2, label='Pred')
    ax[i].plot(t_test[ns_plot_st:ns_plot],apred_da[ns_plot_st:ns_plot,i],'r-', lw=2, label='Pred-DA')
    ax[i].fill_between(t_test[ns_plot_st:ns_plot], yp_usc[ns_plot_st:ns_plot,i], 
                       yn_usc[ns_plot_st:ns_plot,i],color='r', alpha=0.2, label='SD-Band')
    # ax[i].axvspan(1, ns, color='gray', alpha=0.3)    

ax[0].legend()
fig.tight_layout()
plt.show()

#%%
filename = filename = os.path.join(directory, 'results_optimal.npz') 
np.savez(filename, t = t, atrue = atrue, pivot = pivot, 
         PHIw = PHIw, PHIw_opt = PHIw_opt, sst_avg = sst_avg,
         atest = atest, apred = apred, apred_da = apred_da, 
         yp = yp_usc, yn = yn_usc, t_test = t_test,
         z = z, oib = np.array(oib)
         )

#%%
ufom = sst_masked[:,ns_train:]

utrue = PODrec(atest,PHIw)
utrue = utrue + sst_avg

upred = PODrec(apred,PHIw)
upred = upred + sst_avg

upred_da = PODrec(apred_da,PHIw)
upred_da = upred_da + sst_avg        

rmse_fom_true = np.linalg.norm(ufom - utrue, axis=0)/np.sqrt(nes)
rmse_fom_ml = np.linalg.norm(ufom - upred, axis=0)/np.sqrt(nes)
rmse_fom_ml_denkf = np.linalg.norm(ufom - upred_da, axis=0)/np.sqrt(nes)

rmse_true_ml = np.linalg.norm(utrue - upred, axis=0)/np.sqrt(nes)
rmse_true_ml_denkf = np.linalg.norm(utrue - upred_da, axis=0)/np.sqrt(nes)

#%%
point_sensors = random.sample(range(nes), 4)

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,5),sharex=True)
ax = ax.flat
ns_plot = 500

for i in range(4):
    ax[i].plot(ufom[point_sensors[i],:ns_plot],label='FOM')
    ax[i].plot(utrue[point_sensors[i],:ns_plot],label='True')
    ax[i].plot(upred[point_sensors[i],:ns_plot],label='Pred')
    ax[i].plot(upred_da[point_sensors[i],:ns_plot],label='Pred-DA')

ax[1].legend()
plt.show()

#%%
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,5),sharex=True)

ax[0].plot(t_test, rmse_fom_true, label='FOM-True')
ax[0].plot(t_test, rmse_fom_ml, label='FOM-ML')
ax[0].plot(t_test, rmse_fom_ml_denkf, label='FOM-ML-DEnKF')

ax[1].plot(t_test, rmse_true_ml, label='True-ML')
ax[1].plot(t_test, rmse_true_ml_denkf, label='True-ML-DEnKF')

# ax[0].set_ylim([100,200])
# ax[1].set_ylim([10,300])

for i in range(2):    
    ax[i].legend()
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$||\epsilon||$')
    
plt.show()

#%%
import seaborn as sns

# matplotlib histogram
# plt.hist(rmse_true_ml, color = 'blue', edgecolor = 'black',
#          bins = int(180/5))

# seaborn histogram
# sns.distplot(rmse_true_ml, hist=True, kde=False, 
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4),sharex=True)

sns.distplot(rmse_true_ml, hist=False, kde=True, 
              bins=int(180/2), color = 'darkblue', 
              hist_kws={}, #'edgecolor':'black'
              kde_kws={'shade':True,'linewidth': 2},
              norm_hist=False,
              label='NIROM')

sns.distplot(rmse_true_ml_denkf, hist=False, kde=True, 
             bins=int(180/2), color = 'red', 
             hist_kws={}, #'edgecolor':'red'
             kde_kws={'shade':True,'linewidth': 2},
             norm_hist=False,
             label='NIROM-DEnKF')

ax.legend()
ax.set_xlabel('Forecast RMSE')
ax.set_ylabel('Probability Density')
ax.set_yscale('linear')
ax.set_xlim([0,2.5])
# ax.set_ylim([1e-5,0.05])
fig.tight_layout()
plt.show()

#%%
from scipy.stats.kde import gaussian_kde
from numpy import linspace
# create fake data
data = rmse_fom_ml
# this create the kernel, given an array it will estimate the probability over that values
kde = gaussian_kde( data )
# these are the values over wich your kernel will be evaluated
dist_space = linspace( min(data), max(data), 100 )
# plot the results
plt.plot( dist_space, kde(dist_space) ,'k-')


# create fake data
data = rmse_fom_ml_denkf
# this create the kernel, given an array it will estimate the probability over that values
kde = gaussian_kde( data )
# these are the values over wich your kernel will be evaluated
dist_space = linspace( min(data), max(data), 100 )
# plot the results
plt.plot( dist_space, kde(dist_space) ,'r-')

plt.show()

#%%
Tfom_mid = np.zeros(not_nan_array.shape[0])
Tfom_final = np.zeros(not_nan_array.shape[0])

Ttrue_mid = np.zeros(not_nan_array.shape[0])
Ttrue_final = np.zeros(not_nan_array.shape[0])

Tpred_mid = np.zeros(not_nan_array.shape[0])
Tpred_final = np.zeros(not_nan_array.shape[0])

Tpred_mid_da = np.zeros(not_nan_array.shape[0])
Tpred_final_da = np.zeros(not_nan_array.shape[0])

mid = int(ns_test/2)
final = ns_test - 1

Tfom_mid[Tfom_mid == 0] = 'nan'
Tfom_mid[not_nan_array] = ufom[:,mid]
Tfom_final[Tfom_final == 0] = 'nan'
Tfom_final[not_nan_array] = ufom[:,final]

Ttrue_mid[Ttrue_mid == 0] = 'nan'
Ttrue_mid[not_nan_array] = utrue[:,mid]
Ttrue_final[Ttrue_final == 0] = 'nan'
Ttrue_final[not_nan_array] = utrue[:,final]

Tpred_mid[Tpred_mid == 0] = 'nan'
Tpred_mid[not_nan_array] = upred[:,mid]
Tpred_final[Tpred_final == 0] = 'nan'
Tpred_final[not_nan_array] = upred[:,final]

Tpred_mid_da[Tpred_mid_da == 0] = 'nan'
Tpred_mid_da[not_nan_array] = upred_da[:,mid]
Tpred_final_da[Tpred_final_da == 0] = 'nan'
Tpred_final_da[not_nan_array] = upred_da[:,final]

tfom_mid = np.flipud((Tfom_mid.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
tfom_final = np.flipud((Tfom_final.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

ttrue_mid = np.flipud((Ttrue_mid.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
ttrue_final = np.flipud((Ttrue_final.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

trec_mid = np.flipud((Tpred_mid.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
trec_final = np.flipud((Tpred_final.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

trec_mid_da = np.flipud((Tpred_mid_da.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))
trec_final_da = np.flipud((Tpred_final_da.reshape(len(lat[0,:]),len(lon[0,:]),order='F')))

#%%
fig,ax = plt.subplots(2,2, figsize=(10,5))
axs = ax.flat

current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)

vmin, vmax = -5.0, 35.0
cs = axs[0].imshow(tfom_final,cmap='RdYlBu',vmin=vmin,vmax=vmax)
cs = axs[1].imshow(ttrue_final,cmap='RdYlBu',vmin=vmin,vmax=vmax)
cs = axs[2].imshow(trec_final,cmap='RdYlBu',vmin=vmin,vmax=vmax)
cs = axs[3].imshow(trec_final_da,cmap='RdYlBu',vmin=vmin,vmax=vmax)

#axs.grid()
# fig.colorbar(cs, ax=axs[0], orientation='vertical',shrink=1)
    
fig.tight_layout()
plt.show()

#%%
fig,axs = plt.subplots(3,2, figsize=(10,8))

current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)

current_cmap = plt.cm.get_cmap('RdBu')
current_cmap.set_bad(color='black',alpha=0.8)

vmin, vmax = -5.0, 35.0
cs = axs[0,0].imshow(ttrue_final,cmap='RdYlBu',vmin=vmin,vmax=vmax)
cs = axs[1,0].imshow(trec_final,cmap='RdYlBu',vmin=vmin,vmax=vmax)
cs = axs[2,0].imshow(trec_final_da,cmap='RdYlBu',vmin=vmin,vmax=vmax)

vmin, vmax = -2.0, 2.0
cs = axs[0,1].imshow(tfom_final - ttrue_final,cmap='RdBu')
cs = axs[1,1].imshow(trec_final - ttrue_final,cmap='RdBu',vmin=3*vmin,vmax=2*vmax)
cs = axs[2,1].imshow(trec_final_da - ttrue_final,cmap='RdBu',vmin=vmin,vmax=vmax)

axs[0,0].set_title('True')
axs[0,1].set_title('FOM - True')

#axs.grid()
# fig.colorbar(cs, ax=axs[2,0], orientation='vertical',shrink=1)
    
fig.tight_layout()
plt.show()