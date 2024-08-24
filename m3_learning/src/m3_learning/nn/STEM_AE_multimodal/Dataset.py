import numpy as np
import hyperspy.api as hs
import h5py
from skimage.draw import disk
import dask.array as da
from dask.diagnostics import ProgressBar

from itertools import zip_longest

import pyNSID
import os
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import Normalizer,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm as tqdm
import glob
import torch
from bisect import bisect_left,bisect_right
import time
from skimage.morphology import binary_dilation, binary_erosion,disk
from pdb import set_trace as bp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


'''From carolin:
Cropping below zero is fine in principle. In practice, I would crop below -20eV since peaks can broaden and shift on the energy axis a little bit. The maximum of the zero loss peak (the really big sharp one) is supposed to be at zero.

The relevant spectral regions depend largely on the elements present in the material. Different elements show up at different energies, and some elements are in regions where we won't really be able to measure them at all.
Generally speaking, there are two main parts of an EELS spectrum: the low loss region (0- ~50 eV) and the core loss region (everything after that). The low loss region will always be relevant to us and we should include it in all models. The core loss region is where things get element-specific. For the data we sent, the parts that are relevant are the 750-820eV region (Co) and the 380-550eV regions (O and N). The sample also contains Au, but it would show up at such high energies that we can't see it.

I think the preprocessing steps you outlined all make sense. The only thing I would do otherwise is to align the zero loss peak for each pixel, i.e. shift spectra along the x-axis such that the maximum of the zero loss peak is actually at 0eV. If you look carefully, you can see that it moves by a few eV as the electron beam moves across the sample. This is normal and we correct for it by shifting the spectra for each pixel accordingly.
'''


class STEM_Dataset(Dataset):
    """Class for the STEM dataset.
    """
    def __init__(self, save_path, data_path,overwrite=False,**kwargs):
        """Initialization of the class.

        Args:
            save_path (string): path where the hyperspy file is located
        """
        self.save_path = save_path
        self.h5_name = f'{save_path}/combined_data.h5'

        # create and sort metadata 
        self.meta = {}
        path_list = glob.glob(f'{save_path}/*/*/*/SI data (*)/Diffraction SI.dm4')
        def get_number(path):
            return int(path.split('/')[-2].split(' ')[-1][1:-1])
        def get_particle(path):
            return path.split('/')[-3]
        path_list.sort(key=get_number)
        path_list.sort(key=get_particle)
        self.meta['path_list'] = path_list
        
        # create/ open h5 file
        if not os.path.exists(self.h5_name): h = h5py.File(self.h5_name,'w')
        else: h = h5py.File(self.h5_name,'r+')

        print('fetching metadata...')
        self.meta['particle_list'] = []
        self.data_list = []
        self.meta['shape_list'] = []
        self.bad_files = []
        self.meta['particle_inds'] = [0]
        self.meta['sample_inds'] = [0]

        # go through data files and fill metadata
        for i,path in enumerate(tqdm(self.meta['path_list'])):
            try:
                s = hs.load(path, lazy=True)
                self.data_list.append(s.data)
                self.meta['particle_list'].append(path.split('/')[-3] + path.split('/')[-2].split(' ')[-1])
                self.meta['shape_list'].append(s.data.shape)
                self.meta['particle_inds'].append(self.meta['particle_inds'][-1] + s.data.shape[0]*s.data.shape[1])
                if i>1 and self.meta['particle_list'][-1].split('(')[0] != self.meta['particle_list'][-2].split('(')[0]:
                    self.meta['sample_inds'].append(i) # start of new sample
                # # print(path)
            except:
                self.bad_files.append(path)
                self.meta['path_list'].remove(path)
                print('bad',path)
                
        print(len(self.meta['shape_list']), 'valid samples')

        self.shape = self.__len__(),128,128

        # create h5 dataset, fill metadata, and transfer data from dm4 files to h5
        if overwrite or 'processed_data' not in h:
            if 'processed_data' in h: del h['processed_data']
            print('writing processed_data h5 dataset')
            h.create_dataset('processed_data',
                              shape=(sum( [shp[0]*shp[1] for shp in self.meta['shape_list']] ),
                                    128, 128),
                              dtype=float)
            
            for k,v in self.meta.items(): # write metadata
                    h['processed_data'].attrs[k] = v

            for i,data in enumerate(tqdm(self.data_list)): # fill data
                h['processed_data'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
                    np.log(np.array(data.reshape((-1, 128,128))) + 1)    
                    # da.log(data.reshape((-1, 128,128)) + 1) 

        # scaling
        print("fitting scaler...")
        # sample = h['processed_data'][np.arange(0,self.__len__(),10000)]
        self.scaler = StandardScaler()
        self.scaler.fit( h['processed_data'][0:self.__len__():5000].reshape(-1,128*128) )

        print('finding brightfield indices')
        # figure out mask
        self.BF_inds=[]
        for i in tqdm(range(len(self.data_list))):
            start = self.meta['particle_inds'][i]
            stop = self.meta['particle_inds'][i+1]
            img = h['processed_data'][start:stop:20].mean(0)
            thresh = img.mean()+img.std()*7
            self.BF_inds.append(np.argwhere(img>thresh).T)

        print('done')

    def __len__(self):
        return sum( [shp[0]*shp[1] for shp in self.meta['shape_list']] )
    
    def __getitem__(self,index): ## TODO: fix bf masking
        with h5py.File(self.h5_name, 'r+') as h5:
            img = h5['processed_data'][index]
            img = img.reshape(-1,128*128)
            img = self.scaler.transform(img)
            img = img.reshape(128,128)
            mean = img.mean()
            std = img.std()
            mask = abs(img)<mean+std*5
            # img[self.BF_inds]

            # return img
            return index,img*mask

    def open_h5(self):
        return h5py.File(self.h5_name, 'r+')

    def close_h5(self): 
        with h5py.File(self.h5_name, 'r+') as h5:
            h5.close()

    def view_log(self,index):
        with h5py.File(self.h5_name, 'r+') as h5:
            return h5['processed_data'][index]


        # # Determine which dask array to access based on the index
        # dask_array_index = index // (self.dask_arrays[0].shape[0] * self.dask_arrays[0].shape[1])
        # dask_array_offset = index % (self.dask_arrays[0].shape[0] * self.dask_arrays[0].shape[1])

        # # Load the diffraction pattern from dask array
        # diffraction_pattern = self.dask_arrays[dask_array_index][dask_array_offset // self.dask_arrays[0].shape[1],
        #                                                          dask_array_offset % self.dask_arrays[0].shape[1]]

        # # preprocessing

        # # Return the diffraction pattern as input and a dummy label (can be anything since we're not using it)
        # return diffraction_pattern, torch.tensor(0)

    def subtract_background(self,img,**kwargs):
        return img - gaussian_filter(img,**kwargs)
    
    def apply_scaler(self):
        h = h5py.File(self.h5_name,'r+')
        t,a,b,x,y = h['raw_data'].shape
        data = h['processed'][:].T.reshape(x*y,-1)
        print('standard scaling')
        data = StandardScaler().fit_transform(data)
        print('normalizing 0-1')
        data -= data.min(axis=0)
        data /= data.max(axis=0)
        print('writing to h5')
        h['processed'][:] = data.reshape(y,x,-1).T
        h.close()
        
    def apply_mask(self,bbox=None,center=None,radius=None):
        """apply a mask in the shape of a circle. 
        Arguments can either include a square around the brightfield or the center and radius

        Args:
            square (tuple, optional): (x1,x2,y1,y2) of bounding box. Defaults to None.
            center (tuple, optional): (x,y) indices of center of mask. Defaults to None.
            radius (int, optional): radius of mask. Defaults to None.
        """        
        h = h5py.File(self.h5_name,'r+')
        print('Masking')
        for sample,i in enumerate(tqdm()):
            h['processed'][i]=data*mask+(-mask+1)*h['processed'][i].mean()
        h.close()

    def apply_threshold(self,thresh):
        h = h5py.File(self.h5_name,'r+')
        args = np.argwhere(h['processed']>thresh)
        h['processed'][args] = thresh
        h.close()


class STEM_EELS_Dataset(Dataset):
    """Class for the STEM dataset.
    """
    def __init__(self,save_path,mode,kernel_size=8,
                 EELS_roi={'LL':[], 'HL': []},
                 overwrite_eels=False,
                 overwrite_diff=False,
                 **kwargs):
        """Initialization of the class.

        Args:
            mode (string): tells you how to return the data. Given as [{'processed','kernel_sum'}, {'eels', 'diff', 'both'}]
            save_path (string): path where the hyperspy file is located
            EELS_roi (dict): EELS region of interest. Separated by LL and HL. 
                Entries are lists of tuples, which are indices of regions of interest.
                Default {'LL':[], 'HL': []}.
            overwrite (bool): whether the delete and rewrite h5v file. Default False.
        """
    
        # assert the EELS_roi is alright
        assert len(EELS_roi['LL']+EELS_roi['HL'])>0, 'Set regions of interest for EELS'
        if mode[0]=='kernel_sum': self.mode = [mode[0]+f'_{kernel_size}', mode[1]]
        else: self.mode=mode #TODO: make setter; have the setter adjust the scalers when the mode changes.
        
        self.save_path = save_path
        self.h5_name = f'{save_path}/combined_data.h5'
        
        # create and sort metadata 
        print('fetching metadata...')
        self.meta = {}
        stem_path_list = glob.glob(f'{save_path}/TRI*/diff*/Diffraction SI.dm4')
        eels_ll_list = glob.glob(f'{save_path}/TRI*/eels*/EELS LL SI.dm4')
        eels_hl_list = glob.glob(f'{save_path}/TRI*/eels*/EELS HL SI.dm4')
        
        #functions for sorting
        def get_number(path): return int(path.split('/')[-2].split('-')[1])
        def get_particle(path): return path.split('/')[-3].split('TRI-8c-5-')[-1]
        stem_path_list.sort(key=get_number)
        stem_path_list.sort(key=get_particle)
        eels_ll_list.sort(key=get_number)
        eels_ll_list.sort(key=get_particle)
        eels_hl_list.sort(key=get_number)
        eels_hl_list.sort(key=get_particle)
        
        # init metadata
        self.meta['path_list'] = list(zip(stem_path_list,eels_ll_list,eels_hl_list))
        self.meta['particle_list'] = [] # names of particle 'folder(#)' 
        self.data_list = [] # TODO: make tuple (stem, eels)
        self.meta['shape_list'] = []
        # self.meta['scale'] = [] ## TODO: implement s.axes_manager['x'].scale, shape (x,y,sig)
        self.bad_files = [] # TODO: if stem or eels is bad, throw out both
        self.meta['particle_inds'] = [0]
        self.meta['sample_inds'] = [0]
                
        # go through data files and fill metadata
        for i,(dpath,lpath,hpath) in enumerate(tqdm(self.meta['path_list'])):
            try:
                diff = hs.load(dpath, lazy=True)
                ll = hs.load(lpath,lazy=True)
                hl = hs.load(hpath,lazy=True)
                self.data_list.append((diff.data, ll.data, hl.data))
                self.meta['particle_list'].append(f'AuCo({get_number(dpath):02d})')
                if mode[0]==f'kernel_sum': self.meta['shape_list'].append((diff.data.shape[0]-kernel_size,
                                                                                         diff.data.shape[1]-kernel_size))
                else: self.meta['shape_list'].append((diff.data.shape[0], diff.data.shape[1]))
                self.meta['particle_inds'].append(self.meta['particle_inds'][-1] + \
                                                  self.meta['shape_list'][-1][0]*self.meta['shape_list'][-1][1])
                if i==0: # TODO: just save the whole axis manager
                    self.meta['diff_scale'] = diff.axes_manager['x'].scale
                    self.meta['diff_dims'] = (diff.data.shape[2], diff.data.shape[3])
                    self.meta['loss_scale'] = ll.axes_manager['Energy loss'].scale
                    self.meta['loss_offsets'] = [ll.axes_manager['Energy loss'].offset, 
                                                 hl.axes_manager['Energy loss'].offset]
                if i>1 and self.meta['particle_list'][-1].split('(')[0] != self.meta['particle_list'][-2].split('(')[0]:
                    self.meta['sample_inds'].append(i) # start of new sample 
            except:
                self.bad_files.append((dpath,lpath,hpath))
                self.meta['path_list'].remove((dpath,lpath,hpath))
                print('bad:')
                print('\t',dpath)
                print('\t',lpath)
                print('\t',hpath)
        print(len(self.meta['shape_list']), 'valid samples')
        self.meta['length'] = self.meta['particle_inds'][-1]
        self.particle_count = len(self.meta['particle_list'])

        # get spectral data cropping region
        print('\ngetting spectral axis labels...')
        x = np.arange(1024)
        llx,hlx = self.get_raw_spectral_axis(1)
        self.raw_x_labels = np.stack([llx,hlx],axis=1)
        l = []
        maxlen=0
        for eV_range in EELS_roi['LL']:
            i0 = bisect_left(llx,eV_range[0])
            i1 = bisect_right(llx,eV_range[1])-1
            l.append((i0,i1,0))
            if i1-i0>maxlen: 
                maxlen = i1-i0
        for eV_range in EELS_roi['HL']:
            i0 = bisect_left(hlx,eV_range[0])
            i1 = bisect_right(hlx,eV_range[1])-1
            l.append((i0,i1,1))
            if i1-i0>maxlen: maxlen = i1-i0

        self.ll_i0 = bisect_left(llx,0)
        self.meta['eels_axis_inds'] = [(i[0], i[0]+maxlen, i[2]) for i in l] # make sure they are all the same length
        self.spec_len = self.meta['eels_axis_inds'][0][1] - self.meta['eels_axis_inds'][0][0] + 2
        self.eels_chs = len(self.meta['eels_axis_inds'])
        self.meta['eels_axis_labels'] = [np.linspace( self.raw_x_labels[inds[0]][inds[2]],
                                                self.raw_x_labels[inds[1]][inds[2]],
                                                self.spec_len) for inds in self.meta['eels_axis_inds']]

        # TODO: make this into a separate function
        if self.mode[0]=='processed_data': 
            self.write_processed(overwrite_eels, overwrite_diff)
        elif self.mode[0][:10]=='kernel_sum': 
            self.write_kernel_sum(kernel_size, overwrite_eels, overwrite_diff)
        
        # create h5 dataset, fill metadata, and transfer data from dm4 files to h5, and do 0 alignment
        with h5py.File(self.h5_name,'a') as h:
            self.scalers = [ {'diff': Pipeline([
                                        ('standard_scaler', StandardScaler()),
                                        ('minmax_scaler', MinMaxScaler())
                                    ]),
                            'eels': [ Pipeline([('standard_scaler', StandardScaler()),
                                                ('minmax_scaler', MinMaxScaler()) ]) for sc in range(self.eels_chs) ]
                                      } for p in self.meta['particle_list'] ]
             
            if self.mode[1]=='diff' or self.mode[1]=='both':
                print('fitting diff scalers...') # TODO: make custom class to fit scalars nd-wise
                tic = time.time()
                for i,p in enumerate(tqdm(self.meta['particle_list'])):
                    self.scalers[i]['diff'].fit( h[f'{self.mode[0]}/diff'][0:self.meta['length']-1:100].reshape(-1,512*512) )
                toc = time.time()
                print(f'\tDiffraction finished')
                
                ## figure out mask positions
                # TODO: do the kernel sum diffraction and fix the scaler to be more flexible to the mode
                print('finding brightfield indices...')
                self.BF_inds = []
                self.BF_mask=[]
                for i in tqdm( range( len( self.data_list ) ) ):
                    start = self.meta['particle_inds'][i]
                    stop = self.meta['particle_inds'][i+1]
                    img = h[f'{self.mode[0]}/diff'][start:stop:50,0].mean(0)
                    thresh = img.mean()+img.std()*30
                    inds = np.argwhere(img>thresh).T
                    self.BF_inds.append(inds)
                    
                    mask = np.zeros(img.shape)
                    mask[inds[0],inds[1]]=1
                    mask = binary_dilation(mask,footprint=disk(5))
                    mask = binary_erosion(mask,footprint=disk(2))
                    mask = binary_dilation(mask,footprint=disk(8))
                    mask = 1.-mask
                    self.BF_mask.append(mask)
            
            if self.mode[1]=='eels' or self.mode[1]=='both':
                ## find eels background for each sample
                print('finding High Loss background spectrum...')
                bkgs=[]
                for i in tqdm(range(len(self.meta['particle_list']))):
                    start = self.meta['particle_inds'][i]
                    stop = self.meta['particle_inds'][i+1]
                    ind_bkgs = []
                    for ind, spec_ind in enumerate(self.meta['eels_axis_inds']):
                        spec = h[f'{self.mode[0]}/eels'][start:stop,ind].mean(0)
                        x = np.linspace(0, self.spec_len-1, self.spec_len) # TODO: should I make the x-axis regular range or according to eels scale?
                        [a,b,c] = np.polyfit(x,spec,2)
                        ind_bkgs.append(a*x**2 + b*x + c) 
                    bkgs.append(ind_bkgs) 
                self.bkgs = bkgs # TODO: should I make this into an array?
                
                print('fitting eels scalers...') # TODO: make custom class to fit scalars nd-wise
                # toc = time.time()
                for i,scalers in enumerate(tqdm(self.scalers)):
                    start = self.meta['particle_inds'][i]
                    stop = self.meta['particle_inds'][i+1]
                    for ch,scaler in enumerate(scalers['eels']):
                        scaler.fit( h[f'{self.mode[0]}/eels'][start:stop,ch] - self.bkgs[i][ch])
                # tic = time.time()
                print(f'\tEELS finished')

            
        print('done')
        self.shape = ( (self.__len__(),1,512,512), (self.__len__(),self.eels_chs,self.spec_len) )

    def get_raw_spectral_axis(self,ind=0,inds=[(0,-1,0), (0,-1,1)]):
        x = np.arange(1024)
        llx = self.meta['loss_scale']*x + self.meta['loss_offsets'][0]
        hlx = self.meta['loss_scale']*x + self.meta['loss_offsets'][1]
        return llx,hlx
        
    def __len__(self): 
        with h5py.File(self.h5_name, 'r+') as h5:
            return self.meta['length']
        
    def write_means(self,overwrite_eels=False,overwrite_diff=False):
        '''assume you have already written the necessary processed datasets'''
        with h5py.File(self.h5_name,'a') as h:

            if overwrite_eels: # eels min max scaling
                print(f'\nwriting mean eels data in {self.mode[0]}')
                try: del h[f'{self.mode[0]}/eels_mean_image']
                except: pass
                try: del h[f'{self.mode[0]}/eels_mean_spectrum']
                except: pass
                h[f'{self.mode[0]}'].create_dataset('eels_mean_image', shape=(len(self),self.eels_chs), dtype=float)
                h[f'{self.mode[0]}'].create_dataset('eels_mean_spectrum', shape=(len(self.meta['shape_list']),self.eels_chs,self.spec_len), dtype=float)
                
                for i in tqdm(range(len(self.meta['shape_list']))):
                    start, stop = self.meta['particle_inds'][i], self.meta['particle_inds'][i+1]
                    if stop<len(self): _,eels = self[start:stop]
                    else: _,eels = self[start:]
                    
                    h[f'{self.mode[0]}/eels_mean_image'][start:stop,:] = [eels.mean(axis=-1) for spec in eels]
                    h[f'{self.mode[0]}/eels_mean_spectrum'][i] = sum(eels)/len(eels)
                    h.flush()
            
    def get_mean_image(self,p,e):
        with h5py.File(self.h5_name,'r+') as h:
            return h[f'{self.mode[0]}/eels_mean_image'][self.meta['particle_inds'][p]:self.meta['particle_inds'][p+1], 
                                                        e].reshape(self.meta['shape_list'][p])
            
    def get_mean_spectrum(self,p,e):
        with h5py.File(self.h5_name,'r+') as h:
            return h[f'{self.mode[0]}/eels_mean_spectrum'][p,e]
        
    def write_processed(self,overwrite_eels=False,overwrite_diff=False):
        with h5py.File(self.h5_name,'a') as h:
            if 'processed_data' not in h:
                print('\nwriting processed_data h5 datasets')
                overwrite_eels = True
                overwrite_diff = True
                h.create_group('processed_data')       
                
                for k,v in self.meta.items(): # write metadata
                        if isinstance(v[0],tuple):
                            h['processed_data'].attrs[k] = [tup[0] for tup in v]
                        else:
                            h['processed_data'].attrs[k] = v         
                                   
            if overwrite_eels: # eels min max scaling
                print('\nwriting eels datasets')
                del h['processed_data/eels']
                h['processed_data'].create_dataset('eels', shape=(self.length,self.eels_chs,self.spec_len), dtype=float)
                
                for i,data in enumerate(tqdm(self.data_list)): 
                    # find 0 peak shift
                    shift_0 = data[1].argmax(axis=2).flatten() - self.ll_i0
                    
                    for ind, spec_ind in enumerate(self.meta['eels_axis_labels']):
                        dset_slice = h['processed_data/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind]
                        istart = shift_0.rechunk(chunks=(1024,1)) + spec_ind[0]
                        data_ = data[spec_ind[2]+1].reshape((-1,1024)).rechunk(chunks=(1024,1024,))
                        
                        # function to slice data according to peak
                        def slice_data(dat, start, block_info=None, investigate=False):
                            block_shape = dat.shape
                            if dat.size == 0 or start.size == 0:
                                raise ValueError("Received an empty block")
                            # Initialize an empty array for the result
                            new_chunk = np.empty((block_shape[0], self.spec_len))
                            for i in range(block_shape[0]):
                                start_idx = start[i]
                                sliced = dat[i, start_idx:start_idx + self.spec_len]
                                new_chunk[i] = da.log(sliced+1)#.rechunk(block_shape[0],self.spec_len)
                            new_chunk = (new_chunk - new_chunk.min())/new_chunk.max()
                            return new_chunk 
                        
                        result = da.map_blocks(slice_data, data_, istart, 
                                               dtype=data_.dtype,
                                               drop_axis = [1],
                                               new_axis=[1], 
                                               chunks = (1024,self.spec_len), )     
                        da.store(result, dset_slice)
                        h['processed_data/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind] = dset_slice
                        h.flush()
                              
            if overwrite_diff: # take the log. min max scaling
                print('\nwriting diff datasets')
                del h['processed_data/diff']
                h['processed_data'].create_dataset('diff', shape=(self.length,1,512, 512), dtype=float)
                for i,data_ in enumerate(tqdm(self.data_list)): 
                    # 4%|▎         | 1/27 [00:18<08:10, 18.85s/it]\
                    data__ = da.log(data_[0].reshape((-1,1,512,512)) + 1)
                    data__ = (data__ - data__.min())/data__.max()
                    h['processed_data/diff'][self.meta['particle_inds'][i]:\
                                                        self.meta['particle_inds'][i+1]] = data__
                    h.flush()             
                   
    def write_kernel_sum(self,ksize,overwrite_eels=False,overwrite_diff=False):
        # kshape_list = []
        # kparticle_inds = [0]
        # for ds,es,_ in self.meta['shape_list']:
        #     kshape_list.append((es[0]-ksize, es[1]-ksize, es[2]))
        #     kparticle_inds.append(kparticle_inds[-1] + \
        #                                         kshape_list[-1][0]*kshape_list[-1][1])
        # klength = sum( [shp[0]*shp[1] for shp in kshape_list])

        # create h5 dataset, fill metadata, and transfer data from dm4 files to h5, and do 0 alignment
        with h5py.File(self.h5_name,'a') as h:
            if f'kernel_sum_{ksize}' not in h:
                print(f'\nwriting kernel_sum_{ksize} h5 datasets')
                overwrite_eels = True
                # overwrite_diff = True
                h.create_group(f'kernel_sum_{ksize}')       
                
            # for k,v in self.meta.items(): # write metadata
            #         if not isinstance(v,list):
            #             h[f'kernel_sum_{ksize}'].attrs[k] = v
            #         elif isinstance(v[0],tuple):
            #             h[f'kernel_sum_{ksize}'].attrs[k] = [tup[0] for tup in v]
            #         else:
            #             h[f'kernel_sum_{ksize}'].attrs[k] = v                    
            # h[f'kernel_sum_{ksize}'].attrs.__setitem__('shape_list', kshape_list)
            # h[f'kernel_sum_{ksize}'].attrs.__setitem__('particle_inds', kparticle_inds)
            # h[f'kernel_sum_{ksize}'].attrs.__setitem__('length', klength)

            if overwrite_eels: # eels min max scaling
                print(f'\nwriting eels data with kernel size {ksize}')
                try: del h[f'kernel_sum_{ksize}/eels']
                except: pass
                h[f'kernel_sum_{ksize}'].create_dataset('eels', shape=(len(self),self.eels_chs,self.spec_len), dtype=float)
                
                for i,data in enumerate(tqdm(self.data_list)):
                    # find 0 peak shift
                    shift_0 = data[1].argmax(axis=2).flatten() - self.ll_i0
                    
                    for ind, spec_ind in enumerate(self.meta['eels_axis_inds']):
                        dset_slice = h[f'kernel_sum_{ksize}/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind]
                        istart = shift_0.rechunk(chunks=(1024,1)) + spec_ind[0]
                        data_ = data[spec_ind[2]+1].reshape((-1,1024)).rechunk(chunks=(1024,1024,))
                        
                        # function to slice data according to peak
                        def slice_data(dat, start, block_info=None, investigate=False):
                            block_shape = dat.shape
                            if dat.size == 0 or start.size == 0:
                                raise ValueError("Received an empty block")
                            # Initialize an empty array for the result
                            new_chunk = np.empty((block_shape[0], self.spec_len))
                            for i in range(block_shape[0]):
                                start_idx = start[i]
                                sliced = dat[i, start_idx:start_idx + self.spec_len]
                                new_chunk[i] = da.log(sliced+1)#.rechunk(block_shape[0],self.spec_len)
                            new_chunk = (new_chunk - new_chunk.min())/new_chunk.max()
                            return new_chunk 
                    
                    for ind, spec_ind in enumerate(self.meta['eels_axis_inds']):
                        dset_slice = h[f'kernel_sum_{ksize}/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind]
                        istart = shift_0.rechunk(chunks=(1024,1)) + spec_ind[0]
                        ch_size = int(data[1].shape[0]*data[1].shape[0]/16)
                        data_ = data[spec_ind[2]+1].reshape((-1,1024)).rechunk(chunks=(ch_size,1024,))
                        
                        # function to slice data according to peak
                        def slice_data(dat, start, block_info=None, investigate=False):
                            block_shape = dat.shape
                            if dat.size == 0 or start.size == 0:
                                raise ValueError("Received an empty block")
                            # Initialize an empty array for the result
                            new_chunk = np.empty((block_shape[0], self.spec_len))
                            for i in range(block_shape[0]):
                                start_idx = start[i]
                                sliced = dat[i, start_idx:start_idx + self.spec_len]
                                new_chunk[i] = da.log(sliced+1)#.rechunk(block_shape[0],self.spec_len)
                            return new_chunk 
                        
                        result = da.map_blocks(slice_data, data_, istart, 
                                               dtype=data_.dtype,
                                               drop_axis = [1],
                                               new_axis=[1], 
                                               chunks = (ch_size,self.spec_len), )                  
                        result = result.reshape(data[1].shape[0],data[1].shape[1],-1).rechunk(chunks=(data[1].shape[0],data[1].shape[1],self.spec_len))
                        
                        # print(kshape_list[i],result)
                        def kernel_sum(dat, block_info=None, investigate=False):
                            block_shape=dat.shape
                            if dat.size == 0:
                                raise ValueError("Received an empty block")
                            new_chunk = np.empty((self.meta['shape_list'][i][0], self.meta['shape_list'][i][1], block_shape[-1]))
                            for n in range(block_shape[0]-ksize):
                                for m in range(block_shape[1]-ksize):
                                    new_chunk[n,m] = dat[n:n+ksize,m:m+ksize].sum(axis=(0,1))
                            return new_chunk
                        
                        result = da.map_blocks(kernel_sum, result, 
                                               dtype=result.dtype,
                                               chunks=(self.meta['shape_list'][i][0], self.meta['shape_list'][i][1],self.spec_len)) 
                        result = result.reshape(-1,self.spec_len)
                        # result.compute()
                        da.store(result, dset_slice)
                        h[f'kernel_sum_{ksize}/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind] = dset_slice
                        h.flush()
                        
    def get_index(self,p,a,b):
        return self.meta['particle_inds'][p]+a*self.meta['diff_dims'][0]+b
    

    def open_h5(self):
        return h5py.File(self.h5_name, 'r+')
    
    def get_particle(self,i):
        start = self.meta['particle_inds'][i]
        stop = self.meta['particle_inds'][i+1]
        with h5py.File(self.h5_name, 'r+') as h:
            diff = h['processed_data/diff'][start:stop]
            ll = h['processed_data/ll'][start:stop]
            hl = h['processed_data/hl'][start:stop]
        return diff,ll,hl
    
    def flattened_coord(self,x,y,p):
        sh = self.meta['shape_list'][p][0]
        return x*sh[0]+y

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_single_index(index)
        elif isinstance(index, (slice, list, np.ndarray)):
            return self._get_slice_or_fancy_index(index)
        elif isinstance(index, tuple):
            return self._get_tuple_index(index)
        else:
            raise TypeError("Index must be an integer, slice, tuple, or array-like.")
    
    def _get_single_index(self, index):
        with h5py.File(self.h5_name, 'r+') as h5:
            return self._retrieve_data(h5, [index])

    def _get_slice_or_fancy_index(self, index):
        with h5py.File(self.h5_name, 'r+') as h5:
            if isinstance(index, slice):
                indices = list(range(*index.indices(len(self))))
            else:  # list or np.ndarray
                indices = list(index)
            return self._retrieve_data(h5, indices)

    def _get_tuple_index(self, index):
        if len(index) == 1:
            if isinstance(index[0], int):
                return self._get_single_index(index[0])
            else:
                return self._get_slice_or_fancy_index(index[0])
        else:
            indices = []
            for idx in index:
                if isinstance(idx, int):
                    indices.append([idx])
                elif isinstance(idx, slice):
                    indices.append(list(range(*idx.indices(len(self)))))
                elif isinstance(idx, (list, np.ndarray)):
                    indices.append(list(idx))
                else:
                    raise TypeError("Tuple elements must be an integer, slice, or array-like")

            # Combine indices for multi-dimensional indexing
            combined_indices = np.ix_(*indices)
            flat_indices = [tuple(idx) for idx in np.nditer(combined_indices, flags=['refs_ok'], order='C')]

            with h5py.File(self.h5_name, 'r+') as h5:
                return self._retrieve_data(h5, flat_indices)
    
    def _retrieve_data(self, h5, indices):
        indices_list = []
        diff_list = []
        eels_list = []

        for idx in indices:
            if isinstance(idx, tuple):
                # Use only the first index from the first axis
                first_index = idx[0]
                i = bisect_right(self.meta['particle_inds'], first_index) - 1
                indices_list.append(first_index)

                if self.mode[1] == 'eels' or self.mode[1] == 'both':
                    # Case where index has 2 elements: (n_idx, eels_ch_idx)
                    n_idx, eels_ch_idx = idx[0],idx[1]

                    # Retrieve the specific slice from the dataset
                    eels_ = h5[f'{self.mode[0]}/eels'][n_idx, eels_ch_idx]

                    # Apply background subtraction and scaling to the slice
                    eels = self.scalers[i]['eels'][eels_ch_idx].transform(
                        (eels_ - self.bkgs[i][eels_ch_idx]).reshape(-1,self.spec_len) # treat as single sample
                    ).squeeze()
                    if len(idx) == 3: eels = eels[idx[2]]
                    eels_list.append(eels)

            else:
                # Handle single or flattened indices
                i = bisect_right(self.meta['particle_inds'], idx) - 1
                indices_list.append(idx)
                
                if self.mode[1] == 'diff' or self.mode[1] == 'both':
                    diff = h5[f'{self.mode[0]}/diff'][idx]
                    diff = diff.reshape(-1, 512 * 512)
                    diff = self.scalers[i]['diff'].transform(diff)
                    diff = diff.reshape(1, 512, 512) * self.BF_mask[i]
                    diff_list.append(diff)
                
                if self.mode[1] == 'eels' or self.mode[1] == 'both':
                    eels_ = h5[f'{self.mode[0]}/eels'][idx]
                    eels = np.array([
                        self.scalers[i]['eels'][ch].transform(
                            (eels_[ch] - self.bkgs[i][ch]).reshape(1, -1)
                        ).squeeze()
                        for ch in range(self.eels_chs)
                    ])
                    eels_list.append(eels)

        # Convert lists to numpy arrays
        indices_array = np.array(indices_list).squeeze()
        diff_array = np.array(diff_list).squeeze() if diff_list else None
        eels_array = np.array(eels_list).squeeze() if eels_list else None

        # Return only the non-empty arrays
        return_tuple = (indices_array,)
        if diff_array is not None:
            return_tuple += (diff_array,)
        if eels_array is not None:
            return_tuple += (eels_array,)

        return return_tuple

    