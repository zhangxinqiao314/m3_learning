import numpy as np
import hyperspy.api as hs
import h5py
from skimage.draw import disk
import dask.array as da
from dask.diagnostics import ProgressBar

import pyNSID
import os
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import Normalizer,StandardScaler
from tqdm import tqdm as tqdm
import glob
import torch
from bisect import bisect_left,bisect_right
import time
from skimage.morphology import binary_dilation, binary_erosion,disk
from pdb import set_trace as bp
import matplotlib.pyplot as plt


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
    def __init__(self,save_path,
                 EELS_roi={'LL':[], 'HL': []},
                 overwrite_diff=False,
                 overwrite_eels=False,
                 **kwargs):
        """Initialization of the class.

        Args:
            save_path (string): path where the hyperspy file is located
            EELS_roi (dict): EELS region of interest. Separated by LL and HL. 
                Entries are lists of tuples, which are indices of regions of interest.
                Default {'LL':[], 'HL': []}.
            overwrite (bool): whether the delete and rewrite h5v file. Default False.
        """
    
        # assert the EELS_roi is alright
        assert len(EELS_roi['LL']+EELS_roi['HL'])>0, 'Set regions of interest for EELS'
        
        self.save_path = save_path
        self.h5_name = f'{save_path}/combined_data.h5'

        # create and sort metadata 
        print('fetching metadata...')
        self.meta = {}
        stem_path_list = glob.glob(f'{save_path}/*/diff*/Diffraction SI.dm4')
        eels_ll_list = glob.glob(f'{save_path}/*/eels*/EELS LL SI.dm4')
        eels_hl_list = glob.glob(f'{save_path}/*/eels*/EELS HL SI.dm4')
        
        def get_number(path):
            return int(path.split('/')[-2].split('-')[-1])
        def get_particle(path):
            return path.split('/')[-3].split('TRI-8c-5-')[-1]
        
        stem_path_list.sort(key=get_number)
        stem_path_list.sort(key=get_particle)
        eels_ll_list.sort(key=get_number)
        eels_ll_list.sort(key=get_particle)
        eels_hl_list.sort(key=get_number)
        eels_hl_list.sort(key=get_particle)
        
        self.meta['path_list'] = list(zip(stem_path_list,eels_ll_list,eels_hl_list))
        self.meta['particle_list'] = [] # names of particle 'folder(#)' TODO: make 2 digit counting number
        self.data_list = [] # TODO: make tuple (stem, eels)
        self.meta['shape_list'] = []
        self.meta['scale'] = [] ## TODO: implement s.axes_manager['x'].scale, shape (x,y,sig)
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
                self.meta['shape_list'].append((diff.data.shape, ll.data.shape, hl.data.shape))
                self.meta['particle_inds'].append(self.meta['particle_inds'][-1] + \
                                                    diff.data.shape[0]*diff.data.shape[1])
                self.meta['scale'].append( (diff.axes_manager['x'].scale, 
                                            [ll.axes_manager['Energy loss'].scale, ll.axes_manager['Energy loss'].offset], 
                                            [hl.axes_manager['Energy loss'].scale, hl.axes_manager['Energy loss'].offset]) )
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
        self.length = sum( [shp[0][0]*shp[0][1] for shp in self.meta['shape_list']])
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
            if i1-i0>maxlen: 
                maxlen = i1-i0

        self.ll_i0 = bisect_left(llx,0)
        self.meta['eels_axis_labels'] = [(i[0], i[0]+maxlen, i[2]) for i in l] # make sure they are all the same length
        self.spec_len = self.meta['eels_axis_labels'][0][1] - self.meta['eels_axis_labels'][0][0] +1
        self.eels_chs = len(self.meta['eels_axis_labels'])
        
        self.shape = ( (self.__len__(),512,512), (self.__len__(),self.spec_len) )

        # create h5 dataset, fill metadata, and transfer data from dm4 files to h5, and do 0 alignment
        with h5py.File(self.h5_name,'a') as h:
            if 'processed_data' not in h:
                print('\nwriting processed_data h5 datasets')
                overwrite_eels = False
                overwrite_diff = False
                h.create_group('processed_data')       
                
                for k,v in self.meta.items(): # write metadata
                        if isinstance(v[0],tuple):
                            h['processed_data'].attrs[k] = [tup[0] for tup in v]
                        else:
                            h['processed_data'].attrs[k] = v         
                                   
            if overwrite_eels:
                print('\nwriting eels datasets')
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
                            # print(block_info)
                            block_shape = dat.shape
                            
                            # Check if the block is empty (should not be the case)
                            if dat.size == 0 or start.size == 0:
                                raise ValueError("Received an empty block")
                            
                            # Initialize an empty array for the result
                            new_chunk = np.empty((block_shape[0], self.spec_len))
                            for i in range(block_shape[0]):
                                # Extract the start index for this row
                                start_idx = start[i]
                                # Perform the slicing
                                sliced = dat[i, start_idx:start_idx + self.spec_len]
                                new_chunk[i] = np.log(sliced+1)#.rechunk(block_shape[0],self.spec_len)
                            return new_chunk 
                        result = da.map_blocks(slice_data, data_, istart, 
                                               dtype=data_.dtype,
                                               drop_axis = [1],
                                               new_axis=[1], 
                                               chunks = (1024,self.spec_len),
                                               )     
                        da.store(result, dset_slice)
                        h['processed_data/eels'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1],ind] = dset_slice
                        h.flush()
                                
                                
            if overwrite_diff:
                print('\nwriting diff datasets')
                h['processed_data'].create_dataset('diff', shape=(self.length,1,512, 512), dtype=float)
                for i,data_ in enumerate(tqdm(self.data_list)): 
                    # 4%|▎         | 1/27 [00:18<08:10, 18.85s/it]
                    h['processed_data/diff'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
                        np.log(np.array(data_[0].reshape((-1,1,512,512))) + 1)
                    h.flush()
                    
                  
            print('fitting scalers...') # TODO: make custom class to fit scalars nd-wise
            self.scalers = {'diff': StandardScaler(),
                            'eels': [StandardScaler() for sc in range(self.eels_chs)] }
            tic = time.time()
            self.scalers['diff'].fit( h['processed_data/diff'][0:self.length-1:100].reshape(-1,512*512) )
            toc = time.time()
            print(f'\tDiffraction finished: {abs(tic-toc)} s')
            
            [scaler.fit( h['processed_data/eels'][0:self.length-1:25, 0]) for scaler in self.scalers['eels']]
            tic = time.time()
            print(f'\tEELS finished {abs(tic-toc)} s')

            ## figure out mask positions
            print('finding brightfield indices...')
            self.BF_inds = []
            self.BF_mask=[]
            for i in tqdm(range(len(self.data_list))):
                start = self.meta['particle_inds'][i]
                stop = self.meta['particle_inds'][i+1]
                img = h['processed_data/diff'][start:stop:50,0].mean(0)
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


            ## find eels background for each sample
            print('finding High Loss background spectrum...')
            bkgs=[]
            for i in tqdm(range(len(self.meta['particle_list']))):
                start = self.meta['particle_inds'][i]
                stop = self.meta['particle_inds'][i+1]
                ind_bkgs = []
                for ind, spec_ind in enumerate(self.meta['eels_axis_labels']):
                    spec = h['processed_data/eels'][start:stop,ind].mean(0)
                    [a,b,c] = np.polyfit(np.arange(self.spec_len),spec,2)
                    x = np.linspace(0, self.spec_len-1, self.spec_len) # TODO: should I make the x-axis regular range or according to eels scale?
                    ind_bkgs.append(a*x**2 + b*x + c) 
                bkgs.append(ind_bkgs) 
            self.bkgs = bkgs # TODO: should I make this into an array?
            
        print('done')

    def __len__(self):
        return self.length
    
    # TODO: eels background subtration
    ## TODO: fancy indexing methods
    def __getitem__(self,index):
        with h5py.File(self.h5_name, 'r+') as h5:
            # which particle are we on
            i = bisect_right(self.meta['particle_inds'],index)-1
            
            img = h5['processed_data/diff'][index]
            img = img.reshape(-1,512*512)
            img = self.scalers['diff'].transform(img)
            img = img.reshape(512,512)
            # TODO: dask implementation? only if rly slow
            eels_ = h5['processed_data/eels'][index] # shape (num_ch, spec_len)
            
            eels = [self.scalers['eels'][ch].transform(eels_[ch].reshape(1,-1)).squeeze() - self.bkgs[i][ch] \
                    for ch in range(self.eels_chs)]
            
            return index,img*self.BF_mask[i], eels # TODO: does this need to be returned as array?
        
    def get_raw_spectral_axis(self,ind=0):
        x = np.arange(1024)
        llx = self.meta['scale'][ind][1][0]*x + \
                    self.meta['scale'][ind][1][1]
        hlx = self.meta['scale'][ind][2][0]*x + \
                    self.meta['scale'][ind][2][1]
        return llx,hlx

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
    
    
# class NDStandardScaler(TransformerMixin):
#     def __init__(self, **kwargs):
#         self._scaler = StandardScaler(copy=True, **kwargs)
#         self._orig_shape = None

#     def fit(self, X, **kwargs):
#         X = np.array(X)
#         # Save the original shape to reshape the flattened X later
#         # back to its original shape
#         if len(X.shape) > 1:
#             self._orig_shape = X.shape[1:]
#         X = self._flatten(X)
#         self._scaler.fit(X, **kwargs)
#         return self

#     def transform(self, X, **kwargs):
#         X = np.array(X)
#         X = self._flatten(X)
#         X = self._scaler.transform(X, **kwargs)
#         X = self._reshape(X)
#         return X

#     def _flatten(self, X):
#         # Reshape X to <= 2 dimensions
#         if len(X.shape) > 2:
#             n_dims = np.prod(self._orig_shape)
#             X = X.reshape(-1, n_dims)
#         return X

#     def _reshape(self, X):
#         # Reshape X back to it's original shape
#         if len(X.shape) >= 2:
#             X = X.reshape(-1, *self._orig_shape)
#         return X
