import numpy as np
import hyperspy.api as hs
import h5py
from skimage.draw import disk
import dask.array as da
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
    def __init__(self, save_path,overwrite=False,**kwargs):
        """Initialization of the class.

        Args:
            save_path (string): path where the hyperspy file is located
        """
        self.save_path = save_path
        self.h5_name = f'{save_path}/combined_data.h5'

        # create and sort metadata 
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

        print('fetching metadata...')
        self.meta['particle_list'] = [] # names of particle 'folder(#)' TODO: make 2 digit counting number
        self.data_list = [] # TODO: make tuple (stem, eels)
        self.meta['shape_list'] = []
        self.meta['scale'] = [] ## TODO: implement s.axes_manager['x'].scale, shape (x,y,sig)
        self.bad_files = [] # TODO: if stem or eels is bad, throw out both
        self.meta['particle_inds'] = [0]
        self.meta['sample_inds'] = [0]

        # go through data files and fill metadata
        # TODO: why doesn't assert false work?
        for i,(dpath,lpath,hpath) in enumerate(tqdm(self.meta['path_list'])):
            try:
                diff = hs.load(dpath, lazy=True)
                ll = hs.load(lpath,lazy=True)
                hl = hs.load(hpath,lazy=True)
                self.data_list.append((diff.data, ll.data, hl.data))
                self.meta['particle_list'].append(f'AuCo({get_number(dpath):02d})')
                # (real,diff), 
                self.meta['shape_list'].append((diff.data.shape, ll.data.shape, hl.data.shape))
                self.meta['particle_inds'].append(self.meta['particle_inds'][-1] + \
                                                    diff.data.shape[0]*diff.data.shape[1])
                self.meta['scale'].append((diff.axes_manager['x'].scale, 
                                           ll.axes_manager['Energy loss'].scale, 
                                           hl.axes_manager['Energy loss'].scale))
                
                if i>1 and self.meta['particle_list'][-1].split('(')[0] != self.meta['particle_list'][-2].split('(')[0]:
                    self.meta['sample_inds'].append(i) # start of new sample
                # # print(path)
            except:
                self.bad_files.append((dpath,lpath,hpath))
                self.meta['path_list'].remove((dpath,lpath,hpath))
                print('bad:')
                print('\t',dpath)
                print('\t',lpath)
                print('\t',hpath)
                
        print(len(self.meta['shape_list']), 'valid samples')
        self.length = sum( [shp[0][0]*shp[0][1] for shp in self.meta['shape_list']])
        self.shape = self.__len__(),128,128
        
        # create h5 dataset, fill metadata, and transfer data from dm4 files to h5
        with h5py.File(self.h5_name,'a') as h:
            if overwrite or 'processed_data' not in h:
                if 'processed_data' in h: del h['processed_data']
                print('writing processed_data h5 datasets')
                
                processed_group = h.create_group('processed_data')                
                processed_group.create_dataset('diff', shape=(self.length,512, 512), dtype=float)                
                processed_group.create_dataset('ll', shape=(self.length,1024), dtype=float)                
                processed_group.create_dataset('hl', shape=(self.length,1024), dtype=float)
                
                for k,v in self.meta.items(): # write metadata
                        if isinstance(v[0],tuple):
                            h['processed_data/diff'].attrs[k] = [tup[0] for tup in v]
                            h['processed_data/ll'].attrs[k] = [tup[1] for tup in v]
                            h['processed_data/hl'].attrs[k] = [tup[2] for tup in v]
                        else:
                            h['processed_data/diff'].attrs[k] = v
                            h['processed_data/ll'].attrs[k] = v
                            h['processed_data/hl'].attrs[k] = v
                        

                for i,data in enumerate(tqdm(self.data_list)): # fill data
                    
                    h['processed_data/diff'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
                        np.log(np.array(data[0].reshape((-1, 512,512))) + 1)    
                        
                    h['processed_data/ll'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
                        np.array(data[1].reshape((-1, 1024)))
                          
                    h['processed_data/hl'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
                        np.array(data[2].reshape((-1, 1024)))

            # scaling
            print("fitting scalers...")
            # sample = h['processed_data'][np.arange(0,self.__len__(),10000)]
            self.scalers = {'diff': StandardScaler(),
                            'll': StandardScaler(),
                            'hl': StandardScaler() }
            tic = time.time()
            self.scalers['diff'].fit( h['processed_data/diff'][0:self.length-1:100].reshape(-1,512*512) )
            toc = time.time()
            print(f'Diffraction finished: {abs(tic-toc)} s')
            
            self.scalers['ll'].fit( h['processed_data/ll'][:])
            tic = time.time()
            print(f'Low Loss finished {abs(tic-toc)} s')
            
            self.scalers['hl'].fit( h['processed_data/hl'][:])
            toc=time.time()
            print(f'High Loss finished {abs(tic-toc)} s') 

            ## TODO: figure out mask positions in the init function
            print('finding brightfield indices...')
            # figure out mask
            self.BF_inds = []
            self.BF_mask=[]
            for i in tqdm(range(len(self.data_list))):
                start = self.meta['particle_inds'][i]
                stop = self.meta['particle_inds'][i+1]
                img = h['processed_data/diff'][start:stop:50].mean(0)
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

            ## TODO: eels subtract background for each sample
            print('finding High Loss background spectrum...')
            # figure out mask
            self.HL_bkgs=[]
            for i in tqdm(range(len(self.data_list))):
                start = self.meta['particle_inds'][i]
                stop = self.meta['particle_inds'][i+1]
                spec = h['processed_data/hl'][start:stop].mean(0)
                [a,b,c] = np.polyfit(np.arange(1024),spec,2)
                x = np.linspace(0,1023,1024)
                self.HL_bkgs.append(a*x**2 + b*x + c) 
                    
        print('done')

    def __len__(self):
        return self.length
    
    # TODO: eels background subtration
    ## TODO: fancy indexing methods
    def __getitem__(self,index):
        with h5py.File(self.h5_name, 'r+') as h5:
            i = bisect_right(self.meta['particle_inds'],index)-1
            
            img = h5['processed_data/diff'][index]
            img = img.reshape(-1,512*512)
            img = self.scalers['diff'].transform(img)
            img = img.reshape(512,512)
            
            ll = h5['processed_data/ll'][index].reshape(1,1024)
            ll = self.scalers['ll'].transform(ll).squeeze()

            hl = h5['processed_data/hl'][index]#-self.HL_bkgs[i]
            hl = self.scalers['hl'].transform(hl.reshape(1,1024)).squeeze()
            
            return index,img*self.BF_mask[i],ll,hl
            
            
            # mean = img.mean()
            # std = img.std()
            # mask = abs(img)<mean+std*5

            # return img
            # return index,img*mask

    def open_h5(self):
        return h5py.File(self.h5_name, 'r+')

    def view_log(self,index):
        with h5py.File(self.h5_name, 'r+') as h5:
            return h5['processed_data'][index]

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

    # # def preprocess(self,mask_center=False,crop=False,sub_bkg=False,thresh=False):
    # #     # mask_center
    # #     # try:
    # #         (bx1,bx2,by1,by2)=mask_center
    # #         center = int((bx1+bx2)/2),int((by1+by2)/2)
    # #         r = max(abs(bx2-center[0])+1,abs(by2-center[1])+1)
    # #         rr,cc = disk(center,r)

    # #         # mask=np.zeros((self.data.shape[-2],self.data.shape[-1]))
    # #         # mask[rr,cc] = 0
    # #         # self.processed = da.map_blocks(lambda x: x[:,:,:, rr, cc] * 0.0, 
    # #         #                                self.processed, 
    # #         #                                dtype=self.processed.dtype,
    # #         #                                chunks=self.processed.chunks)
    # #         # self.processed=test
    # #     #     for i,j in list(zip(rr,cc)):
    # #     #             self.processed[:,:,:,i,j] = 0
