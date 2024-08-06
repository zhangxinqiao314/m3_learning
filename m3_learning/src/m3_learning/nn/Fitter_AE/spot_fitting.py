import h5py
import py4DSTEM
import tqdm
import xml.etree.ElementTree as ET
import dask.array as da
import re
import time

import torch

class Stacked_4DSTEM(): 
    '''
    saves data in h5 file with the following format:
    / <root>
    |---attrs: <>   
    |---group: '/metadata' 
        |---attrs: <>
        |---group: '/metadata/20deg' 
            ...
        |---group: '/metadata/30deg' 
            ...
        ...
        ...
    |---dataset: '/raw_data' (-1,a,b,x,y)
        |---attrs: <>   
    |---dataset: '/processed_data' (-1,1,x,y)
        |---attrs: <>   
    '''
    def __init__(self,h5_filepath,file_names,diff_list,meta_list) -> None:
        self.h5_filepath = h5_filepath
        self.file_names = file_names
        self.diff_list = diff_list
        self.meta_list = meta_list
        
        self.max = 0
        self.min = 0
        
    @staticmethod
    def parse_xml_to_dict(element):
    # Function to recursively parse XML into a dictionary
        parsed_dict = {}
        if element.attrib:
            parsed_dict.update(element.attrib)
        if element.text and element.text.strip():
            parsed_dict['text'] = element.text.strip()
        for child in element:
            child_dict = Stacked_4DSTEM.parse_xml_to_dict(child)
            if child.tag in parsed_dict:
                if not isinstance(parsed_dict[child.tag], list):
                    parsed_dict[child.tag] = [parsed_dict[child.tag]]
                parsed_dict[child.tag].append(child_dict)
            else:
                parsed_dict[child.tag] = child_dict
        return parsed_dict

    @staticmethod
    def parse_xml_file(xml_file):
    # Function to parse XML file and convert it into a dictionary
    # TODO: want to mke this indexing?
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return {root.tag: Stacked_4DSTEM.parse_xml_to_dict(root)}
    
    @staticmethod
    def add_dict_to_h5(group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                try: sub_group = group.create_group(key)
                except: sub_group = group[key]
                Stacked_4DSTEM.add_dict_to_h5(sub_group, value)
            elif isinstance(value, list):
                try: group.create_dataset(key, data=value)
                except: group[key]=value
            else:
                group.attrs[key] = value
    
    @staticmethod    
    def log(h5_file, in_name, out_name):
        assert in_name in h5_file.keys() and out_name in h5_file.keys()
        with h5py.File(h5_file, 'r+') as f:
            data = f[in_name]
            dask_data = da.from_array(data,chunks=(1, 128, 128, 128, 128))
            log_scaled_data = da.log1p(dask_data) 
            log_scaled_data = log_scaled_data.rechunk((1, 1, 128, 128))
            log_scaled_data = log_scaled_data.reshape((-1, 1, 128, 128))
            f[out_name] = log_scaled_data
    
    @staticmethod    
    def standard_scale(h5_file, in_name, out_name):
        with h5py.File(h5_file, 'r') as f:
            raw_data = f[in_name]
            raw_data_dask = da.from_array(raw_data, chunks=(1, 128, 128, 128, 128))
            
            def standard_scaling(scan):
                mean = scan.mean()
                std = scan.std()
                return (scan - mean) / std
            
            standardized_data = raw_data_dask.map_blocks(standard_scaling)
            standardized_data = standardized_data.rechunk((1, 1, 128, 128))
            standardized_data = standardized_data.reshape((-1, 1, 128, 128))
            f[out_name] = standardized_data
            
    @staticmethod     
    def min_max_scale(h5_file, in_name, out_name):
        with h5py.File(h5_file, 'r') as f:
            raw_data = f[in_name]
            raw_data_dask = da.from_array(raw_data, chunks=(1, 128, 128, 128, 128))
            def min_max_scaling(scan):
                min_val = scan.min()
                max_val = scan.max()
                return (scan - min_val) / (max_val - min_val)
            min_max_scaled_data = raw_data_dask.map_blocks(min_max_scaling)
            min_max_scaled_data = min_max_scaled_data.rechunk((1, 1, 128, 128))
            min_max_scaled_data = min_max_scaled_data.reshape((-1, 1, 128, 128))
            f[out_name] = min_max_scaled_data
    
                           
    def write_metadata(self,overwrite=False):
        metadata_dict = {}
        for i,temp_label in enumerate(tqdm(self.file_names)):
            metadata = Stacked_4DSTEM.parse_xml_file(self.meta_list[i])
            metadata_dict[temp_label] = metadata
        with h5py.File(self.h5_filepath, 'a') as h:
            if overwrite: 
                if 'metadata' in h.keys(): del h['overwrite']
            Stacked_4DSTEM.add_dict_to_h5(metadata_dict)
        
    def write_raw_data(self,overwrite=False):
        with h5py.File(self.h5_filepath, 'a') as h:
            if overwrite: 
                if 'raw_data' in h.keys(): del h['raw_data']
            try: 
                raw_data = h.create_dataset('raw_data', shape=([len(self.diff_list)]+self.orig_shape))
                for i in tqdm(range(len(self.temps_list))):
                    datacube = py4DSTEM.import_file(self.diff_list[i])
                    meta_dict = Stacked_4DSTEM.parse_xml_file(self.meta_list[i])
                    raw_data[i] = datacube.data
            except: 
                print('Raw data already written. Set overwrite=True')
                
    def write_processed_data(self,overwrite=False,
                             dset_name='processed_data',
                             process_list = [Stacked_4DSTEM.log,
                                             Stacked_4DSTEM.standard_scale,
                                             Stacked_4DSTEM.min_max_scale]
                             ):
        with h5py.File(self.h5_filepath, 'a') as h:
            if overwrite: 
                if dset_name in h.keys(): del h[dset_name]
            try: 
                processed_data = h.create_dataset(dset_name, shape=([len(self.diff_list)]+self.orig_shape))
                for process in process_list: #TODO: add timing info?
                    print(process, end='')
                    tic = time.time()
                    process(self.h5_filepath,'raw_data',dset_name)
                    print(f': {tic-time.time()} s')
            except: 
                print('Raw data already written. Set overwrite=True')
    
    def __len__(self):
        with h5py.File(self.h5_filepath, 'a') as h: return h['processed_data'].shape[0]
    
    def __getitem__(self,i):
        with h5py.File(self.h5_filepath, 'a') as h:
            return h['processed_data'][i]
    
# TODO: make dataloader
class Spot_Dataset(Stacked_4DSTEM):
    def __init__(self,
                 h5_filepath, diff_list, meta_list, # (,file_names) for super
                 file_names = ['20deg', '30deg', '50deg', '80deg', '100deg', '120deg', '130deg', '140deg', '150deg', 
                               'T_140Crampdown', 'T_120Crampdown', 'T_100Crampdown', 'T_80Crampdown', 'T_30C_rampdown', 'T_50Crampdown', 'T_20C_rampdown'],
                 temps_list = [20, 30, 50, 80, 100, 120, 130, 140, 150, 140, 120, 100, 80, 50, 30, 20], # C
                 tile_coords = [(63, 62), (53,31),(43,51),(33,71), (53,81),(73,92),(83,72), (93,52),(73,41)],
                 orig_shape = [128,128,128,128],
                 interp_size = [128,128,256,256],
                 device = 'cuda:0'
                 ) -> None:
        super.__init__(h5_filepath,file_names,diff_list,meta_list)
        self.device = device
        def extract_number(string):
            match = re.search(r'\d+', string)
            return int(match.group()) if match else None
        self.temps_list = [extract_number(name) for name in file_names]
        self.tile_coords = tile_coords
        self.tile_slices = [(slice(coord[0]-self.r, coord[0]+self.r), 
                             slice(coord[1]-self.r, coord[1]+self.r)) for coord in self.tile_coords])
        self.orig_shape = orig_shape
        self.interp_size = interp_size
        self.masking = True
        
    @staticmethod
    def get_tiles(diff, slices):
        '''diff shape (128,128)'''
        tile_list = []
        for sl in slices:
            tile_list.append(diff[..., sl[0], sl[1]])
        return tile_list
            
    def get_tiles(self,diff):
        ''' call static get_tiles method and use self.tile_coords, self.r
        needs to be able to be able to handel data of shape (*,128,128) 
        and return resI ult as a tensor of shape (#tiles,*,128,128)'''
        tiles = Spot_Dataset.get_tiles_static(diff, self.tile_slices)
        tiles = torch.stack([torch.tensor(tile, device=self.device) for tile in tiles])
        return tiles
    
    # def __len__(self) already did in super init
    
    def __getitem__(self,i):
        '''output of shape (#tiles,*,128,128).
        With dataloader: (batchsize, #tiles,*,128,128)'''
        with h5py.File(self.h5_filepath, 'a') as h:
            diff = h['processed_data'][i]
            return self.get_tiles[diff]


# TODO: make ae

# TODO: make spot visualizer (plot diffraction and have spot locations)

# TODO: make tiling function