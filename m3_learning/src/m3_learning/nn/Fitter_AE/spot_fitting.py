import h5py
import py4DSTEM
from tqdm import tqdm
import xml.etree.ElementTree as ET
import dask.array as da
import re
import time
from dask.diagnostics import ProgressBar
import numpy as np
from .. import Fitter_functions
from .. import Regularization

import torch


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imshow_tensor(x):
    import matplotlib.pyplot as plt
    plt.imshow(x.detach().cpu().numpy());
    plt.colorbar()
    plt.show()

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
        self.orig_shape = list(py4DSTEM.import_file(self.diff_list[0]).data.shape)
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
                print('start dict',group,key)
                Stacked_4DSTEM.add_dict_to_h5(sub_group, value)
                print('done dict',group,key)
            elif isinstance(value, list):
                value = {str(i): v for i,v in enumerate(value)}
                try: group.create_dataset(key, data=value)
                except: group[key]=value
                finally:
                    try: sub_group = group.create_group(key)
                    except: sub_group = group[key]
                    print('start list',group,key)
                    Stacked_4DSTEM.add_dict_to_h5(sub_group, value)
                    print('done list',group,key)
            else:
                try: group.attrs[key] = group.attrs[key].append(value)
                except: group.attrs[key] = value
    
    # TODO: parallelize scaling methods
    @staticmethod    
    def log(h5_filepath, in_name, out_name, **kwargs):
        with h5py.File(h5_filepath, 'r+') as f:
            s = f[in_name].shape
            s_ = f['raw_data'].shape
            assert in_name in f.keys() and out_name in f.keys()
            if len(s)==5:
                for i in tqdm(range(s[0])):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = np.log1p(f[in_name][i]-f[in_name][i].min()).reshape(-1,1,s_[3],s_[4])
            else: 
                for i in tqdm(range(int(s[0]/s_[0]/s_[1]))):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = np.logp1(f[in_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]]-f[in_name][i].min()).reshape(-1,1,s_[3],s_[4])

    @staticmethod    
    def standard_scale(h5_filepath, in_name, out_name):
        with h5py.File(h5_filepath, 'r+') as f:
            s = f[in_name].shape
            assert in_name in f.keys() and out_name in f.keys()
            def standard_scaling(scan):
                mean = scan.mean()
                std = scan.std()
                return (scan - mean) / std
            s_ = f['raw_data'].shape
            assert in_name in f.keys() and out_name in f.keys()
            if len(s)==5:
                for i in tqdm(range(s[0])):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = standard_scaling(f[in_name][i]).reshape(-1,1,s_[3],s_[4])
            else: 
                for i in tqdm(range(int(s[0]/s_[2]/s_[1]))):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = standard_scaling(f[in_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]].reshape(-1,1,s_[3],s_[4]))
               
    @staticmethod     
    def min_max_scale(h5_filepath, in_name, out_name):
        with h5py.File(h5_filepath, 'r+') as f:
            s = f[in_name].shape
            assert in_name in f.keys() and out_name in f.keys()
            def min_max_scaling(scan):
                min_val = scan.min()
                max_val = scan.max()
                return (scan - min_val) / (max_val - min_val)
            s_ = f['raw_data'].shape
            assert in_name in f.keys() and out_name in f.keys()
            if len(s)==5:
                for i in tqdm(range(s[0])):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = min_max_scaling(f[in_name][i]).reshape(-1,1,s_[3],s_[4])
            else: 
                for i in tqdm(range(int(s[0]/s_[2]/s_[1]))):
                    f[out_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]] = min_max_scaling(f[in_name][i*s_[2]*s_[1]:(i+1)*s_[2]*s_[1]].reshape(-1,1,s_[3],s_[4]))
                                  
    def write_metadata(self,overwrite=False): #TODO: wtf doesn work
        metadata_dict = {}
        for i,temp_label in enumerate(self.file_names):
            metadata = Stacked_4DSTEM.parse_xml_file(self.meta_list[i])
            metadata_dict[temp_label] = metadata
        # with h5py.File(self.h5_filepath, 'a') as h:
        #     if overwrite: 
        #         if 'metadata' in h.keys(): del h['metadata']
        #         h.create_group('metadata')
        #     Stacked_4DSTEM.add_dict_to_h5(h['metadata'],metadata_dict)
        self.metadata = metadata_dict
        
    def write_raw_data(self,overwrite=False):
        with h5py.File(self.h5_filepath, 'a') as h:
            if overwrite:
                if 'raw_data' in h.keys(): del h['raw_data']
                raw_data = h.create_dataset('raw_data', shape=([len(self.diff_list)]+self.orig_shape))
                for i,diff in enumerate(tqdm(self.diff_list)):
                    datacube = py4DSTEM.import_file(diff)
                    raw_data[i] = datacube.data
            else: print('Raw data already written. Set overwrite=True')
    
    def get_raw_data(self,i):
        with h5py.File(self.h5_filepath, 'a') as h:
            return h['raw_data'][i]
    
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
                #  temps_list = [20, 30, 50, 80, 100, 120, 130, 140, 150, 140, 120, 100, 80, 50, 30, 20], # C
                 tile_coords = [(63, 62), (53,31),(43,51),(33,71), (53,81),(73,92),(83,72), (93,52), (73,41)],
                 r = 10,
                 orig_shape = [128,128,128,128],
                 interp_size = [128,128,256,256],
                 device = 'cuda:0' ) -> None:
        super().__init__(h5_filepath,file_names,diff_list,meta_list)
        self.device = device
        def extract_number(string):
            match = re.search(r'\d+', string)
            return int(match.group()) if match else None
        self.temps_list = [extract_number(name) for name in file_names]
        self.r = r
        self.tile_coords = tile_coords
        self.tile_slices = [(slice(coord[0]-self.r, coord[0]+self.r), 
                             slice(coord[1]-self.r, coord[1]+self.r)) for coord in self.tile_coords]
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
            
    def _get_tiles(self,diff):
        ''' call static get_tiles method and use self.tile_coords, self.r
        needs to be able to be able to handel data of shape (*,128,128) 
        and return resI ult as a tensor of shape (#tiles,*,128,128)'''
        tiles = Spot_Dataset.get_tiles(diff, self.tile_slices)
        tiles = torch.stack([torch.tensor(tile, device=self.device) for tile in tiles])
        return torch.tensor(diff),tiles
    
    # def __len__(self) already did in super init
    
    def __getitem__(self,i):
        '''output of shape (#tiles,*,128,128).
        With dataloader: (batchsize, #tiles,*,128,128)'''
        with h5py.File(self.h5_filepath, 'a') as h:
            diff = h['processed_data'][i]
            diff,tiles = self._get_tiles(diff)
            return diff.to(self.device),tiles.to(self.device)

# TODO: make ae utils


from torch.autograd import Variable

class PV_CC_ST_AE_utils():
    
    def __init__(self, dset, device='cuda:0', lr=3e-5,batch_size=32,
                 # PV stuff
                 pv_fit_channels=128, pv_enc_channels = 128,
                 pv_ln_coef = 0.01, pv_embedding_size = 10, # A, I_b, x, y, wx, wy, nu, t
                 pv_limits =[1, 1, 10, 10, 10, 10, 0.5], # A, I_b, x, y, wx, wy
                 fitter_function=Fitter_functions.generate_pseudovoigt,
                 masking='circle' # options: 'circle', 'square', 'threshold'
        
        ):    # TODO: add more hyperparams
        '''  '''
        self.dset = dset 
        self.seed = 12
        self.device = dset.device
        self.temps_list = dset.temps_list
        self.r = dset.r
        self.tile_coords = dset.tile_coords
        self.tile_slices = dset.tile_slices
        self.orig_shape = dset.orig_shape
        self.interp_size = dset.interp_size
        self.masking = True
        self.file_names = dset.file_names
        self.diff_list = dset.diff_list
        self.meta_list = dset.meta_list
        self.tile_coords = dset.tile_coords
        self.lr = lr
        self.batch_size=batch_size
        
        # PV stuff
        self.pv_fit_channels = pv_fit_channels
        self.pv_enc_channels = pv_enc_channels
        self.pv_ln_coef = pv_ln_coef
        self.pv_embedding_size = pv_embedding_size
        self.pv_limits = pv_limits
        self.fitter_function = fitter_function
        self.masking = masking
        
        # CCSTAE stuff
        self.st_pool_list = [4,4,2]
        self.embedding_size = len(self.dset.tile_coords*2)
        self.reg_coef = 0
        self.scale_coef = 0
        self.shear_coef = 0
        self.norm_order = 1
        self.scale_penalty = 0
        self.shear_penalty = 0
        self.mask_list = None
        self.batch_para = 1
        self.weighted_mse = True
        self.weight_coef = 2
        self.upgrid_img = False
        self.soft_threshold = 1.5
        self.hard_threshold = 3
        self.con_div = 15
        
        # Training stuff
        self.start_epoch = 0
        self.primary_loss_function = Regularization.HigherOrderLoss(order=2)
        self.ln_loss_function = Regularization.LN_loss(ln_parm=2)
        self.mask_thresh = 1e-2
        self.num_pxs = 25

    def compile_model(self):
        # self.pv_encoder = 
        
        self.pv_model = spot.make_models( 
                                    enc_channels=self.pv_fit_channels,
                                    filter_channels=self.pv_enc_channels, 
                                    kernel_size = self.r*2,
                                    embedding_size=self.pv_embedding_size, 
                                    device=self.device,
                                    lr=self.lr, 
                                    limits=self.pv_limits,
                                    masking=self.masking)
        #TODO: put in the hyperparams
        self.encoder = CC_ST_AE.Encoder(original_step_size=[self.orig_shape[-2],self.orig_shape[-1]],
                                    pool_list=self.st_pool_list,
                                    conv_size=128,
                                    device=self.device,
                                    rotation=False,
                                    rotate_clockwise=False,
                                    translation=False,
                                    interpolate=True,
                                    up_size=256,
                                    reduced_size=18, # number of spots*2
                                    num_base=18, # number of spots*2
                                    emb_function='spots',
                                    coords=self.tile_coords,
                                    r=self.r
                                    ).to(self.device)

        self.pv_cc_st_ae = CC_ST_AE.PV_Joint(pv_model=self.pv_model, 
                                        coords=self.tile_coords,
                                        r=self.r,
                                        orig_size=self.orig_shape,
                                        encoder=self.encoder,
                                        decoder=None,
                                        device=self.device,
                                        interp_size=self.interp_size,
                                        masking=self.masking).to(self.device)

        self.optimizer = optim.Adam(self.pv_cc_st_ae.parameters(), lr=self.lr)

    def load_weights(self,path_checkpoint,return_checkpoint=False,embedding_h5_filepath=''):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
            return_checkpoint (bool, Optional): whether to return the checkpoint loaded. Default False
            embedding_h5_filepath (string, Optional): name of the embedding h5 file with checkpoints written in
        
        Returns:
            checkpoint (Optional)
        """
        self.checkpoint = torch.load(path_checkpoint)  
        self.path_checkpoint = path_checkpoint
        self.pv_cc_st_ae.load_state_dict(self.checkpoint["pv_cc_st_ae"])
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.pv_encoder.load_state_dict(self.checkpoint['pv_encoder'])
        self.pv_model.load_state_dict(self.checkpoint['pv_model'])
        self.encoder.load_state_dict(self.checkpoint["encoder"])
        self.epoch = self.checkpoint["epoch"]+1
        self.loss_dict = self.checkpoint['loss_dict']
        self.checkpoint_name = self.path_checkpoint.split('/')[-1].split('.pkl')[0] 
        self.primary_loss_function.load_state_dict(self.checkpoint['loss_function']),
        self.ln_loss_function.load_state_dict(self.checkpoint['ln_loss_function']),
        self.mask_thresh = self.checkpoint['mask_thresh']
        self.csv_file_path = f'{os.path.split(self.path_checkpoint)[0]}/training_losses.csv'
    
    def Train(self,mask_thresh=0.01,i=25, masking=True,
                pv_loss_order=1, pv_ln_coef=0, pv_ln_parm=2,
                reg_coef=0.,soft_threshold=0.1,norm_order=2,
                max_learning_rate=1e-4,
                seed=12,
                epochs=100,
                with_scheduler=True,
                ln_parm=1,
                folder_path='./pc_cc_st_ae/',
                best_train_loss=None,
                loss_order_scheduler = lambda x: x//20+1,
              ln_coef_scheduler = lambda x: 5e-4*(x//15),
              ln_parm_scheduler = lambda x: 2/(x//10+1),
              lr_scheduler = lambda x: 3e-4/(x//5+1) ):
    
        dataloader = torch.utils.data.DataLoader(self.dset, batch_size=self.batch_size, shuffle=True)
    
        make_folder(folder_path)
        self.csv_file_path = f'{folder_path}/training_losses.csv'

        # set seed
        torch.manual_seed(self.seed)

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None: best_train_loss = float('inf')

        # get datetime for saving
        today = datetime.datetime.now()
        date = today.strftime('(%Y-%m-%d, %H:%M)')
        
        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            self.primary_loss_function = spot.HigherOrderLoss(order = loss_order_scheduler(epoch))
            self.optimizer = torch.optim.Adam(self.pv_cc_st_ae.parameters(), lr=lr_scheduler(epoch))
            self.ln_loss_function = spot.LN_loss(ln_parm=ln_parm_scheduler(epoch))
            
            loss_dict = self.loss_function(dataloader, 
                                            reg_coef=0.,
                                            norm_order=norm_order )
            
            print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {loss_dict['train_loss']:.2e}')
            print('.............................')
            date = today.strftime('(%Y-%m-%d, %H:%M)')

            # if with_scheduler: schedular.step()
            if best_train_loss > loss_dict['train_loss']:
                best_train_loss = loss_dict['train_loss']
                
            checkpoint = {
                "pv_cc_st_ae": self.pv_cc_st_ae.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'pv_encoder': self.pv_encoder.state_dict(), 
                'pv_model': self.pv_model.state_dict(),
                "encoder": self.encoder.state_dict(),
                "epoch": epoch,
                'loss_dict': loss_dict,
                'loss_function': self.primary_loss_function.state_dict(),
                'ln_loss_function': self.ln_loss_function.state_dict(),
                "mask_thresh": self.mask_thresh,
                'loss': loss_dict,
                }
            
            if epoch >= 0:
                # lr_ = format(optimizer.param_groups[0]['lr'], '.5f')
                file_path = folder_path + '/Weight_' + date +\
                    f'epoch:{epoch:04d}_lr:'+\
                    f'_trainloss:{loss_dict["train_loss"]:.2e}.pkl'
                torch.save(checkpoint, file_path)
                
                self.write_loss_csv(self.csv_file_path,epoch,loss_dict)

    def loss_function(self,dataloader,mask_thresh=0.01,i=25,masking=True,
                        pv_ln_coef=0,
                        reg_coef=0.,soft_threshold=0.1,norm_order=2):
        # set the train mode
        self.pv_cc_st_ae.train()
        pv_loss=0
        pv_ln_loss=0
        train_loss = 0
        L2_loss = 0
        Scale_Loss = 0
        Shear_Loss = 0
        
        for i,data in enumerate(tqdm(dataloader)):
            loss_=0
            x,tiles = data
            self.optimizer.zero_grad()
            
            # run model
            (   affine_base,
                ae_base,
                predicted_input,
                ae_coords,
                scaler_shear,
                rotation,
                translation,
                adj_mask,
                raw_masks,
                base_masks,
                x_inp, 
            ), (
                g_embedding, 
                g_pred,
                mask_loss, 
                mask_thresh, 
            num_pxs) = self.pv_cc_st_ae(data)

            loss_ = mask_loss
            
            
            # PV LN loss
            if self.pv_ln_coef>0: 
                pv_ln = self.ln_loss_function(g_embedding[:,:,0],self.pv_ln_coef)
                loss_+=pv_ln
                pv_ln += pv_ln.item()
                
            # CCAEST
            
            # calculate l norm from generated base 
            if reg_coef>0: 
                l2_loss = (reg_coef*torch.norm(ae_base.squeeze(), p=norm_order) / x.shape[0])
                loss_ += l2_loss
                L2_loss += l2_loss.item()
            
            # calculate scale penalty
            scale_loss = self.scale_coef * (
                torch.mean(F.relu(abs(scaler_shear[:, 0, 0] - 1) - self.scale_penalty))
                + torch.mean(F.relu(abs(scaler_shear[:, 1, 1] - 1) - self.scale_penalty)))
            
            # calculate shear penalty
            shear_loss = self.shear_coef * torch.mean(
                F.relu(abs(scaler_shear[:, 0, 1]) - self.shear_penalty))
            
            # add l norm, scale penalty and shear penalty to loss 
            loss_ += (scale_loss + shear_loss)
            
            # calculate MSE. DEAL LATER with multiple bases
            # intersect = torch.argwhere( new_list[0] + old_list-1 )
            # b_,x_,y_ = intersect.T
            mask = torch.argwhere( base_masks )
            raw =  torch.argwhere( raw_masks )
            b_,x_,y_ = mask.T
            b__, x__,y__ = raw.T
            mask_factor = len(b_)/(base_masks[0].shape[-2]*base_masks[0].shape[-1])
            loss_ = loss_ + F.mse_loss(  # compare base prediction
                    ae_base[b_,:,x_,y_],
                    affine_base[b_,:,x_,y_],
                reduction="mean") + F.mse_loss( # compare recon and input
                    predicted_input[b__,:,x__,y__],
                    x_inp[b__,:,x__,y__],
                reduction="mean")

            if loss_ > soft_threshold: # For keeping loss on same oom to not have to change lr? 
                # loss += 0 
                loss_ += F.l1_loss( # compare base prediction
                    ae_base[b_,:,x_,y_],
                    affine_base[b_,:,x_,y_],
                reduction="mean") + F.l1_loss( # compare recon and input
                    predicted_input[b__,:,x__,y__],
                    x_inp[b__,:,x__,y__],
                reduction="mean")

            train_loss += loss_.item()/mask_factor            
            pv_loss += mask_loss.item()
            Scale_Loss += scale_loss.item()
            Shear_Loss += shear_loss.item()
            
            loss_.backward()
            self.optimizer.step() 

        loss_dict = {"train_loss": train_loss / len(dataloader),
                     'mask_loss': pv_loss/len(dataloader),
                    'pv_ln_loss': ( pv_ln_loss/len(dataloader) ),
                    "l2_loss": L2_loss / len(dataloader),
                    "scale_loss": Scale_Loss / len(dataloader),
                    "shear_loss": Shear_Loss / len(dataloader),}
        
        print('masked pixels:', i)   
        return loss_dict
    
    def write_loss_csv(self,csv_file_path,epoch,loss_dict):
        if epoch==0:
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file,fieldnames=list(loss_dict.keys())+['epoch'])
                writer.writeheader()
        
        with open(csv_file_path, 'a', newline='') as file:    
            loss_dict.update({'epoch': epoch})
            writer = csv.DictWriter(file, fieldnames=loss_dict.keys())
            writer.writerow(loss_dict)
        
    def check_emb_h5(self,length,folder_path='./pc_cc_st_ae_embedding/',emb_h5_filename='embeddings.h5',verbose=True):        
        make_folder(folder_path)
        self.emb_h5_path = f'{folder_path}/{emb_h5_filename}'

        try: h = h5py.File(self.emb_h5_path,'r+')
        except: h = h5py.File(self.emb_h5_path,'w')
        
        try: h.create_dataset(f'embedding_{self.checkpoint_name}',data = np.zeros([length,
                                                                                   self.embedding_size]))
        except: pass 
        try: h.create_dataset(f'scaleshear_{self.checkpoint_name}',data = np.zeros([length,6]))
        except: pass
        try: h.create_dataset(f'rotation_{self.checkpoint_name}',data = np.zeros([length,6]))
        except: pass
        try: h.create_dataset(f'translation_{self.checkpoint_name}',data = np.zeros([length,6]))
        except: pass        
        # try: h.create_dataset(f'pv_embedding_{self.checkpoint_name}',data = np.zeros([length,
        #                                                                               len(self.tile_coords),
        #                                                                             self.pv_fit_channels,
        #                                                                             self.pv_embedding_size]))
        # except: pass
        
        if verbose: [print(key) for key in h.keys()]
        
        h.flush()
        h.close()

    def get_embeddings(self,folder_path='./pc_cc_st_ae_embedding/',emb_h5_filename='embeddings.h5'):
            dataloader = torch.utils.data.DataLoader(self.dset, batch_size=self.batch_size, shuffle=False)
        
            self.check_emb_h5(len(self.dset),folder_path,emb_h5_filename)
            
            with h5py.File(self.emb_h5_path,'r+') as h:
                embedding_ = h[f'embedding_{self.checkpoint_name}']
                scale_shear_ = h[f'scaleshear_{self.checkpoint_name}']
                rotation_ = h[f'rotation_{self.checkpoint_name}'] 
                translation_ = h[f'translation_{self.checkpoint_name}']
                pv_embedding_ = h[f'pv_embedding_{self.checkpoint_name}']
                
                for i,x in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                    full, tiles=x
                    with torch.no_grad():
                        test_values = [Variable(full.to(self.device)).float(), 
                                    Variable(tiles.to(self.device)).float() ]
                        (   affine_base,
                            ae_base,
                            predicted_input,
                            ae_coords,
                            scaler_shear,
                            rotation,
                            translation,
                            adj_mask,
                            raw_masks,
                            base_masks,
                            x_inp, 
                        ), (
                            pv_embedding, #[b_*s_, ch, numpar]
                            pv_pred,
                            mask_loss, 
                            mask_thresh, 
                        num_pxs) = self.pv_cc_st_ae(x)
                        
                        embedding_[(i)*len(full):(i+1)*len(full), :] = ae_coords.reshape(len(full),-1).cpu()
                        scale_shear_[(i)*len(full):(i+1)*len(full), :] = scaler_shear.reshape(-1,6).cpu().detach().numpy()
                        rotation_[(i)*len(full):(i+1)*len(full), :] = rotation.reshape(-1,6).cpu().detach().numpy()
                        translation_[(i)*len(full):(i+1)*len(full), :] = translation.reshape(-1,6).cpu().detach().numpy()
                        # pv_embedding_[(i)*len(full):(i+1)*len(full)] = pv_embedding.reshape(len(full),len(self.tile_coords),
                                                                                            # -1,self.pv_embedding_size).cpu().detach().numpy()
                    
                    
                    h.flush()
            h.close()

# TODO: make spot visualizer (plot diffraction and have spot locations)

def draw_bounding_box(ax,coord,r,**kwargs):
    x, y = coord
    rect = patches.Rectangle((y-r,x-r), 2*r, 2*r, 
                             linewidth=kwargs.get('linewidth', 1), 
                             edgecolor=kwargs.get('edgecolor'), 
                             linestyle=kwargs.get('linestyle'),
                             facecolor='none')
    ax.add_patch(rect)
    ax.plot(y,x,
            color=kwargs.get('edgecolor'), 
            marker=kwargs.get('marker'))

def draw_bounding_boxes(dset,i,
                        colors=['#ffffff',  # White
                                '#000000',  # Black
                                '#00ffff',  # Cyan
                                '#ff00ff',  # Magenta
                                '#ff7f0e',  # Orange
                                '#ff0000',  # Bright Red
                                '#e377c2',  # Bright Pink
                                # '#2ca02c',  # Bright Green
                                '#8c564b',   # Dark Brown
                                '#ffd700',  # Gold
                                '#1e90ff'   # Dodger Blue
                                ],
                        with_axis=False,
                        with_colorbar=False,
                        **kwargs):
    ''' kwargs include the lineshape and the center coord shape'''
    diff,spots = dset[i]
    try: diff = diff.detach().cpu().numpy()
    except: pass
    ax = plt.subplot()
    ax_im = ax.imshow(diff[0])
    for c,coord in enumerate(dset.tile_coords):
        draw_bounding_box(ax,coord,dset.r,
                          edgecolor=colors[c],
                          linestyle='-',
                          marker='o')
    if not with_axis: ax.axis('off')
    if with_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax_im, cax=cax,**kwargs)
        
    return ax

def show_tile(dset,i,j,
              with_axis=False,
              with_colorbar=False,
              **kwargs):
    diff,spots = dset[i]
    try: spots = spots.detach().cpu().numpy()
    except: pass
    ax = plt.subplot()
    ax_im = ax.imshow(spots[j][0])
    if not with_axis: ax.axis('off')
    if with_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax_im, cax=cax,**kwargs)
    return ax
