import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..STEM_AE.STEM_AE import ConvBlock, IdentityBlock
from .spot_fitting import *
from ....m3_learning.viz import make_folder
from ....m3_learning.nn.Regularization import LN_loss #, ContrastiveLoss, DivergenceLoss, Sparse_Max_Loss,

import os
from datetime import date
import wandb
from tqdm import tqdm


# TODO: do you want this or the encoder from stead ae folder?
class Multiscale2DFitter(nn.Module):
    def __init__(self, function, x_data, input_channels, num_params, num_fits, limits=[1, 1, 10, 10, 10, 10, 0.5], scaler=None, 
                 post_processing=None, device="cuda", loops_scaler=None, flatten_from=1, 
                 x1_ch_list=[8,6,4], x1_pool=64, x2_pool_list=[16,8,4], x2_ch_list=[8,16],
                 dense_list=[24,16,8], **kwargs):
        """_summary_


        Args:
            function (_type_): _description_
            x_data (_type_): _description_
            input_channels (_type_): number of channels in orig data. ex. 2 channels: high loss, low loss
            num_params (_type_): number of parameters needed to generate the fit
            num_fits (_type_): the number of peaks to include
            limits (list):  limits for [A_g, A_l, x, sigma, gamma, nu]
            scaler (_type_, optional): _description_. Defaults to None.
            post_processing (_type_, optional): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to "cuda".
            loops_scaler (_type_, optional): _description_. Defaults to None.
            flatten_from (int, optional): _description_. Defaults to 1.
        """        

        self.input_channels = input_channels
        self.scaler = scaler
        self.function = function
        self.x_data = x_data
        self.device = device
        self.num_params = num_params
        self.num_fits = num_fits
        self.loops_scaler = loops_scaler
        self.flat_dim = flatten_from

        super().__init__()
        
        self.limits = limits

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels,
                      out_channels=x1_ch_list[0], kernel_size=7),
            nn.SELU(), nn.Conv2d(in_channels=x1_ch_list[0], #8
                                 out_channels=x1_ch_list[1], kernel_size=7),
            nn.SELU(), nn.Conv2d(in_channels=x1_ch_list[1], #6
                                 out_channels=x1_ch_list[2], kernel_size=5), #4
            nn.SELU(), nn.AdaptiveAvgPool2d(x1_pool)
        )
        x1_outsize = x1_pool*x1_ch_list[-1]
        # print(x1_outsize)
        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(x1_outsize, x1_outsize//2), nn.SELU(),
            nn.Linear(x1_outsize//2, x1_outsize//4), nn.SELU(),
            nn.Linear(x1_outsize//4, x1_outsize//8), nn.SELU(),
        )
        xfc_outsize = x1_outsize//8
        # print(xfc_outsize)
        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=x1_ch_list[2], out_channels=x2_ch_list[0], kernel_size=5), nn.SELU(), 
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[0], kernel_size=5), nn.SELU(), 
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[0], kernel_size=5), nn.SELU(), 
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[0], kernel_size=5), nn.SELU(), 
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[0], kernel_size=5), nn.SELU(), 
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[0], kernel_size=5),  nn.SELU(), 
            nn.AdaptiveAvgPool1d(x2_pool_list[0]),  # Adaptive pooling layer
            nn.Conv1d(in_channels=x2_ch_list[0], out_channels=x2_ch_list[1], kernel_size=3), nn.SELU(), 
            nn.AdaptiveAvgPool1d(x2_pool_list[1]),  # Adaptive pooling layer
            nn.Conv1d(in_channels=x2_ch_list[1], out_channels=x1_ch_list[-1], kernel_size=3), nn.SELU(), 
            nn.AdaptiveAvgPool1d(x2_pool_list[2]),  # Adaptive pooling layer
        )
        x2_outsize = x2_pool_list[2]*x1_ch_list[-1]
        # print(x2_outsize)
        # Flatten layer
        self.flatten_layer = nn.Flatten() # flatten to batch,-1

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(xfc_outsize+x2_outsize, dense_list[0]), nn.SELU(),
            nn.Linear(dense_list[0], dense_list[1]), nn.SELU(),
            nn.Linear(dense_list[1], self.num_params*num_fits*input_channels),
        )

    def forward(self, x, n=-1, return_sum=True):
        # output shape - samples, spec_channels, frequency
        # x = torch.swapaxes(x, 1, 2)
        s=x.shape
        x = self.hidden_x1(x)
        # print('x1 shape:', x.shape)
        xfc = torch.reshape(x, (x.shape[0], -1))  # batch size, channels, features
        xfc = self.hidden_xfc(xfc)
        # print('xfc shape:', xfc.shape)

        # batch size, spec_channels, timesteps
        # x = torch.reshape(x, (x.shape[0], -1))
        x = self.hidden_x2(x)
        # print('x2 shape:', x.shape)
        cnn_flat = self.flatten_layer(x)
        # print('flat:',cnn_flat.shape)

        encoded = torch.cat((cnn_flat, xfc), -1)  # merge dense and 1d conv.
        # print('encoded shape:', encoded.shape)
        embedding = self.hidden_embedding(encoded)
        embedding = embedding.reshape(x.shape[0]*self.input_channels, self.num_fits, self.num_params)
        
        # unscaled_param = embedding

        # if self.scaler is not None:
        #     # corrects the scaling of the parameters
        #     unscaled_param = (
        #         embedding *
        #         torch.tensor(self.scaler.var_ ** 0.5).cuda()
        #         + torch.tensor(self.scaler.mean_).cuda()
        #     )
        # else:
            # unscaled_param = embedding

        # passes to the pytorch fitting function
        fits,params = self.function(
            embedding, self.x_data, limits=self.limits, device=self.device,return_params=True)
        # print('fitted shape:', fits.shape)
        
        if not return_sum:
            return params, fits
        
        out = fits.sum(axis=1).reshape(s)

        # Does the post processing if required
        if self.post_processing is not None:
            out = self.post_processing.compute(out)
        else:
            out = out

        if self.loops_scaler is not None:
            out_scaled = (out - torch.tensor(self.loops_scaler.mean).cuda()) / torch.tensor(
                self.loops_scaler.std).cuda()
        else:
            out_scaled = out
        
        return params,out_scaled

        if self.training == True:
            return out_scaled, unscaled_param
        if self.training == False:
            # this is a scaling that includes the corrections for shifts in the data
            embeddings = (unscaled_param.cuda() - torch.tensor(self.scaler.mean_).cuda()
                          )/torch.tensor(self.scaler.var_ ** 0.5).cuda()
            return out_scaled, embeddings, unscaled_param


class PV_Encoder_2D(nn.Module):
    """Encoder block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, original_step_size, pooling_list, embedding_size, conv_size, num_fits):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(PV_Encoder_2D, self).__init__()

        blocks = []

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]

        number_of_blocks = len(pooling_list)

        blocks.append(ConvBlock(t_size=conv_size,
                                n_step=original_step_size))
        blocks.append(IdentityBlock(
            t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(
            pooling_list[0], stride=pooling_list[0]))
        
        original_step_1 = [0,0]
        for i in range(1, number_of_blocks):
            original_step_size = [
                original_step_size[0] // pooling_list[i - 1],
                original_step_size[1] // pooling_list[i - 1],
            ]
            if i==1: original_step_1 = original_step_size
            blocks.append(ConvBlock(t_size=conv_size,
                                    n_step=original_step_size))
            blocks.append(
                IdentityBlock(t_size=conv_size, n_step=original_step_size)
            )
            blocks.append(nn.MaxPool2d(
                pooling_list[i], stride=pooling_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        original_step_size = [
            original_step_size[0] // pooling_list[-1],
            original_step_size[1] // pooling_list[-1],
        ]
        input_size = original_step_size[0]*original_step_size[1] + original_step_1[0]*original_step_1[1]

        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, num_fits, 3, stride=1, padding=1, padding_mode="zeros"
        )

        self.relu_1 = nn.ReLU()

        self.dense = nn.Linear(input_size, embedding_size)

    def forward(self, x):
        """Forward pass of the encoder

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        out1 = out
        for i in range(self.layers):
            if i==3:
                out1 = out
                out1 = self.cov2d_1(out1) # skip connection to last layer
                out1 = torch.flatten(out1, start_dim=2)
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=2)
        out = torch.cat((out, out1), dim=2)
        out = self.dense(out)
        selection = self.relu_1(out)
        return selection


class PV_AE(nn.Module):
    """AE block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, encoder, fitter_function, 
                 limits=[1, 10, 10, 10, 10, 0.5]
                 ):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(PV_AE, self).__init__()
        
        self.encoder = encoder
        self.fitter_function = fitter_function
        self.limits = limits
        self.device = next(self.parameters()).device
        
    def forward(self,tiles):
        ''' tiles shape (batch*spots,ch,x,y)'''
        embedding = self.encoder(tiles)
        fits = self.fitter_function(embedding, tiles.shape[-2:], limits=self.limits, device=self.device) # (batch*spots,fits,x,y)
        
        if self.training: return embedding, fits.sum(axis=1)
        else: return embedding, fits


class PV_Fitter_2D():
        
    def __init__(self, function, 
                 encoder, encoder_init,
                 autoencoder, autoencoder_init, 
                 dset, input_channels, num_params, num_fits, 
                 limits=[1, 10, 10, 10, 10, 0.5], scaler=None, 
                 post_processing=None, device="cuda", 
                 loops_scaler=None, flatten_from=1, 
                 learning_rate=3e-5,
                 emb_h5 = './embeddings_2D.h5',
                 gen_h5= './generated_2D.h5',
                 folder='./save_folder',
                 wandb_project = None):
        """_summary_

        Args:
            function (_type_): _description_
            x_data (_type_): _description_
            input_channels (_type_): number of channels in orig data. ex. 2 channels: high loss, low loss
            num_params (_type_): number of parameters needed to generate the fit
            num_fits (_type_): the number of peaks to include
            limits (list): values of [A_g, x, sigma, A_l, gamma, nu]. Defaults to: [1,975,25,1,25,1]
            scaler (_type_, optional): _description_. Defaults to None.
            post_processing (_type_, optional): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to "cuda".
            loops_scaler (_type_, optional): _description_. Defaults to None.
            flatten_from (int, optional): _description_. Defaults to 1.
        """        
        self.input_channels = input_channels
        self.scaler = scaler
        self.function = function
        self.dset = dset
        self.post_processing = post_processing
        self.device = device
        self.num_params = num_params
        self.num_fits = num_fits
        self.limits = limits
        self.loops_scaler = loops_scaler
        self.flat_dim = flatten_from
        self.learning_rate = learning_rate

        self._checkpoint = None
        self._folder = folder
        self.wandb_project = wandb_project
                
        self.emb_h5 = emb_h5
        self.gen_h5 = emb_h5
        
        self.train = False

        self.encoder = encoder(**self.encoder_init).to(self.device)        
        self.Fitter = autoencoder(function=self.function,**autoencoder_init).to(self.device)
        # self.Fitter = self.Fitter.to(self.device)
        # sets the datatype of the model to float32
        self.Fitter.type(torch.float32)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.Fitter.parameters(), lr=self.learning_rate
        )
        
    @property 
    def file(self): return self._file
        
    @property
    def folder(self): return self._folder
    @folder.setter
    def folder(self,value): self._folder = value
    
    @property
    def check(self): return self._check
    
    @property
    def checkpoint(self): return self._checkpoint
    
    @checkpoint.setter
    def checkpoint(self, value):
        self._checkpoint = value
        try:
            folder,file = os.path.split(self._checkpoint)
            self._file = file
            self._check = file.split('.pkl')[0]
            self._folder = folder
        except:
            self.check = None
            self.folder = None
            self.file = None
            
    def open_embedding_h(self):
        h = h5py.File(f'{self.folder}/{self.emb_h5}','r+')
        try:
            self.embedding = h[f'embedding_{self.check}']
        except:
            pass
        
        return h
    
    def open_generated_h(self):
        h = h5py.File(f'{self.folder}/{self.gen_h5}','r+')
        try: self.generated = h[self.check]
        except: pass
        return h
    
    def compile_model(self):
        """function that complies the neural network model
        """
        pass

    ## TODO: implement embedding calculation/unshuffler
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0, 
              coef_2=0,
              coef_3=0,
              coef_4=0,
              coef_5=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=2,
              epoch_=None,
              batch_size=32,
              best_train_loss=None,
              save_emb_every=None,
              minibatch_logging_rate=None,
              wandb_init={}):
        """Function that trains the model
        
            Args:
                data (torch.tensor): Data to train the model.
                max_learning_rate (float, optional): Sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
                coef_1 (float, optional): Hyperparameter for ln loss. Defaults to 0.
                coef_2 (float, optional): Hyperparameter for contrastive loss. Defaults to 0.
                coef_3 (float, optional): Hyperparameter for divergency loss. Defaults to 0.
                coef_4 (float, optional): Hyperparameter for an additional loss term. Defaults to 0.
                seed (int, optional): Sets the random seed. Defaults to 12.
                epochs (int, optional): Number of epochs to train. Defaults to 100.
                with_scheduler (bool, optional): Sets if you should use the learning rate cycler. Defaults to True.
                ln_parm (int, optional): Order of the Ln regularization. Defaults to 2.
                epoch_ (int, optional): Current epoch for continuing training. Defaults to None.
                batch_size (int, optional): Sets the batch size for training. Defaults to 32.
                best_train_loss (float, optional): Current loss value to determine if you should save the value. Defaults to None.
                save_emb_every (int, optional): Frequency (in epochs) to save embeddings. Defaults to None.
                minibatch_logging_rate (int, optional): Frequency (in minibatches) to log training progress. Defaults to None.
                wandb_init (dict, optional): Initialization parameters for Weights & Biases logging. Defaults to {}. 
"""
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(self.folder)

        # set seed
        torch.manual_seed(seed)

        # builds the dataloader
        self.DataLoader_ = DataLoader(self.dset, batch_size=batch_size, shuffle=True)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None: best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None: self.start_epoch = 0
        else: self.start_epoch = epoch_

        if self.wandb_project is not None:  
            wandb_init['project'] = self.wandb_project
            wandb.init(**wandb_init) # figure out config later
            
        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            if save_emb_every is not None and epoch % save_emb_every == 0: # tell loss function to give embedding
                print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
                print('.............................')
                fill_embeddings = self.get_embedding(data, train=True)

            loss_dict = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, coef_4, ln_parm,
                fill_embeddings=fill_embeddings, minibatch_logging_rate=minibatch_logging_rate)
            # divide by batches inplace
            loss_dict.update( (k,v/len(self.DataLoader_)) for k,v in loss_dict.items())
            
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {loss_dict["train_loss"]:.4f}')
            print('.............................')

          #  schedular.step()
            lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
            self.checkpoint = self.folder + f'/{save_date}_' +\
                f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                f'_trainloss:{loss_dict["train_loss"]:.4f}.pkl'
            self.save_checkpoint(epoch,
                                loss_dict=loss_dict,
                                coef_1=coef_1, 
                                coef_2=coef_2,
                                coef_3=coef_3,
                                coef_4=coef_4,
                                coef_5=coef_5,
                                ln_parm=ln_parm)

            if save_emb_every is not None and epoch % save_emb_every == 0: # tell loss function to give embedding
                h = self.embedding.file
                check = self.checkpoint.split('/')[-1][:-4]
                h[f'embedding_{check}'] = h[f'embedding_temp']
                h[f'scaleshear_{check}'] = h[f'scaleshear_temp']
                h[f'rotation_{check}'] = h[f'rotation_temp'] 
                h[f'translation_{check}'] = h[f'translation_temp']
                self.embedding = h[f'embedding_{check}']
                self.scale_shear = h[f'scaleshear_{check}']           
                self.rotation = h[f'rotation_{check}']         
                self.translation = h[f'translation_{check}']
                del h[f'embedding_temp']         
                del h[f'scaleshear_temp']          
                del h[f'rotation_temp']          
                del h[f'translation_temp']
                        
        if scheduler is not None:
            scheduler.step()

    def save_checkpoint(self,epoch,loss_dict,**kwargs):
        checkpoint = {
            "Fitter": self.Fitter.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch,
            'loss_dict': loss_dict,
            'loss_params': kwargs,
        }
        torch.save(checkpoint, self.checkpoint)

    def loss_function(self,
                      train_iterator,
                      coef1=0,
                      coef2=0,
                      coef3=0,
                      coef4=0,
                      coef5=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False,
                      minibatch_logging_rate=None,
                      mask_params={'min_threshold':3e-5,'coef':0.01,'channels':1,'ln_parm':2,'at_least':0}):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef1 (float, optional): Ln hyperparameter. Defaults to 0.
            coef2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef3 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.Fitter.train()

        # loss of the epoch
        loss_dict = {'ln_loss': 0,
                    #  'contras_loss': 0,
                    #  'maxi_loss': 0,
                    #  'masking_loss': 0,
                     'train_loss': 0,
                    #  'sparse_max_loss': 0,
                    #  'l2_batchwise_loss': 0,
                     }
        # weighted_ln_ = Weighted_LN_loss(coef=coef1,channels=self.num_fits).to(self.device)
        ln_ = LN_loss(coef1, ln_parm).to(self.device)
        # con_l = ContrastiveLoss(coef2).to(self.device)
        # maxi_ = DivergenceLoss(train_iterator.batch_size, coef3).to(self.device)
        # sparse_max = Sparse_Max_Loss(min_threshold=self.learning_rate,
        #                                 channels=self.num_fits, 
        #                                 coef=coef4).to(self.device)
        
        for i,(idx,x) in enumerate(tqdm(train_iterator, leave=True, total=len(train_iterator))):
            # tic = time.time()

            x = x.to(self.device, dtype=torch.float)

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x = self.Fitter(x)
            else: embedding, sd, mn, predicted_x = self.Fitter(x)

            if coef1 > 0: 
                reg_loss_1 = ln_(embedding[:,:,0])
                loss_dict['weighted_ln_loss']+=reg_loss_1
            else: reg_loss_1 = 0

            # if coef2 > 0: 
            #     contras_loss = con_l(embedding[:,:,0])
            #     loss_dict['contras_loss']+=contras_loss
            # else: contras_loss = 0
                
            # if coef3 > 0: 
            #     maxi_loss = maxi_(embedding[:,:,0])
            #     loss_dict['maxi_loss']+=maxi_loss
            # else: maxi_loss = 0
            
            # if coef4 > 0: # sparse_max_loss
            #     sparse_max_loss = sparse_max(embedding[:,:,0])
            #     loss_dict['sparse_max_loss']+=sparse_max_loss
            # else: sparse_max_loss = 0
            
            # if coef5 > 0: # set so the variation in x < fwhm, but the smaller the better.
            #     l2_loss = coef5*( (embedding[:,:,1]/embedding[:,:,2]).max(dim=0).values - \
            #                       (embedding[:,:,1]/embedding[:,:,2]).min(dim=0).values ).mean()
            #     loss_dict['l2_batchwise_loss'] += l2_loss
            # else: l2_loss = 0
            
            # reconstruction loss TODO: try getting rid of this
            masking_loss = masking_threshold_loss(min_threshold=3e-5,coef=0.01,channels=1,ln_parm=2,at_least=at_least)
            loss,mask,at_least = masking_loss(x, predicted_x)

            loss_dict['masking_loss'] += loss.item()
            
            loss = loss + reg_loss_1 #+ contras_loss - maxi_loss + l2_loss
            loss_dict['train_loss'] += loss.item()

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()
            
            # fill embedding if the correct epoch
            if fill_embeddings:
                sorted, indices = torch.sort(idx)
                sorted = sorted.detach().numpy()
                self.embedding[sorted] = embedding[indices].cpu().detach().numpy()
                # print('\tt', abs(tic-toc)) # 2.7684452533721924
            if minibatch_logging_rate is not None: 
                if i%minibatch_logging_rate==0: wandb.log({k: v/(i+1) for k,v in loss_dict.items()})

        return loss_dict

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.Fitter.load_state_dict(checkpoint['Fitter'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        try:
            with self.open_embedding_h() as h:
                print('Generated available')
        except Exception as error:
            print(error)
            print('Embeddings not opened')

        try:
            with self.open_generated_h() as h:
                print('Generated available')
        except Exception as error:
            print(error)
            print('Generated not opened')

        
    def load_loss_data(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        loss_dict = checkpoint['loss_dict']
        loss_params = checkpoint['loss_params']
        return start_epoch,loss_dict, loss_params
    
    def get_embedding(self, data, batch_size=32,train=True,check=None):
        """extracts embeddings from the data

        Args:
            data (torch.tensor): data to get embeddings from
            batch_size (int, optional): batchsize for inference. Defaults to 32.

        Returns:
            torch.tensor: predicted embeddings
        """

        # builds the dataloader
        dataloader = DataLoader(data, batch_size, shuffle=False)
        s = data.shape[1]
        try:
            try: h = h5py.File(f'{self.folder}/{self.emb_h5}','w')
            except: h = h5py.File(f'{self.folder}/{self.emb_h5}','r+')

            try: check = self.checkpoint.split('/')[-1][:-4]
            except: check=check
            
            # make embedding dataset
            try:
                embedding_ = h.create_dataset(f'embedding_{check}', 
                                            #   data = np.zeros([s[0], s[1], self.num_fits, self.num_params]),
                                              shape=(s[0], s[1], self.num_fits, self.num_params),
                                              dtype='float32')  
            except: 
                embedding_ = h[f'embedding_{check}']
                
            # make fitted dataset
            try:
                fits_ = h.create_dataset(f'fits_{check}', 
                                        #  data = np.zeros([s[0],self.num_fits,s[1],s[2]]),
                                         shape = (s[0],s[1],self.num_fits,s[2]),
                                         dtype='float32')  
            except:
                fits_ = h[f'fits_{check}']

            self.embedding = embedding_
            self.fits = fits_

        except Exception as error:
            print(error) 
            assert train,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
                
        if train: 
            print('Created empty h5 embedding datasets to fill during training')
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            s=embedding_.shape
            for i, (_,x) in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                with torch.no_grad():
                    value = x
                    batch_size = x.shape[0]
                    test_value = Variable(value.to(self.device))
                    test_value = test_value.float()
                    embedding,fit = self.Fitter(test_value,return_sum=False)
                    
                    self.embedding[i*batch_size:(i+1)*batch_size] = embedding.reshape(batch_size,s[1],s[2],s[3]).cpu().detach().numpy()
                    self.fits[i*batch_size:(i+1)*batch_size] = fit.reshape(batch_size,s[1],self.num_fits,-1).cpu().detach().numpy()
                   
        h.close()
        
    def get_clusters(self,dset,scaled_array,n_components=None,n_clusters=None):
        if n_components == None:
            print('Getting scree plot...')
            pca = PCA()
            pca.fit(scaled_array)
            plt.clf()
            plt.plot(pca.explained_variance_ratio_.cumsum(),marker='o')
            plt.show()
            n_components = int(input("Choose number of PCA components: "))
            
        print(f'PCA with {n_components} components...')
        pca = PCA(n_components)
        transformed = pca.fit_transform(scaled_array)
        
        if n_clusters == None:
            print('Getting elbow plot...')
            wcss = []
            for i in tqdm(range(10,self.stacked_embedding_size+7)):
                kmeans_pca = KMeans(n_clusters=i,init='k-means++',random_state=42)
                kmeans_pca.fit(transformed[::100])
                wcss.append(kmeans_pca.inertia_)
            plt.clf()
            plt.plot(range(10,self.stacked_embedding_size+7),wcss,marker='o')
            plt.show()
            n_clusters = int(input('Choose number of clusters: '))
            
        print(f'Clustering with {n_clusters} clusters...')
        kmeans_pca = KMeans(n_clusters=n_clusters,init='k-means++',random_state=42)
        kmeans_pca.fit(transformed)
        cluster_list = []
        for i,particle in enumerate(dset.meta['particle_list']):
            img = kmeans_pca.labels_[dset.meta['particle_inds'][i]:\
                dset.meta['particle_inds'][i+1] ].reshape(dset.meta['shape_list'][i][0][0],
                                                            dset.meta['shape_list'][i][0][1])
            cluster_list.append(img)
        self.cluster_list = cluster_list
        self.cluster_labels = kmeans_pca.labels_
        print('Done')
        return cluster_list,kmeans_pca.labels_

    def generate_range(self,dset,checkpoint,
                         ranges=None,
                         generator_iters=50,
                         averaging_number=100,
                         overwrite=False,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space.
        Saves to embedding h5 dataset

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # assert not self.train, 'set self.train to False if calculating manually'
        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            channels = kwargs["channels"]
        else:
            channels = np.arange(self.embedding_size)
        
        if "ranges" in kwargs:
            ranges = kwargs["ranges"]

        # gets the embedding if a specific embedding is not provided
        try:
            embedding = self.embedding
        except Exception as error:
            print(error)
            assert False, 'Make sure model is set to appropriate embeddings first'

        try: # try opening h5 file
            try: # make new file
                h = h5py.File(self.gen_h5_path,'w')
            except: # open existing file
                h = h5py.File(self.gen_h5_path,'r+')

            check = checkpoint.split('/')[-1][:-4]
            try: # make new dataset
                if overwrite and check in h: del h[check]
                self.generated = h.create_dataset(check,
                                            shape=( len(dset.meta['particle_list']),
                                                            generator_iters,
                                                            len(channels), 
                                                            dset.eels_chs,
                                                            dset.spec_len),
                                            dtype='float32') 
            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        for p,p_name in enumerate(dset.meta['particle_list']): # each sample
            print(p, p_name)
            with self.open_embedding_h() as he:
                data=self.embedding[dset.meta['particle_inds'][p]:\
                                    dset.meta['particle_inds'][p+1]].astype('float32')
                
            # loops around the number of iterations to generate
            for i in tqdm(range(generator_iters)):
                
                # loops around all of the embeddings
                for j, channel in enumerate(channels):

                    if ranges is None: # span this range when generating
                        ranges = np.stack((np.min(data, axis=0),
                                        np.max(data, axis=0)), axis=1)

                    # linear space values for the embeddings
                    value = np.linspace(ranges[j][0], ranges[j][1],
                                        generator_iters)

                    # finds the nearest points to the value and then takes the average
                    # average number of points based on the averaging number
                    idx = find_nearest(
                        data[:,channel],
                        value[i],
                        averaging_number)

                    # computes the mean of the selected index to yield 2D image
                    gen_value = np.mean(data[idx], axis=0)

                    # specifically updates the value of the mean embedding image to visualize 
                    # based on the linear spaced vector
                    gen_value[channel] = value[i]

                    generated = self.generate_spectra(gen_value).squeeze()       
                    # generates diffraction pattern
                    self.generated[dset.meta['particle_inds'][p]: \
                                   dset.meta['particle_inds'][p+1],i,j] = generated
        h.close()

        # return self.generated
              
    def generate_spectra(self, embedding):
        """generates spectra from embeddings

        Args:
            embedding (torch.tensor): predicted embeddings to decode

        Returns:
            torch.tensor: decoded spectra
        """

        embedding = torch.from_numpy(np.atleast_2d(embedding).astype('float32')).to(self.device)
        predicted_1D = self.decoder(embedding)
        predicted_1D = predicted_1D.cpu().detach().numpy()
        
        return predicted_1D
    
    #TODO: make a save checkpoint function
