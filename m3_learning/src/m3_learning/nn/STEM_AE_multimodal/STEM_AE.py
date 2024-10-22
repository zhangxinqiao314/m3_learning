import sys
import os
from os.path import join as pjoin

from more_itertools import collapse, collate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.autograd import Variable


from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss, Sparse_Max_Loss, Weighted_LN_loss
from m3_learning.nn.random import random_seed
from m3_learning.optimizers.AdaHessian import AdaHessian
from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import find_nearest
from m3_learning.viz.nn import get_theta

import warnings 
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
import h5py
import time
from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import wandb

# wandb api key: 727b04fe540133105c81a885cc6014d625240436

## TODO: add 1d version of all classes, with Attention
## TODO: add make sure we can easily access rotations and run encoder/decoder separately
## TODO: make sure embedding unshuffler is good with 1D and 2D
## TODO: want to try overfit-underfit model

# sys.path.append(os.path.abspath("./../STEM_AE")) # or whatever the name of the immediate parent folder is

class ConvAutoencoder_2D():
    """builds the convolutional autoencoder
    """

    def __init__(self,
                 encoder_step_size,
                 pooling_list,
                 decoder_step_size,
                 upsampling_list,
                 embedding_size,
                 conv_size,
                 device,
                 checkpoint = None,
                 learning_rate=3e-5,
                 emb_h5_path = './Combined_all_samples_2D/embeddings_2D.h5',
                 gen_h5_path = './Combined_all_samples_2D/generated_2D.h5',
                 ):
        """Initialization function

        Args:
            encoder_step_size (list): sets the size of the encoder
            pooling_list (list): sets the pooling list to define the pooling layers
            decoder_step_size (list): sets the size of the decoder
            upsampling_list (list): sets the size for upsampling
            embedding_size (int): number of embedding channels
            conv_size (int): sets the number of convolutional neurons in the model
            device (torch.device): set the device to run the model
            learning_rate (float, optional): sets the learning rate for the optimizer. Defaults to 3e-5.
        """
        self.encoder_step_size = encoder_step_size
        self.pooling_list = pooling_list
        self.decoder_step_size = decoder_step_size
        self.upsampling_list = upsampling_list
        self.embedding_size = embedding_size
        self.conv_size = conv_size
        self.device = device
        self.learning_rate = learning_rate

        self.checkpoint = checkpoint
        # self.train = train

        self.emb_h5_path = emb_h5_path
        self.gen_h5_path = gen_h5_path

        # complies the network
        self.compile_model()

    def open_embedding_h(self):
        h = h5py.File(self.emb_h5_path,'r+')
        try: 
            check = self.checkpoint.split('/')[-1][:-4]
            self.embedding = h[f'embedding_{check}']
        except: pass
        return h
    
    def open_generated_h(self):
        h = h5py.File(self.gen_h5_path)
        check = self.checkpoint.split('/')[-1][:-4]
        try: self.generated = h[check]
        except: pass
        return h
     
    def compile_model(self):
        """function that complies the neural network model
        """
        # builds the encoder
        self.encoder = Encoder_2D(
            original_step_size=self.encoder_step_size,
            pooling_list=self.pooling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            device=self.device
        ).to(self.device)

        # builds the decoder
        self.decoder = Decoder_2D(
            original_step_size=self.decoder_step_size,
            upsampling_list=self.upsampling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
        ).to(self.device)

        # builds the autoencoder
        self.autoencoder = AutoEncoder_2D(
            self.encoder, self.decoder, self.embedding_size,
            device=self.device).to(self.device)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.learning_rate
        )

        # sets the datatype of the model to float32
        self.autoencoder.type(torch.float32)

    ## TODO: implement embedding calculation/unshuffler
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              save_emb_every=10):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
            save_emb_every (int, optional): 
        """
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)

        # builds the dataloader
        self.DataLoader_ = DataLoader(
            data, batch_size=batch_size, shuffle=True)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_

        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            if epoch % save_emb_every ==0: # tell loss function to give embedding
                print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
                print('.............................')
                fill_embeddings = self.get_embedding(data, train=True)


            train = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, ln_parm,
                fill_embeddings=fill_embeddings)
            train_loss = train
            train_loss /= len(self.DataLoader_)
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')

          #  schedular.step()
            # if best_train_loss > train_loss:
            best_train_loss = train_loss
            checkpoint = {
                "net": self.autoencoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
            }
            if epoch >= 0:
                lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
                file_path = folder_path + f'/{save_date}_' +\
                    f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                    f'_trainloss:{train_loss:.4f}.pkl'
                torch.save(checkpoint, file_path)
                check = file_path.split('/')[-1][:-4]
                self.checkpoint = check

            if epoch%save_emb_every==0:
                with self.open_embedding_h() as h:
                    h[f'embedding_{check}'] = h[f'embedding_']
                    h[f'scaleshear_{check}'] = h[f'embedding_']
                    h[f'rotation_{check}'] = h[f'embedding_'] 
                    h[f'translation_{check}'] = h[f'embedding_']
                    self.embedding = h[f'embedding_{check}']
                    self.scale_shear = h[f'scaleshear_{check}']           
                    self.rotation = h[f'rotation_{check}']         
                    self.translation = h[f'translation_{check}']
                    # del h[f'embedding_']         
                    # del h[f'embedding_']          
                    # del h[f'embedding_']          
                    # del h[f'embedding_']
                        
        if scheduler is not None:
            scheduler.step()

    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)
        
        for idx,x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
            # tic = time.time()
            sorted, indices = torch.sort(idx)
            sorted = sorted.detach().numpy()

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x, scaler_shear,rotation,translation = self.autoencoder(x)
            else: embedding, sd, mn, predicted_x = self.autoencoder(x)

            reg_loss_1 = coef*torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5

            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)

            # reconstruction loss
            mask = (predicted_x!=0)
            loss = F.mse_loss(x, predicted_x, reduction='mean');
            loss = (loss*mask.float()).sum()
            loss /= mask.sum()

            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            train_loss += loss.item()

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()
            
            # fill embedding if the correct epoch
            if fill_embeddings:
                # scaler_shear,rotation,translation = self.autoencoder.temp_affines
                self.embedding[sorted] = embedding[indices].cpu().detach().numpy()
                self.scale_shear[sorted] = scaler_shear[indices].cpu().reshape((-1,6)).detach().numpy()
                self.rotation[sorted] = rotation[indices].reshape((-1,6)).cpu().detach().numpy()
                self.translation[sorted] = translation[indices].reshape((-1,6)).cpu().detach().numpy()
                # print('\tt', abs(tic-toc)) # 2.7684452533721924

        return train_loss

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['net'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        check = path_checkpoint.split('/')[-1][:-4]

        try:
            h = h5py.File(self.emb_h5_path,'r+')
            self.embedding = h[f'embedding_{check}']
            self.scale_shear = h[f'scaleshear_{check}']
            self.rotation = h[f'rotation_{check}']                    
            self.translation = h[f'translation_{check}']
        except Exception as error:
            print(error)
            print('Embedding and affines not opened')

        try:
            h = h5py.File(self.gen_h5_path,'r+')
            self.generated = h[check]
        except Exception as error:
            print(error)
            print('Generated not opened')

    def get_embedding(self, data, batch_size=32,train=True, check = ''):
        """extracts embeddings from the data

        Args:
            data (torch.tensor): data to get embeddings from
            batch_size (int, optional): batchsize for inference. Defaults to 32.

        Returns:
            torch.tensor: predicted embeddings
        """

        # builds the dataloader
        dataloader = DataLoader(data, 
            batch_size, shuffle=False)

        try:
            try: 
                h = h5py.File(self.emb_h5_path,'w')
            except: 
                h = h5py.File(self.emb_h5_path,'r+')

            try: check = self.checkpoint.split('/')[-1][:-4]
            except: check = check
            try:
                embedding_ = h.create_dataset(f'embedding_{check}', data = np.zeros([data.shape[0][0], self.embedding_size]))
                scale_shear_ = h.create_dataset(f'scaleshear_{check}', data = np.zeros([data.shape[0][0],6]))
                rotation_ = h.create_dataset(f'rotation_{check}', data = np.zeros([data.shape[0][0],6]))
                translation_ = h.create_dataset(f'translation_{check}', data = np.zeros([data.shape[0][0],6]))
            except: 
                embedding_ = h[f'embedding_{check}']
                scale_shear_ = h[f'scaleshear_{check}']
                rotation_ = h[f'rotation_{check}']                    
                translation_ = h[f'translation_{check}']

            self.embedding = embedding_
            self.scale_shear = scale_shear_
            self.rotation = rotation_
            self.translation = translation_

        except Exception as error:
            print(error) 
            # assert self.train,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
                
        if train: 
            print('Created empty h5 embedding datasets to fill during training')
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            for i, (_,x) in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                with torch.no_grad():
                    value = x
                    test_value = Variable(value.to(self.device))
                    test_value = test_value.float()
                    embedding,scale_shear,rotation,translation = self.encoder(test_value)
                    
                    self.embedding[i*batch_size:(i+1)*batch_size, :] = embedding.cpu().detach().numpy()
                    self.scale_shear[i*batch_size:(i+1)*batch_size, :] = scale_shear.reshape(-1,6).cpu().detach().numpy()
                    self.rotation[i*batch_size:(i+1)*batch_size, :] = rotation.reshape(-1,6).cpu().detach().numpy()
                    self.translation[i*batch_size:(i+1)*batch_size, :] = translation.reshape(-1,6).cpu().detach().numpy()
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
                         with_affine=False,
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

        channels = np.arange(self.embedding_size)
        # sets the channels to use in the object
        if "channels" in kwargs:
            channels = kwargs["channels"]
        if "ranges" in kwargs:
            ranges = kwargs["ranges"]

        # # gets the embedding if a specific embedding is not provided
        # try:
        #     embedding = self.embedding
        # except Exception as error:
        #     print(error)
        #     assert False, 'Make sure model is set to appropriate embeddings first'
        s = dset.shape[0]    
        try: # try opening h5 file
            try: # make new file
                h = h5py.File(self.gen_h5_path,'a')
            except: # open existing file
                h = h5py.File(self.gen_h5_path,'r+')

            check = checkpoint.split('/')[-1][:-4]
            try: # make new dataset
                if overwrite and check in h: del h[check]
                self.generated = h.create_dataset(check,
                                            data=np.zeros( [len(dset.meta['particle_list']),
                                                            generator_iters,
                                                            len(channels),
                                                            s[2],s[3]] ) )
            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        for p,p_name in enumerate(dset.meta['particle_list']): # each sample
            print(p, p_name)
            with self.open_embedding_h() as he:
                data=self.embedding[dset.meta['particle_inds'][p]:\
                                    dset.meta['particle_inds'][p+1]]
                
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

                    # generates diffraction pattern
                    self.generated[dset.meta['particle_inds'][p]: dset.meta['particle_inds'][p+1],i,j] =\
                        self.generate_spectra(gen_value).squeeze()       
        h.close()

        # return self.generated
                    
    def generate_spectra(self, embedding,with_affine=False):
        """generates spectra from embeddings

        Args:
            embedding (torch.tensor): predicted embeddings to decode

        Returns:
            torch.tensor: decoded spectra
        """

        embedding = torch.from_numpy(np.atleast_2d(embedding)).to(self.device)
        embedding = self.decoder(embedding.float())
        if with_affine:
            embedding = self.transformer(embedding)
        embedding = embedding.cpu().detach().numpy()
        return embedding


class ConvAutoencoder_1D():
    """builds the convolutional autoencoder
    """

    def __init__(self,
                 encoder_step_size,
                 pooling_list,
                 decoder_step_size,
                 upsampling_list,
                 embedding_size,
                 channels,
                 conv_size,
                 device,
                 checkpoint = None,
                 learning_rate=3e-5,
                 emb_h5_path = './Combined_all_samples_1D/embeddings_1D.h5',
                 gen_h5_path = './Combined_all_samples_1D/generated_1D.h5',
                 ):
        """Initialization function

        Args:
            encoder_step_size (list): sets the size of the encoder
            pooling_list (list): sets the pooling list to define the pooling layers
            decoder_step_size (list): sets the size of the decoder
            upsampling_list (list): sets the size for upsampling
            embedding_size (int): number of embedding channels
            conv_size (int): sets the number of convolutional neurons in the model
            device (torch.device): set the device to run the model
            learning_rate (float, optional): sets the learning rate for the optimizer. Defaults to 3e-5.
        """
        self.encoder_step_size = encoder_step_size
        self.pooling_list = pooling_list
        self.decoder_step_size = decoder_step_size
        self.upsampling_list = upsampling_list
        self.embedding_size = embedding_size
        self.conv_size = conv_size
        self.device = device
        self.learning_rate = learning_rate
        self.channels = channels

        self.checkpoint = checkpoint
        # self.train = train

        self.emb_h5_path = emb_h5_path
        self.gen_h5_path = gen_h5_path

        # complies the network
        self.compile_model()

    def open_embedding_h(self):
        check = self.checkpoint.split('/')[-1][:-4]
        h = h5py.File(self.emb_h5_path,'r+')
        try:
            self.embedding = h[f'embedding_{check}']
        except:
            pass
        
        return h
    
    def open_generated_h(self):
        h = h5py.File(self.gen_h5_path)
        check = self.checkpoint.split('/')[-1][:-4]
        try: self.generated = h[check]
        except: pass
        return h
    
    def compile_model(self):
        """function that complies the neural network model
        """
        # builds the encoder
        self.encoder = Encoder_1D(
            original_step_size=self.encoder_step_size,
            pooling_list=self.pooling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            in_channels=self.channels,
            device=self.device,
            attn_heads=3
        ).to(self.device)

        # builds the decoder
        self.decoder = Decoder_1D(
            original_step_size=self.decoder_step_size,
            upsampling_list=self.upsampling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            out_channels=self.channels,
            attn_heads=3
        ).to(self.device)

        # builds the autoencoder
        self.autoencoder = AutoEncoder_1D(
            self.encoder, self.decoder, device=self.device).to(self.device)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.learning_rate
        )

        # sets the datatype of the model to float32
        self.autoencoder.type(torch.float32)

    ## TODO: implement embedding calculation/unshuffler
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              save_emb_every=10):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
            save_emb_every (int, optional): 
        """
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)

        # builds the dataloader
        self.DataLoader_ = DataLoader(
            data, batch_size=batch_size, shuffle=True)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_

        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            if epoch % save_emb_every == 0: # tell loss function to give embedding
                print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
                print('.............................')
                fill_embeddings = self.get_embedding(data, train=True)


            train = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, ln_parm,
                fill_embeddings=fill_embeddings)
            train_loss = train
            train_loss /= len(self.DataLoader_)
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')

          #  schedular.step()
            if best_train_loss > train_loss:
                best_train_loss = train_loss
                checkpoint = {
                    "net": self.autoencoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch,
                    "encoder": self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                if epoch >= 0:
                    lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
                    file_path = folder_path + f'/{save_date}_' +\
                        f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                        f'_trainloss:{train_loss:.4f}.pkl'
                    torch.save(checkpoint, file_path)

            if epoch%save_emb_every==0:
                h = self.embedding.file
                check = file_path.split('/')[-1][:-4]
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

    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)
        
        for idx,x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
            # tic = time.time()

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x = self.autoencoder(x)
            else: embedding, sd, mn, predicted_x = self.autoencoder(x)

            reg_loss_1 = coef*torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5

            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)

            # reconstruction loss
            mask = (predicted_x!=0)
            loss = F.mse_loss(x, predicted_x, reduction='mean');
            loss = (loss*mask.float()).sum()
            loss /= mask.sum()

            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            train_loss += loss.item()

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

        return train_loss

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['net'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        check = path_checkpoint.split('/')[-1][:-4]

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

    def get_embedding(self, data, batch_size=32,train=True,check=''):
        """extracts embeddings from the data

        Args:
            data (torch.tensor): data to get embeddings from
            batch_size (int, optional): batchsize for inference. Defaults to 32.

        Returns:
            torch.tensor: predicted embeddings
        """

        # builds the dataloader
        dataloader = DataLoader(data, batch_size, shuffle=False)

        try:
            try: h = h5py.File(self.emb_h5_path,'w')
            except: h = h5py.File(self.emb_h5_path,'r+')

            try: check = self.checkpoint.split('/')[-1][:-4]
            except: check=check
            
            try:
                embedding_ = h.create_dataset(f'embedding_{check}', 
                                              data = np.zeros([data.shape[1][0], self.embedding_size]),
                                              dtype='float32')
            except: 
                embedding_ = h[f'embedding_{check}']

            self.embedding = embedding_

        except Exception as error:
            print(error) 
            assert train,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
                
        if train: 
            print('Created empty h5 embedding datasets to fill during training')
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            for i, (_,x) in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                with torch.no_grad():
                    value = x
                    test_value = Variable(value.to(self.device))
                    test_value = test_value.float()
                    embedding = self.encoder(test_value)
                    
                    self.embedding[i*batch_size:(i+1)*batch_size, :] = embedding.cpu().detach().numpy()
                   
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


class FitterAutoencoder_1D():
        
    def __init__(self,function, dset, input_channels, num_params, num_fits, limits=[1,975,25,1,25,1], scaler=None, 
                 post_processing=None, device="cuda", 
                 loops_scaler=None,
                 x1_ch_list=[8,6,4], x1_pool=64, 
                 x2_pool_list=[16,8,4], x2_ch_list=[8,16],
                 dense_list=[24,16,8],
                 learning_rate=3e-5,
                 emb_h5 = './embeddings_1D.h5',
                 gen_h5= './generated_1D.h5',
                 folder='./save_folder',
                 wandb_project = None,
                 dataloader_sampler=None,
                 custom_collate_fn=None,
                 sampler_kwargs={}):
        """_summary_

        Args:
            function (type): fitter function used to fit embedding parameters.
            dset (type): The dataset.
            input_channels (type): The number of channels in the original data.
            num_params (type): The number of parameters needed to generate the fit.
            num_fits (type): The number of peaks to include.
            limits (list, optional): The limits on the fit paramters used during regularization. Defaults to [1,975,25,1,25,1].
            scaler (type, optional): The scaler. Defaults to None.
            post_processing (type, optional): The post-processing. Defaults to None.
            device (str, optional): The device. Defaults to "cuda".
            loops_scaler (type, optional): The loops scaler. Defaults to None.
            flatten_from (int, optional): The flatten from value. Defaults to 1.
            x1_ch_list (list, optional): The x1 channel list. Defaults to [8,6,4].
            x1_pool (int, optional): The x1 pool value. Defaults to 64.
            x2_pool_list (list, optional): The x2 pool list. Defaults to [16,8,4].
            x2_ch_list (list, optional): The x2 channel list. Defaults to [8,16].
            dense_list (list, optional): The dense list. Defaults to [24,16,8].
            learning_rate (float, optional): The learning rate. Defaults to 3e-5.
            emb_h5 (str, optional): The path to the h5 file where the embedding and fits is saved. Defaults to './embeddings_1D.h5'.
            folder (str, optional): The folder. Defaults to './save_folder'.
            wandb_project (type, optional): The name of the wandb project to log training results. Defaults to None, and no training is logged. You must create a mandb account and sign in on jupyter notebook to use this feature.
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
        self.learning_rate = learning_rate
        try: self.dataloader_sampler = dataloader_sampler(**sampler_kwargs)
        except: pass
        self.custom_collate=custom_collate_fn

        self._checkpoint = None
        self._folder = folder
        self.wandb_project = wandb_project
                
        self.emb_h5 = emb_h5
        self.gen_h5 = emb_h5
        
        self.train = False

        # complies the network
        self.compile_model()
        
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
        self.Fitter = Multiscale1DFitter(function=self.function,
                                 x_data = self.dset,
                                 input_channels=self.dset.eels_chs,
                                 num_params=self.num_params,
                                 num_fits=self.num_fits,
                                 limits=self.limits,
                                 device='cuda:0',
                                 flatten_from = 1,
                            )
        self.Fitter = self.Fitter.to(self.device)
        # sets the datatype of the model to float32
        self.Fitter.type(torch.float32)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.Fitter.parameters(), lr=self.learning_rate
        )

    ## TODO: implement embedding calculation/unshuffler
    def Train(self,
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
              binning=False,
              weight_by_distance=False,
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
        self.DataLoader_ = DataLoader(self.dset, sampler=self.dataloader_sampler, collate_fn=self.custom_collate)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_

        if self.wandb_project is not None:  
            wandb_init['project'] = self.wandb_project
            wandb.init(**wandb_init) # figure out config later
            
        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            # if save_emb_every is not None and epoch % save_emb_every == 0: # tell loss function to give embedding
            #     print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
            #     print('.............................')
            #     fill_embeddings = self.get_embedding(data, train=True)


            loss_dict = self.loss_function( self.DataLoader_,
                                            coef1=coef_1,
                                            coef2=coef_2,
                                            coef3=coef_3,
                                            coef4=coef_4,
                                            coef5=coef_5,
                                            ln_parm=ln_parm,
                                            fill_embeddings=fill_embeddings,
                                            minibatch_logging_rate=minibatch_logging_rate,
                                            binning=binning,
                                            weight_by_distance=weight_by_distance, )
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

    # TODO: calculate norms on max intensity
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
                      binning=False,
                      weight_by_distance=False):
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
        loss_dict = {'weighted_ln_loss': 0,
                    #  'contras_loss': 0,
                    #  'maxi_loss': 0,
                     'mse_loss': 0,
                     'train_loss': 0,
                     'sparse_max_loss': 0,
                     'l2_batchwise_loss': 0,
                     }
        weighted_ln_ = Weighted_LN_loss(coef=coef1,channels=self.num_fits).to(self.device)
        con_l = ContrastiveLoss(coef2).to(self.device)
        maxi_ = DivergenceLoss(train_iterator.batch_size, coef3).to(self.device)
        sparse_max = Sparse_Max_Loss(min_threshold=self.learning_rate,
                                        channels=self.num_fits, 
                                        coef=coef4).to(self.device)
        
        for i,(idx,x) in enumerate(tqdm(train_iterator, leave=True, total=len(train_iterator))):
            # tic = time.time()
            idx = idx.to(self.device).squeeze()
            x = x.to(self.device, dtype=torch.float).squeeze()

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x = self.Fitter(x)
            else: embedding, sd, mn, predicted_x = self.Fitter(x)
            
            # TODO: bin x and predicted_x based on data retrived from the dataloader
            # TODO: weight by euclidean distance to 1st point in the bin
            # i, idx, x are lists. Each list is the gaussian samples from a batch
            if binning:
                x = list(torch.split(x, self.dataloader_sampler.num_neighbors)) # Split the batch into groups based on the number of neighbors
                predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
                
                if weight_by_distance:
                    idx = torch.split(idx, self.dataloader_sampler.num_neighbors) # Split the indices into groups based on the number of neighbors
                    for i_, sample_group in enumerate(idx):
                        p_ind, shp = self.dataloader_sampler._which_particle_shape(sample_group[0]) # Determine the particle index and shape for the current sample group
                        coords = [((ind - p_ind) % shp[1], int((ind - p_ind) / shp[0])) for ind in sample_group] # Calculate the coordinates relative to the first point in the group
                        weights = torch.tensor([1]+[1 / (1 + ((coords[0][0] - coord[0]) ** 2 + (coords[0][1] - coord[1]) ** 2) ** 0.5) for coord in coords[1:]], # Calculate weights based on the Euclidean distance to the first point
                                                device = self.device) 
                        x[i_] = x[i_]*weights.unsqueeze(-1).unsqueeze(-1)
                        predicted_x[i_] = predicted_x[i_]*weights.unsqueeze(-1).unsqueeze(-1)
                        
                x = torch.stack([x_.mean(dim=0) for x_ in x]) # Sum the tensors in each group and stack them into a new batch
                predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])

            if coef1 > 0: 
                reg_loss_1 = weighted_ln_(embedding[:,:,0])
                loss_dict['weighted_ln_loss']+=reg_loss_1
            else: reg_loss_1 = 0

            if coef2 > 0: 
                contras_loss = con_l(embedding[:,:,0])
                loss_dict['contras_loss']+=contras_loss
            else: contras_loss = 0
                
            if coef3 > 0: 
                maxi_loss = maxi_(embedding[:,:,0])
                loss_dict['maxi_loss']+=maxi_loss
            else: maxi_loss = 0
            
            if coef4 > 0: # sparse_max_loss
                sparse_max_loss = sparse_max(embedding[:,:,0])
                loss_dict['sparse_max_loss']+=sparse_max_loss
            else: sparse_max_loss = 0
            
            if coef5 > 0: # set so the variation in x < fwhm, but the smaller the better.
                l2_loss = coef5*( (embedding[:,:,1]/embedding[:,:,2]).max(dim=0).values - \
                                  (embedding[:,:,1]/embedding[:,:,2]).min(dim=0).values ).mean()
                loss_dict['l2_batchwise_loss'] += l2_loss
                
            else: l2_loss = 0
            
            loss = F.mse_loss(x, predicted_x, reduction='mean');
            
            loss_dict['mse_loss'] += loss.item()
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss + l2_loss
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
        
        try: self.loss_dict = checkpoint['loss_dict']
        except: self.loss_dict = None
        
        try: self.loss_params = checkpoint['loss_params']
        except: self.loss_params = None
        
        try:
            with self.open_embedding_h() as h:
                print('embedding available')
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


class ConvAutoencoder_Multimodal():
    """builds the convolutional autoencoder
    """# TODO: decorator and setters for self.autoencoder.*stuff*

    def __init__(self,
                 encoder_step_size_1D,
                 encoder_step_size_2D,
                 pooling_list_1D,
                 pooling_list_2D,
                 decoder_step_size_1D,
                 decoder_step_size_2D,
                 upsampling_list_1D,
                 upsampling_list_2D,
                 embedding_size_1D,
                 embedding_size_2D,
                 embedding_size,
                 in_channels,
                 conv_size_1D,
                 conv_size_2D,
                 device,
                 attn_heads,
                 checkpoint = '',
                 learning_rate=3e-5,
                 emb_h5_path = './Combined_all_samples/embeddings.h5',
                 gen_h5_path = './Combined_all_samples/generated.h5',
                 in_parallel = True
                 ):
        """Initialization function

        Args:
            encoder_step_size (list): sets the size of the encoder
            pooling_list (list): sets the pooling list to define the pooling layers
            decoder_step_size (list): sets the size of the decoder
            upsampling_list (list): sets the size for upsampling
            embedding_size (int): number of embedding channels
            conv_size (int): sets the number of convolutional neurons in the model
            device (torch.device): set the device to run the model
            learning_rate (float, optional): sets the learning rate for the optimizer. Defaults to 3e-5.
        """
        self.encoder_step_size_1D = encoder_step_size_1D
        self.encoder_step_size_2D = encoder_step_size_2D

        self.pooling_list_1D = pooling_list_1D
        self.pooling_list_2D = pooling_list_2D

        self.decoder_step_size_1D = decoder_step_size_1D
        self.decoder_step_size_2D = decoder_step_size_2D

        self.upsampling_list_1D = upsampling_list_1D
        self.upsampling_list_2D = upsampling_list_2D

        self.embedding_size_1D = embedding_size_1D
        self.embedding_size_2D = embedding_size_2D
        self.stacked_embedding_size = embedding_size_1D+embedding_size_2D
        self.embedding_size = embedding_size


        self.conv_size_1D = conv_size_1D
        self.conv_size_2D = conv_size_2D
        
        self.channels_1D = in_channels
        self.device = device
        self.learning_rate = learning_rate
        self.attn_heads = attn_heads

        self.checkpoint = checkpoint
        # self.train = train

        self.emb_h5_path = emb_h5_path
        self.gen_h5_path = gen_h5_path
        self.in_parallel = in_parallel

        # complies the network
        self.compile_model()

    def open_embedding_h(self):
        check = self.checkpoint.split('/')[-1][:-4]
        h = h5py.File(self.emb_h5_path,'r+')
        try:
            self.embedding = h[f'embedding_{check}']
            self.scale_shear = h[f'scaleshear_{check}']
            self.rotation = h[f'rotation_{check}']                    
            self.translation = h[f'translation_{check}']
        except: 
            pass
        return h
    
    def open_generated_h(self):
        return h5py.File(self.gen_h5_path,'r+')
        # check = checkpoint.split('/')[-1][:-4]
        # return h[check]
    
    def compile_model(self):
        """function that complies the neural network model
        """
        # TODO:builds the 1d encoder
        self.encoder_1D = Encoder_1D(
            original_step_size=self.encoder_step_size_1D,
            pooling_list=self.pooling_list_1D,
            embedding_size=self.embedding_size_1D,
            conv_size=self.conv_size_1D,
            in_channels=self.channels_1D,
            device=self.device,
            attn_heads=self.attn_heads,
        ).to(self.device)

        # TODO: builds the 2d encoder
        self.encoder_2D = Encoder_2D(
            original_step_size=self.encoder_step_size_2D,
            pooling_list=self.pooling_list_2D,
            embedding_size=self.embedding_size_2D,
            conv_size=self.conv_size_2D,
            device=self.device,
        ).to(self.device)

        # TODO: builds the 1d decoder
        self.decoder_1D = Decoder_1D(
            original_step_size=self.decoder_step_size_1D,
            upsampling_list=self.upsampling_list_1D,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size_1D,
            out_channels=self.channels_1D,
            attn_heads=self.attn_heads,
        ).to(self.device)

        # TODO: builds the 2d decoder
        self.decoder_2D = Decoder_2D(
            original_step_size=self.decoder_step_size_2D,
            upsampling_list=self.upsampling_list_2D,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size_2D,
        ).to(self.device)


        # TODO:builds the autoencoder
        self.autoencoder = AutoEncoder(
            self.encoder_1D, self.encoder_2D, 
            self.decoder_1D, self.decoder_2D,
            self.stacked_embedding_size, self.embedding_size,
            self.device)
        if self.in_parallel:
            self.autoencoder = nn.DataParallel(self.autoencoder)
        self.autoencoder = self.autoencoder.to(self.device)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.learning_rate
        )

        # sets the datatype of the model to float32
        self.autoencoder.type(torch.float32)

    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              save_emb_every=1):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
            save_emb_every (int, optional): 
        """
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)

        # builds the dataloader
        self.DataLoader_ = DataLoader(
            data, batch_size=batch_size, shuffle=True)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, 
                step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_

        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            if epoch % save_emb_every ==0: # tell loss function to give embedding every however many epochs
                print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
                print('.............................')
                fill_embeddings = self.get_embedding(data, check=f'temp_{epoch}', no_calculate=True)

            train_dict = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, ln_parm,
                fill_embeddings=fill_embeddings)
            train_dict = {key: value / len(self.DataLoader_) for key, value in train_dict.items()}
            train_loss = sum([val for val in train_dict.values()])
            
            print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')

          #  schedular.step()
            # if best_train_loss > train_loss:
            best_train_loss = train_loss
            checkpoint = {
                "net": self.autoencoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": 0,
                "encoder_1D": self.encoder_1D.state_dict(),
                "encoder_2D": self.encoder_2D.state_dict(),
                'decoder_1D': self.decoder_1D.state_dict(),
                'decoder_2D': self.decoder_2D.state_dict(),
                'loss_dict': train_dict
            }
            if epoch >= 0:
                lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
                file_path = folder_path + f'/{save_date}_' +\
                    f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                    f'_trainloss:{train_loss:.4f}.pkl'
                torch.save(checkpoint, file_path)
                self.checkpoint = file_path

            if epoch%save_emb_every==0: #TODO: why isn't this saving properly?
                with self.open_embedding_h() as h:
                    check = file_path.split('/')[-1][:-4]
                    h[f'embedding_{check}'] = h[f'embedding_temp'] # combined embedding
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
                    print(f'saved {check} ')
                    # h.flush()
                    # h.close()
                        
        if scheduler is not None:
            scheduler.step()

    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.autoencoder.train()
        loss_dict = {'contrastive': 0,
                     'divergence': 0,
                     'l1': 0,
                     'l2': 0,
                     'mse_1d': 0,
                     'mse_2d': 0,}
        
        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)

        for idx,diff,spec in tqdm(train_iterator, leave=True, total=len(train_iterator)):
            # tic = time.time()
            sorted, indices = torch.sort(idx)
            sorted = sorted.detach().numpy()

            diff = diff.to(self.device, dtype=torch.float)
            spec = spec.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(diff.shape[0], coef2).to(self.device) # based on batchsize

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: 
                embedding, predicted_1D, predicted_2D, scale_shear,rotation,translation = self.autoencoder((diff, spec))
            else: 
                embedding, sd, mn, predicted_1D, predicted_2D = self.autoencoder((diff, spec))

            reg_loss_1 = coef*torch.norm(embedding, ln_parm).to(self.device)/diff.shape[0]
            # if reg_loss_1 == 0: reg_loss_1 = 0.5

            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)

            # reconstruction loss
            mask = (predicted_2D!=0)
            loss_1D = F.mse_loss(spec, predicted_1D, reduction='mean');
            loss_2D = F.mse_loss(diff, predicted_2D, reduction='mean');
            loss_2D = (loss_2D*mask.float()).sum()
            loss_2D /= mask.sum()

            loss_dict['contrastive']+=contras_loss.item()
            loss_dict['divergence']-=maxi_loss.item()
            loss_dict['l1']+=reg_loss_1.item()
            loss_dict['mse_1d']+=loss_1D.item()
            loss_dict ['mse_2d']+=loss_2D.item()
            
            loss = loss_1D + loss_2D + reg_loss_1 + contras_loss - maxi_loss
            train_loss += loss.item()

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()
            
            # fill embedding if the correct epoch
            if fill_embeddings:
                with self.open_embedding_h() as h:
                    # scale_shear,rotation,translation = self.autoencoder.temp_affines
                    self.embedding[sorted] = embedding[indices].cpu().detach().numpy()
                    self.scale_shear[sorted] = scale_shear[indices].cpu().reshape((-1,6)).detach().numpy()
                    self.rotation[sorted] = rotation[indices].reshape((-1,6)).cpu().detach().numpy()
                    self.translation[sorted] = translation[indices].reshape((-1,6)).cpu().detach().numpy()
                    # print('\tt', abs(tic-toc)) # 2.7684452533721924
                    h.flush()

        return loss_dict

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['net'])
        self.encoder_1D.load_state_dict(checkpoint['encoder_1D'])
        self.encoder_2D.load_state_dict(checkpoint['encoder_2D'])
        self.decoder_1D.load_state_dict(checkpoint['decoder_1D'])
        self.decoder_2D.load_state_dict(checkpoint['decoder_2D'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        check = path_checkpoint.split('/')[-1][:-4]

        try:
            h = h5py.File(self.emb_h5_path,'r+')
            self.embedding = h[f'embedding_{check}']
            self.scale_shear = h[f'scaleshear_{check}']
            self.rotation = h[f'rotation_{check}']                    
            self.translation = h[f'translation_{check}']
        except Exception as error:
            print(error)
            print('Embedding and affines not opened')

        try:
            h = h5py.File(self.gen_h5_path,'r+')
            self.generated = h[check]
        except Exception as error:
            print(error)
            print('Generated not opened')

    def get_embedding(self, data, check = None, batch_size=32, no_calculate=True):
        """extracts embeddings from the data

        Args:
            data (torch.tensor): data to get embeddings from
            batch_size (int, optional): batchsize for inference. Defaults to 32.

        Returns:
            torch.tensor: predicted embeddings
        """

        # builds the dataloader
        dataloader = DataLoader(data, batch_size, shuffle=False)

        try:
            try: h = h5py.File(self.emb_h5_path,'w')
            except: h = h5py.File(self.emb_h5_path,'r+')

            if check==None: 
                check = self.checkpoint.split('/')[-1][:-4]
            
            try:
                embedding_ = h.create_dataset(f'embedding_{check}', data = np.zeros([data.shape[0][0], self.embedding_size]))
                scale_shear_ = h.create_dataset(f'scaleshear_{check}', data = np.zeros([data.shape[0][0],6]))
                rotation_ = h.create_dataset(f'rotation_{check}', data = np.zeros([data.shape[0][0],6]))
                translation_ = h.create_dataset(f'translation_{check}', data = np.zeros([data.shape[0][0],6]))
            except: 
                embedding_ = h[f'embedding_{check}']
                scale_shear_ = h[f'scaleshear_{check}']
                rotation_ = h[f'rotation_{check}']                    
                translation_ = h[f'translation_{check}']

            self.embedding = embedding_
            self.scale_shear = scale_shear_
            self.rotation = rotation_
            self.translation = translation_

        except Exception as error:
            print(error) 
            assert no_calculate,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
            h.flush()
                
        if no_calculate: 
            print('Created empty h5 embedding datasets to fill during training')
            h.flush()
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            self.autoencoder.get_embeddings = True
            for x in tqdm(dataloader, leave=True, total=len(dataloader)):
                with torch.no_grad():
                    i=x[0].detach().numpy()
                    test_values = [Variable(x[1].to(self.device)).float(), 
                                   Variable(x[2].to(self.device)).float() ]
                    embedding,scale_shear,rotation,translation = self.autoencoder(test_values)
                    
                    self.embedding[i[0]:i[-1]+1, :] = embedding.cpu()
                    self.scale_shear[i[0]:i[-1]+1, :] = scale_shear.reshape(-1,6).cpu().detach().numpy()
                    self.rotation[i[0]:i[-1]+1, :] = rotation.reshape(-1,6).cpu().detach().numpy()
                    self.translation[i[0]:i[-1]+1, :] = translation.reshape(-1,6).cpu().detach().numpy()
        h.flush()
        h.close()

    def stack_emb_affines(self):
        with self.open_embedding_h() as h:
            check = self.checkpoint.split('/')[-1][:-4]
            scaler = StandardScaler()
            emblist = [h[f'embedding_{check}'][:,i].reshape(-1,1) for i in range(32)]
            scaled_embs = list(map( scaler.fit_transform,emblist) )
            rotation = scaler.fit_transform(get_theta(h[f'rotation_{check}']).reshape(-1,1))
            translationx = scaler.fit_transform(h[f'translation_{check}'][:,2].reshape(-1,1))
            translationy = scaler.fit_transform(h[f'translation_{check}'][:,5].reshape(-1,1))
            scalex = scaler.fit_transform(h[f'scaleshear_{check}'][:,0].reshape(-1,1))
            shearx = scaler.fit_transform(h[f'scaleshear_{check}'][:,1].reshape(-1,1))
            scaley = scaler.fit_transform(h[f'scaleshear_{check}'][:,4].reshape(-1,1))
            shearx = scaler.fit_transform(h[f'scaleshear_{check}'][:,3].reshape(-1,1))
            scaled_embs+=[rotation,translationx,translationy,scalex,shearx,scaley,shearx]
        return np.stack(scaled_embs,axis=1).squeeze()
    
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

    def generate_by_labels(self,meta,labels,
                            checkpoint=None,
                            channels=None,
                            #  averaging_number=100,
                            overwrite=False,
                            with_affine=False,
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

        assert not self.train, 'set self.train to False'
        
        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        if channels==None: channels = np.arange(self.embedding_size)

        # gets the embedding if a specific embedding is not provided
        try:
            embedding = self.embedding
        except Exception as error:
            print(error)
            assert False, 'Make sure model is set to appropriate embeddings first'

        try: # try opening h5 file
            try: # make new file
                h = h5py.File(self.gen_h5_path,'r+')
            except: # open existing file
                h = h5py.File(self.gen_h5_path,'w')

            if checkpoint==None: check = self.checkpoint.split('/')[-1][:-4]
            else: check = checkpoint
            
            try: # make new dataset
                if overwrite and check in h: del h[check]
                self.generated = h.create_dataset(check,
                                            data=np.zeros( [len(meta['particle_list']),
                                                            len(np.unique(labels)),
                                                            len(channels),
                                                            128,128] ) )
                self.generated.attrs['clustered']=True      
                print(check)
            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        # for p,p_name in enumerate(meta['particle_list']): # each sample
        #     print(p, p_name)
        #     data=self.embedding[meta['particle_inds'][p]:\
        #                         meta['particle_inds'][p+1]]
        for p,p_name in enumerate(meta['particle_list']): # each sample
            print(p, p_name)
            # shape 128*128, 32
            data=self.embedding[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
            data_labels=labels[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
            inds=None
            # loops around the number of iterations to generate
            for i in tqdm(np.unique(labels)):
                inds = np.argwhere(data_labels==i)
            
                # # specifically updates the value of the mean embedding image to visualize 
                # # based on the linear spaced vector
                # gen_value[channel] = value[i]

                # # generates diffraction pattern
                # for ind in inds:
                #     self.generated[ind,i] = self.generate_spectra(embedding[ind,i,j]).squeeze()
                # gen_values

                # generate spectrum at each index
                gen_value = np.atleast_2d(data[inds]).mean(axis=0).squeeze()
                
                self.generated[p,i] = self.generate_spectra(gen_value)
                # time.sleep(0.5)
        h.close()

    def generate_range(self,meta,checkpoint,
                         ranges=None,
                         generator_iters=50,
                         averaging_number=100,
                         overwrite=False,
                         with_affine=False,
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

        assert not self.train, 'set self.train to False if calculating manually'
        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        channels = np.arange(self.embedding_size)
        # sets the channels to use in the object
        if "channels" in kwargs:
            channels = kwargs["channels"]
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
            try: # make new generated file
                if overwrite and check in h: del h[check]
                self.generated_diff = h.create_dataset(check,
                                            data=np.zeros( [len(meta['particle_list']),
                                                            generator_iters,
                                                            len(channels),
                                                            self.encoder_step_size_2D[0],
                                                            self.encoder_step_size_2D[1]] ) )
                self.generated_spec = h.create_dataset(check,
                                            data=np.zeros( [len(meta['particle_list']),
                                                            generator_iters,
                                                            len(channels),
                                                            self.channels_1D,
                                                            self.encoder_step_size_1D] ) )
            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        for p,p_name in enumerate(meta['particle_list']): # each sample
            print(p, p_name)
            data=self.embedding[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
                
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

                    # computes the mean of the selected index across embeddings shp (1,emb_size)
                    gen_value = np.mean(data[idx], axis=0)

                    # specifically updates the value of the mean embedding image to visualize 
                    # based on the linear spaced vector
                    gen_value[channel] = value[i]

                    # generates diffraction pattern
                    self.generated[meta['particle_inds'][p]: meta['particle_inds'][p+1],i,j] =\
                        self.generate_spectra(gen_value).squeeze()       
        h.close()

        # return self.generated
                    
    def generate_spectra(self, embedding,with_affine=False):
        """generates spectra from embeddings

        Args:
            embedding (torch.tensor): predicted embeddings to decode

        Returns:
            torch.tensor: decoded spectra
        """

        embedding = torch.from_numpy(np.atleast_2d(embedding)).to(self.device)
        predicted_1D = self.dec_1D(embedding)
        predicted_2D = self.dec_2D(embedding)
        if with_affine:
            predicted_2D = self.transformer(predicted_2D)
        predicted_1D = predicted_1D.cpu().detach().numpy()
        predicted_2D = predicted_2D.cpu().detach().numpy()
        return predicted_1D,predicted_2D


class Multiscale1DFitter(nn.Module):
    def __init__(self, function, x_data, input_channels, num_params, num_fits, limits=[1,975,25,1,25,1], scaler=None, 
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
        self.post_processing = post_processing
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
            nn.SELU(), nn.Conv1d(in_channels=x1_ch_list[0], #8
                                 out_channels=x1_ch_list[1], kernel_size=7),
            nn.SELU(), nn.Conv1d(in_channels=x1_ch_list[1], #6
                                 out_channels=x1_ch_list[2], kernel_size=5), #4
            nn.SELU(), nn.AdaptiveAvgPool1d(x1_pool)
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
            embedding, self.x_data, limits=self.limits, device=self.device, return_params=True )
        # print('fitted shape:', fits.shape)
        
        if not return_sum: return params, fits
        
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


class ComplexPostProcessor:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute(self, fits):
        # extract and return real and imaginary
        real = torch.real(fits)
        real_scaled = (real - torch.tensor(self.dataset.raw_data_scaler.real_scaler.mean).cuda()) / torch.tensor(
            self.dataset.raw_data_scaler.real_scaler.std
        ).cuda()
        imag = torch.imag(fits)
        imag_scaled = (imag - torch.tensor(self.dataset.raw_data_scaler.imag_scaler.mean).cuda()) / torch.tensor(
            self.dataset.raw_data_scaler.imag_scaler.std
        ).cuda()
        out = torch.stack((real_scaled, imag_scaled), 2)

        return out


class Model_SHO(nn.Module):

    def __init__(self,
                 model,
                 dataset,
                 model_basename='',
                 training=True,
                 path='Trained Models/SHO Fitter/',
                 device='cuda',
                 **kwargs):

        super().__init__()
        self.model = model
        self.model.dataset = dataset
        self.model.training = True
        self.model_name = model_basename
        self.path = make_folder(path)

    def fit(self,
            data_train,
            batch_size=200,
            epochs=5,
            loss_func=torch.nn.MSELoss(),
            optimizer='Adam',
            seed=42,
            datatype=torch.float32,
            save_all=False,
            write_CSV=None,
            closure=None,
            basepath=None,
            early_stopping_loss=None,
            early_stopping_count=None,
            early_stopping_time=None,
            save_training_loss=True,
            i = None,
            **kwargs):

        loss_ = []

        if basepath is not None:
            path = f"{self.path}/{basepath}/"
            make_folder(path)
            print(f"Saving to {path}")
        else:
            path = self.path

        # sets the model to be a specific datatype and on cuda
        self.to(datatype).to(self.device)

        # Note that the seed will behave differently on different hardware targets (GPUs)
        random_seed(seed=seed)

        torch.cuda.empty_cache()

        # selects the optimizer
        if optimizer == 'Adam':
            optimizer_ = torch.optim.Adam(self.model.parameters())
        elif optimizer == "AdaHessian":
            optimizer_ = AdaHessian(self.model.parameters(), lr=.5)
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer_ = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        elif isinstance(optimizer, dict):
            if optimizer['name'] == "TRCG":
                optimizer_ = optimizer['optimizer'](
                    self.model, optimizer['radius'], optimizer['device'])
        else:
            try:
                optimizer = optimizer(self.model.parameters())
            except:
                raise ValueError("Optimizer not recognized")

        # instantiate the dataloader
        train_dataloader = DataLoader(
            data_train, batch_size=batch_size, shuffle=True)

        # if trust region optimizers stores the TR optimizer as an object and instantiates the ADAM optimizer
        if isinstance(optimizer_, TRCG):
            TRCG_OP = optimizer_
            optimizer_ = torch.optim.Adam(self.model.parameters(), **kwargs)

        total_time = 0
        low_loss_count = 0

        # says if the model have already stopped early
        already_stopped = False

        model_updates = 0

        # loops around each epoch
        for epoch in range(epochs):

            train_loss = 0.0
            total_num = 0
            epoch_time = 0

            # sets the model to training mode
            self.model.train()

            for train_batch in train_dataloader:

                model_updates += 1

                # starts the timer
                start_time = time.time()

                train_batch = train_batch.to(datatype).to(self.device)

                if "TRCG_OP" in locals() and epoch > optimizer.get("ADAM_epochs", -1):

                    def closure(part, total, device):
                        pred, embedding = self.model(train_batch)
                        pred = pred.to(torch.float32)
                        pred = torch.atleast_3d(pred)
                        embedding = embedding.to(torch.float32)
                        loss = loss_func(train_batch, pred)
                        return loss

                    # if closure is not None:
                    loss, radius, cnt_compute, cg_iter = TRCG_OP.step(
                        closure)
                    train_loss += loss * train_batch.shape[0]
                    total_num += train_batch.shape[0]
                    optimizer_name = "Trust Region CG"
                else:
                    pred, embedding = self.model(train_batch)
                    pred = pred.to(torch.float32)
                    pred = torch.atleast_3d(pred)
                    embedding = embedding.to(torch.float32)
                    optimizer_.zero_grad()
                    loss = loss_func(train_batch, pred)
                    loss.backward(create_graph=True)
                    train_loss += loss.item() * pred.shape[0]
                    total_num += pred.shape[0]
                    optimizer_.step()
                    if isinstance(optimizer_, torch.optim.Adam):
                        optimizer_name = "Adam"
                    elif isinstance(optimizer_, AdaHessian):
                        optimizer_name = "AdaHessian"

                epoch_time += (time.time() - start_time)

                total_time += (time.time() - start_time)

                try:
                    loss_.append(loss.item())
                except:
                    loss_.append(loss)

                if early_stopping_loss is not None and already_stopped == False:
                    if loss < early_stopping_loss:
                        low_loss_count += train_batch.shape[0]
                        if low_loss_count >= early_stopping_count:
                            torch.save(self.model.state_dict(),
                                       f"{path}/Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss/total_num}.pth")

                            write_csv(write_CSV,
                                      path,
                                      self.model_name,
                                      i,
                                      self.model.dataset.noise,
                                      optimizer_name,
                                      epoch,
                                      total_time,
                                      train_loss/total_num,
                                      batch_size,
                                      loss_func,
                                      seed,
                                      True,
                                      model_updates)

                            already_stopped = True
                    else:
                        low_loss_count -= (train_batch.shape[0]*5)

            if "verbose" in kwargs:
                if kwargs["verbose"] == True:
                    print(f"Loss = {loss.item()}")

            train_loss /= total_num

            print(optimizer_name)
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                                                              1, epochs, train_loss))
            print("--- %s seconds ---" % (epoch_time))

            # scheduler.step(train_loss)
            # Print the current learning rate (optional)
            current_lr = optimizer_.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

            if save_all:
                torch.save(self.model.state_dict(),
                           f"{path}/{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")

            if early_stopping_time is not None:
                if total_time > early_stopping_time:
                    torch.save(self.model.state_dict(),
                               f"{path}/Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")

                    write_csv(write_CSV,
                              path,
                              self.model_name,
                              i,
                              self.model.dataset.noise,
                              optimizer_name,
                              epoch,
                              total_time,
                              train_loss,  # already divided by total_num
                              batch_size,
                              loss_func,
                              seed,
                              True,
                              model_updates)
                    break

        torch.save(self.model.state_dict(),
                   f"{path}/{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth")
        write_csv(write_CSV,
                  path,
                  self.model_name,
                  i,
                  self.model.dataset.noise,
                  optimizer_name,
                  epoch,
                  total_time,
                  train_loss,  # already divided by total_num
                  batch_size,
                  loss_func,
                  seed,
                  False,
                  model_updates)

        if save_training_loss:
            save_list_to_txt(
                loss_, f"{path}/Training_loss_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.txt")

        self.model.eval()

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def inference_timer(self, data, batch_size=.5e4):
        torch.cuda.empty_cache()

        batch_size = int(batch_size)

        dataloader = DataLoader(data, batch_size)

        # Computes the inference time
        computeTime(self.model, dataloader, batch_size, device=self.device)

    def predict(self, data, batch_size=10000,
                single=False,
                translate_params=True,
                is_SHO=True):

        self.model.eval()

        dataloader = DataLoader(data, batch_size=batch_size)

        # preallocate the predictions
        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        data = data.clone().detach().requires_grad_(True)
        predictions = torch.zeros_like(data.clone().detach())
        params_scaled = torch.zeros((data.shape[0], self.model.num_params))
        params = torch.zeros((data.shape[0], self.model.num_params))

        # compute the predictions
        for i, train_batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size

            if i == num_batches - 1:
                end = num_elements

            pred_batch, params_scaled_, params_ = self.model(
                train_batch.to(self.device))

            if is_SHO:
                predictions[start:end] = pred_batch.cpu().detach()
            else:
                predictions[start:end] = torch.unsqueeze(
                    pred_batch.cpu().detach(), 2) #12/5/2023
            params_scaled[start:end] = params_scaled_.cpu().detach()
            params[start:end] = params_.cpu().detach()

            torch.cuda.empty_cache()

        # converts negative ampltiudes to positive and shifts the phase to compensate
        if translate_params:
            params[params[:, 0] < 0, 3] = params[params[:, 0] < 0, 3] - np.pi
            params[params[:, 0] < 0, 0] = np.abs(params[params[:, 0] < 0, 0])

        if self.model.dataset.NN_phase_shift is not None:
            params_scaled[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params_scaled[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))
            params[:, 3] = torch.Tensor(self.model.dataset.shift_phase(
                params[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift))

        return predictions, params_scaled, params

    @staticmethod
    def mse_rankings(true, prediction, curves=False):

        def type_conversion(data):

            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim-1)

            return data

        true = type_conversion(true)
        prediction = type_conversion(prediction)

        errors = Model.MSE(prediction, true)

        index = np.argsort(errors)

        if curves:
            # true will be in the form [ranked error, channel, timestep]
            return index, errors[index], true[index], prediction[index]

        return index, errors[index]

    @staticmethod
    def MSE(true, prediction):

        # calculates the mse
        mse = np.mean((true.reshape(
            true.shape[0], -1) - prediction.reshape(true.shape[0], -1))**2, axis=1)

        # converts to a scalar if there is only one value
        if mse.shape[0] == 1:
            return mse.item()

        return mse

    @staticmethod
    def get_rankings(raw_data, pred, n=1, curves=True):
        """simple function to get the best, median and worst reconstructions

        Args:
            raw_data (np.array): array of the true values
            pred (np.array): array of the predictions
            n (int, optional): number of values for each. Defaults to 1.
            curves (bool, optional): whether to return the curves or not. Defaults to True.

        Returns:
            ind: indices of the best, median and worst reconstructions
            mse: mse of the best, median and worst reconstructions
        """
        index, mse, d1, d2 = Model.mse_rankings(
            raw_data, pred, curves=curves)
        middle_index = len(index) // 2
        start_index = middle_index - n // 2
        end_index = start_index + n

        ind = np.hstack(
            (index[:n], index[start_index:end_index], index[-n:])).flatten().astype(int)
        mse = np.hstack(
            (mse[:n], mse[start_index:end_index], mse[-n:]))

        d1 = np.stack(
            (d1[:n], d1[start_index:end_index], d1[-n:])).squeeze()
        d2 = np.stack(
            (d2[:n], d2[start_index:end_index], d2[-n:])).squeeze()

        # return ind, mse, np.swapaxes(d1[ind], 1, d1.ndim-1), np.swapaxes(d2[ind], 1, d2.ndim-1)
        return ind, mse, d1, d2

    def print_mse(self, data, labels, is_SHO=True):
        """prints the MSE of the model

        Args:
            data (tuple): tuple of datasets to calculate the MSE
            labels (list): List of strings with the names of the datasets
        """

        # loops around the dataset and labels and prints the MSE for each
        for data, label in zip(data, labels):

            if isinstance(data, torch.Tensor):
                # computes the predictions
                pred_data, scaled_param, parm = self.predict(data, is_SHO=is_SHO)
            elif isinstance(data, dict):
                pred_data, _ = self.model.dataset.get_raw_data_from_LSQF_SHO(
                    data)
                data, _ = self.model.dataset.NN_data()
                pred_data = torch.from_numpy(pred_data)

            # Computes the MSE
            out = nn.MSELoss()(data, pred_data)

            # prints the MSE
            print(f"{label} Mean Squared Error: {out:0.4f}")    
 

def get_gaussian_parameters_1D(embedding,limits,kernel_size,amp_activation=nn.ReLU()): # add activations
    """
    For 1D gaussian
    Parameters:
        embedding (Tensor): The embedding tensor with shape (ch, batch, 6).
        limits (tuple): A tuple containing the limits for [amplitude, mean, and covariance] of 2D gaussian.
        kernel_size (int): The size of the output image.
    Returns:
        tuple: A tuple containing amplitude, theta, mean_x, mean_y, cov_x, cov_y
    """
    amplitude = limits[0]*amp_activation(embedding[:,:,0]) # Look at limits before activations
    m = limits[1]/2
    n = limits[2]/2
    mean = torch.clamp(m*nn.Tanh()(embedding[:,:,1]) + m, min=1e-3, max=limits[1])
    cov = torch.clamp(n*nn.Tanh()(embedding[:,:,2]) + n, min=1e-3, max=limits[2])
    
    return amplitude, mean, cov
    
def get_lorentzian_parameters_1D(embedding,limits,kernel_size,amp_activation=nn.ReLU()): # add activations
    """
    For 1D lorentzian
    Parameters:
        embedding (Tensor): The embedding tensor with shape (ch, batch, 6).
        limits (tuple): A tuple containing the limits for [amplitude, mean, and covariance] of 1D gaussian.
        kernel_size (int): The size of the output image.
    Returns:
        tuple: A tuple containing amplitude, theta, mean_x, mean_y, cov_x, cov_y
    """
    m = limits[1]/2
    amplitude = limits[0]*amp_activation(embedding[:,:,0]) # Look at limits before activations
    gamma_x = torch.clamp(m*nn.Tanh()(embedding[:,:,1]) + m, min=0, max=limits[1])
    eta = (0.5*nn.Tanh()(embedding[:,:,2]) + 0.5)
    return amplitude,gamma_x, eta # look at limits after activations

def generate_pseudovoigt_1D(embedding, dset, limits=[1,1,975,975], device='cpu',return_params=False):
    '''https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        embedding is: 
        A: Area under curve
        I_b: baseline intensity
        x: mean x of the distributions
        wx: x FWHM
        nu: lorentzian character fraction
        t: rotation angle
       
       shape should be (_, num_fits, x_, y_)
    '''
    # TODO: try to have all values in embedding between 0-1
    A = limits[0] * nn.ReLU()(embedding[..., 0]) # area under curve TODO: best way to scale this?
    # Ib = limits[1] * nn.ReLU()(embedding[..., 1])
    x = torch.clamp(limits[1]/2 * nn.Tanh()(embedding[..., 1]) + limits[1]/2, min=1e-3) # mean
    w = torch.clamp(limits[2]/2 * nn.Tanh()(embedding[..., 2]) + limits[2]/2, min=1e-3) # fwhm
    nu = 0.5 * nn.Tanh()(embedding[..., 3]) + 0.5 # fraction voight character

    s = x.shape  # (_, num_fits)
    
    x_ = torch.arange(dset.spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)
    
    # Gaussian component
    gaussian = A.unsqueeze(-1)*(4*torch.log(torch.tensor(2))/torch.pi)**0.5 / w.unsqueeze(-1) * \
            torch.exp(-4*torch.log(torch.tensor(2)) / w.unsqueeze(-1)**2 * (x_-x.unsqueeze(-1))**2)

    # Lorentzian component (simplified version)
    lorentzian = A.unsqueeze(-1)*( 2/torch.pi * w.unsqueeze(-1) / \
                                   (4*(x_-x.unsqueeze(-1))**2 + w.unsqueeze(-1)**2) )
    
    # Pseudo-Voigt profile
    pseudovoigt = nu.unsqueeze(-1)*lorentzian + (1-nu.unsqueeze(-1))*gaussian #+  Ib.unsqueeze(-1)

    if return_params: return pseudovoigt.to(torch.float32), torch.stack([A,x,w,nu],axis=2)
    return pseudovoigt.to(torch.float32)
      
class ConvBlock_1D(nn.Module):
    """Convolutional Block with 3 convolutional layers, 1 layer normalization layer with ReLU and ResNet

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, t_size, n_step, attn_heads,):
        """Initializes the convolutional block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(ConvBlock_1D, self).__init__()
        self.cov1d_1 = nn.Conv1d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_2 = nn.Conv1d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_3 = nn.Conv1d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        
        self.attention_1 = nn.MultiheadAttention(n_step, attn_heads)
        self.attention_2 = nn.MultiheadAttention(n_step, attn_heads)
        self.attention_3 = nn.MultiheadAttention(n_step, attn_heads)
        
        # self.norm_3 = nn.LayerNorm(n_step)
        # self.relu_4 = nn.ReLU()

        self.norm_1 = nn.LayerNorm([n_step])
        self.norm_2 = nn.LayerNorm([n_step])
        self.norm_3 = nn.LayerNorm([n_step])
        self.drop = nn.Dropout(p=0.2)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()
        
    def forward(self, x): ## TODO: should I use all the norm and relu layers?
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        # out = self.cov1d_1(x)
        # out = self.cov1d_2(out)
        # out = self.cov1d_3(out)
        # out = self.norm_3(out)
        # out = self.relu_4(out)
        # out = out.add(x_input)
        
        x_k_v = x_input.transpose(0,1)
        #1
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu_1(out)
        out = out.transpose(0,1)
        out,_ = self.attention_1(out,x_k_v,x_k_v)
        out = out.transpose(0,1)
        #2
        out = self.cov1d_2(out)
        out = self.norm_2(out)
        out = self.relu_2(out)
        out = out.transpose(0,1)
        out,_ = self.attention_2(out,x_k_v,x_k_v)
        out = out.transpose(0,1)
        #3
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_3(out)
        out = out.transpose(0,1)
        out,_ = self.attention_3(out,x_k_v,x_k_v)
        out = out.transpose(0,1)
        
        out = self.relu_4(out)
        out = out.add(x_input)
        output = self.drop(out)
        
        return output    
    

        return out

class ConvBlock_2D(nn.Module):
    """Convolutional Block with 3 convolutional layers, 1 layer normalization layer with ReLU and ResNet

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, t_size, n_step):
        """Initializes the convolutional block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(ConvBlock_2D, self).__init__()
        self.cov2d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_2 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_3 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov2d_1(x)
        out = self.cov2d_2(out)
        out = self.cov2d_3(out)
        out = self.norm_3(out)
        out = self.relu_4(out)
        out = out.add(x_input)

        return out
  
    
class IdentityBlock_1D(nn.Module):

    """Identity Block with 1 convolutional layers, 1 layer normalization layer with ReLU"""

    def __init__(self, t_size, n_step, attn_heads):
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(IdentityBlock_1D, self).__init__()
        self.cov1d_1 = nn.Conv1d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.attention_1 = nn.MultiheadAttention(n_step, attn_heads)
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        x_k_v = x_input.transpose(0,1)
        
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)
        
        out = out.transpose(0,1)
        out,_ = self.attention_1(out,x_k_v,x_k_v) #implement self-attention layer
        out = out.transpose(0,1)
        # out = self.drop(out)       
        
        return out
       
class IdentityBlock_2D(nn.Module):

    """Identity Block with 1 convolutional layers, 1 layer normalization layer with ReLU"""

    def __init__(self, t_size, n_step):
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(IdentityBlock_2D, self).__init__()
        self.cov2d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov2d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class Encoder_1D(nn.Module):
    """Encoder block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, 
                 original_step_size, 
                 pooling_list, 
                 embedding_size, 
                 conv_size, 
                 in_channels, 
                 device,
                 attn_heads):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(Encoder_1D, self).__init__()
        self.device = device
        blocks = []

        self.in_channels = in_channels
        self.input_size = original_step_size
        number_of_blocks = len(pooling_list)

        blocks.append( ConvBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads) )
        blocks.append( IdentityBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads) )
        blocks.append( nn.MaxPool1d(pooling_list[0], stride=pooling_list[0]) )

        for i in range(1, number_of_blocks):
            original_step_size = original_step_size // pooling_list[i - 1]
            
            blocks.append( ConvBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads) )
            blocks.append( IdentityBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads) )
            blocks.append( nn.MaxPool1d(pooling_list[i], stride=pooling_list[i]) )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        original_step_size = original_step_size // pooling_list[-1]
            
        input_size = original_step_size*in_channels

        self.cov1d = nn.Conv1d(
            in_channels, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_1 = nn.Conv1d(
            conv_size, in_channels, 3, stride=1, padding=1, padding_mode="zeros"
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
        out = x.view(-1, self.in_channels, self.input_size)
        out = self.cov1d(out)
        for i in range(self.layers):
            layer = self.block_layer[i]
            out = self.block_layer[i](out)
        out = self.cov1d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        selection = self.relu_1(out)

        return selection

class Encoder_2D(nn.Module):
    """Encoder block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, original_step_size, pooling_list, embedding_size, conv_size, device):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(Encoder_2D, self).__init__()
        self.device = device
        blocks = []

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]

        number_of_blocks = len(pooling_list)

        blocks.append(ConvBlock_2D(t_size=conv_size,
                                n_step=original_step_size))
        blocks.append(IdentityBlock_2D(
            t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(
            pooling_list[0], stride=pooling_list[0]))

        for i in range(1, number_of_blocks):
            original_step_size = [
                original_step_size[0] // pooling_list[i - 1],
                original_step_size[1] // pooling_list[i - 1],
            ]
            blocks.append(ConvBlock_2D(t_size=conv_size,
                                    n_step=original_step_size))
            blocks.append(
                IdentityBlock_2D(t_size=conv_size, n_step=original_step_size)
            )
            blocks.append(nn.MaxPool2d(
                pooling_list[i], stride=pooling_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        original_step_size = [
            original_step_size[0] // pooling_list[-1],
            original_step_size[1] // pooling_list[-1],
        ]
        input_size = original_step_size[0] * original_step_size[1]

        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        self.relu_1 = nn.ReLU()

        self.dense = nn.Linear(input_size, embedding_size)
        self.affine_dense = nn.Linear(embedding_size,7) # TODO: save in (nx6) instead of (nx2x3)
        self.affine = Affine_Transform(self.device, Symmetric = False,
                                    mask_intensity = False)

    def forward(self, x):
        """Forward pass of the encoder

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        selection = self.relu_1(out)

        # get affine transforms from dense
        affine_dense = self.affine_dense(selection) 
        scaler_shear, rotation, translation, mask_parameter = self.affine(affine_dense)

        return selection,scaler_shear, rotation, translation


class Decoder_1D(nn.Module):
    """Decoder class

    Args:
        nn (nn.module): base class for all neural network modules
    """

    def __init__(self,
        original_step_size,
        upsampling_list,
        embedding_size, 
        conv_size,
        out_channels,
        attn_heads
    ):
        """Decoder block

        Args:
            original_step_size (Int): the x and y size of input image
            upsampling_list (Int): the list of parameter for each 2D upsample layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
            pooling_list (List): the list of parameter for each 2D MaxPool layer
        """

        super(Decoder_1D, self).__init__()
        self.input_size = original_step_size
        self.channels = out_channels
                
        self.dense = nn.Linear( # for here, should I do two separate channels, or just repeat the embedding?
            embedding_size, original_step_size*out_channels
        )
        self.cov1d = nn.Conv1d(
            out_channels, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )

        blocks = []
        number_of_blocks = len(upsampling_list)
        blocks.append( 
            ConvBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads)
        )
        blocks.append( 
            IdentityBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads)
        )
        
        for i in range(number_of_blocks):
            blocks.append(
                nn.Upsample(
                    scale_factor=upsampling_list[i]
                )
            )
            original_step_size = original_step_size * upsampling_list[i]
            
            blocks.append(
                ConvBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads)
            )
            blocks.append(
                IdentityBlock_1D(t_size=conv_size, n_step=original_step_size, attn_heads = attn_heads)
            )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        self.output_size = original_step_size

        self.cov1d_1 = nn.Conv1d(
            conv_size, out_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )
        
    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        out = self.dense(x)
        out = out.view(-1, self.channels, self.input_size) # batch_size, numb channels, embedding size
        out = self.cov1d(out)
        for i in range(self.layers):
            # layer = self.block_layer[i]
            out = self.block_layer[i](out)
        out = self.cov1d_1(out)
        output = out.view(-1, self.channels, self.output_size)

        return output

class Decoder_2D(nn.Module):
    """Decoder class

    Args:
        nn (nn.module): base class for all neural network modules
    """

    def __init__(
        self,
        original_step_size,
        upsampling_list,
        embedding_size,
        conv_size,
    ):
        """Decoder block

        Args:
            original_step_size (Int): the x and y size of input image
            upsampling_list (Int): the list of parameter for each 2D upsample layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
            pooling_list (List): the list of parameter for each 2D MaxPool layer
        """

        super(Decoder_2D, self).__init__()
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(
            embedding_size, original_step_size[0] * original_step_size[1]
        )
        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        blocks = []
        number_of_blocks = len(upsampling_list)
        blocks.append(ConvBlock_2D(
            t_size=conv_size, n_step=original_step_size)
        )
        blocks.append(IdentityBlock_2D(
            t_size=conv_size, n_step=original_step_size)
        )
        for i in range(number_of_blocks):
            blocks.append(
                nn.Upsample(
                    scale_factor=upsampling_list[i],
                    mode="bilinear",
                    align_corners=True,
                )
            )
            original_step_size = [
                original_step_size[0] * upsampling_list[i],
                original_step_size[1] * upsampling_list[i],
            ]
            blocks.append(ConvBlock_2D(t_size=conv_size,
                                    n_step=original_step_size))
            blocks.append(
                IdentityBlock_2D(t_size=conv_size, n_step=original_step_size)
            )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        self.output_size_0 = original_step_size[0]
        self.output_size_1 = original_step_size[1]

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        out = self.dense(x)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)

        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        output = out.view(-1, self.output_size_0, self.output_size_1)

        return output


class Affine_Transform(nn.Module):
    def __init__(self,
                 device,
                 scale = True,
                 shear = True,
                 rotation = True,
                 translation = True,
                 Symmetric = True,
                 mask_intensity = True,
                 scale_limit = 0.05,
                 shear_limit = 0.1,
                 rotation_limit = 0.1,
                 trans_limit = 0.15,
                 adj_mask_para=0
                 ):
        super(Affine_Transform,self).__init__()
        self.scale = scale
        self.shear = shear
        self.rotation = rotation
        self.translation = translation
        self.Symmetric = Symmetric
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.rotation_limit = rotation_limit
        self.trans_limit = trans_limit
        self.adj_mask_para = adj_mask_para
        self.mask_intensity = mask_intensity
        self.device = device
        self.count = 0

    def forward(self,out,rotate_value = None):

        if self.scale:
            scale_1 = self.scale_limit*nn.Tanh()(out[:,self.count])+1
            scale_2 = self.scale_limit*nn.Tanh()(out[:,self.count+1])+1
            self.count +=2
        else:
            scale_1 = torch.ones([out.shape[0]]).to(self.device)
            scale_2 = torch.ones([out.shape[0]]).to(self.device)

        if self.rotation:
            if rotate_value!=None:
                # use large mask no need to limit to too small range
                rotate = rotate_value.reshape(out[:,self.count].shape) + self.rotation_limit*nn.Tanh()(out[:,self.count])
                self.count+=1
            else:
                rotate = nn.ReLU()(out[:,self.count])
                self.count+=1
        else:
            rotate = torch.zeros([out.shape[0]]).to(self.device)

        if self.shear:
            if self.Symmetric:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = shear_1
                self.count+=1
            else:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = self.shear_limit*nn.Tanh()(out[:,self.count+1])
                self.count+=2
        else:
            shear_1 = torch.zeros([out.shape[0]]).to(self.device)
            shear_2 = torch.zeros([out.shape[0]]).to(self.device)
        # usually the 4d-stem has symetric shear value, we make xy=yx, that's the reason we don't need shear2

        if self.translation:
            trans_1 = self.trans_limit*nn.Tanh()(out[:,self.count])
            trans_2 = self.trans_limit*nn.Tanh()(out[:,self.count+1])
            self.count +=2
        else:
            trans_1 = torch.zeros([out.shape[0]]).to(self.device)
            trans_2 = torch.zeros([out.shape[0]]).to(self.device)
  
        if self.mask_intensity:
            mask_parameter = self.adj_mask_para*nn.Tanh()(out[:,self.embedding_size:self.embedding_size+1])+1
        else:
            # this project doesn't need mask parameter to adjust value intensity in mask region, so we make it 1 here.
            mask_parameter = torch.ones([out.shape[0]])

        self.count = 0
 
        a_1 = torch.cos(rotate)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones([out.shape[0]]).to(self.device)
        a_5 = torch.zeros([out.shape[0]]).to(self.device)

       # combine shear and strain together
        c1 = torch.stack((scale_1,shear_1), dim=1).squeeze()
        c2 = torch.stack((shear_2,scale_2), dim=1).squeeze()
        c3 = torch.stack((a_5,a_5), dim=1).squeeze()
        scaler_shear = torch.stack((c1, c2, c3), dim=2)

        # Add the rotation after the shear and strain
        b1 = torch.stack((a_1,a_2), dim=1).squeeze()
        b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
        b3 = torch.stack((a_5,a_5), dim=1).squeeze()
        rotation = torch.stack((b1, b2, b3), dim=2)

        d1 = torch.stack((a_4,a_5), dim=1).squeeze()
        d2 = torch.stack((a_5,a_4), dim=1).squeeze()
        d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
        translation = torch.stack((d1, d2, d3), dim=2)

        return scaler_shear, rotation, translation, mask_parameter

## TODO: adjust transformer class so we can apply transforms during decoding
class Transformer(nn.Module):
    def __init__(self,device):
        """AutoEncoder model

        Args:
            enc (nn.Module): Encoder block
            dec (nn.Module): Decoder block
        """
        super().__init__()
        self.device = device

    
    def forward(self, predicted, scaler_shear, rotation, translation):
        """Forward pass of the autoencoder applies and affine grid to the decoder value

        Args:
            x (Tensor): Input (training data)

        Returns:
            embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
        """
        # predicted,affines = x
        # scaler_shear,rotation,translation = affines
        
        # sample and apply affine grids
        size_grid = torch.ones([predicted.shape[0], 1, 
                                predicted.shape[-2], 
                                predicted.shape[-1]])

        grid_1 = F.affine_grid(scaler_shear.to(self.device), 
                                size_grid.size()).to(self.device) # scale shear
        grid_2 = F.affine_grid(rotation.to(self.device), 
                                size_grid.size()).to(self.device) # rotation
        grid_3 = F.affine_grid(translation.to(self.device), 
                                size_grid.size()).to(self.device) # translation
        
        predicted = F.grid_sample(predicted,grid_3) # translation first to center
        predicted = F.grid_sample(predicted,grid_1)
        predicted = F.grid_sample(predicted,grid_2)

        return predicted
    pass
        
        
class AutoEncoder(nn.Module):
    def __init__(self, 
                 enc_1D, enc_2D,
                 dec_1D, dec_2D,
                 stacked_size, emb_size,
                 device,
                 get_embeddings=False,
                 training=True,
                 generator_mode='affine'):
        """AutoEncoder model

        Args:
            enc (nn.Module): Encoder block
            dec (nn.Module): Decoder block
            emb_size ():
            device():
            training (bool): If we are training, return embeddings for writing. If generating, return predicted only
            mode (str): {'affine','no_affine'} whether or not you want to apply affine transform at end
        """
        super().__init__()
        self.device = device
        self.enc_1D = enc_1D
        self.enc_2D = enc_2D
        self.dec_1D = dec_1D
        self.dec_2D = dec_2D
        self.temp_affines = None # for filling embedding info
        self.get_embeddings=get_embeddings
        self.training = training
        self.generator_mode = generator_mode
        self.dense = nn.Linear(stacked_size,emb_size)
        self.relu = nn.ReLU()
        self.temp_affines=None
        self.transformer = Transformer(self.device)
        
        @property
        def get_embeddings(self):
            return self._compute_embeddings

        @get_embeddings.setter
        def get_embeddings(self, value):
            self._compute_embeddings = value

    def forward(self, x):
        """Forward pass of the autoencoder applies and affine grid to the decoder value

        Args:
            x (tuple of Tensor): (diffraction, spectrum) (training data)

       Returns:
            embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
        """
        diff, spec = x
        
        embedding_1D = self.enc_1D(spec) # (batchsize, emb1dsize)
        embedding_2D,scaler_shear,rotation,translation = self.enc_2D(diff) # (batchsize, emb2dsize)
        embedding = torch.cat((embedding_1D,embedding_2D), 1) # (batchsize, emb1dsize + emb2dsize)
        embedding = self.dense(embedding)
        embedding = self.relu(embedding)
        if self.get_embeddings: 
            return embedding,scaler_shear,rotation,translation
        
        predicted_1D = self.dec_1D(embedding)
        predicted_2D = self.dec_2D(embedding).unsqueeze(1)
        self.temp_affines = scaler_shear,rotation,translation # for writing to embedding during training
        
        # if we do not apply affine transforms, return either the embedding or predicted, 
        # based on whether we are writing to embedding file directly during training
        if self.generator_mode=='no_affine':
            if self.training: # writing embedding in real time while training
                return embedding, predicted_1D, predicted_2D # .squeeze() ?
            if not self.training:
                return predicted_1D, predicted_2D

# ### make Transform class? ##TODO: figure out how to do this if we use kwargs
#         # sample and apply affine grids
#         size_grid = torch.ones([predicted_2D.shape[0],1,predicted_2D.shape[-2],predicted_2D.shape[-1]])

#         grid_1 = F.affine_grid(scaler_shear.to(self.device), size_grid.size()).to(self.device) # scale shear
#         grid_2 = F.affine_grid(rotation.to(self.device), size_grid.size()).to(self.device) # rotation
#         grid_3 = F.affine_grid(translation.to(self.device), size_grid.size()).to(self.device) # translation

#         predicted_2D = F.grid_sample(predicted_2D,grid_3) # translation first (ideally to center)
#         predicted_2D = F.grid_sample(predicted_2D,grid_1) # scale shear
#         predicted_2D = F.grid_sample(predicted_2D,grid_2) # rotation
        
        predicted_2D = self.transformer(predicted_2D,scaler_shear,rotation,translation)
### 
        if self.generator_mode=='affine':
            if self.training: 
                return embedding, predicted_1D, predicted_2D, scaler_shear,rotation,translation
            if not self.training: 
                return predicted_1D, predicted_2D 
        
        assert False, 'set train to True/False, and generator_mode to affine/no_affine'

     
class AutoEncoder_1D(nn.Module):
    def __init__(self, enc_1D, dec_1D,
                 device, get_embeddings=False,
                 training=True ):
        """AutoEncoder model

        Args:
            enc (nn.Module): Encoder block
            dec (nn.Module): Decoder block
            emb_size ():
            device():
            training (bool): If we are training, return embeddings for writing. If generating, return predicted only
            mode (str): {'affine','no_affine'} whether or not you want to apply affine transform at end
        """
        super().__init__()
        self.device = device
        self.enc_1D = enc_1D
        self.dec_1D = dec_1D
        self.get_embeddings=get_embeddings
        self.training = training
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the autoencoder applies and affine grid to the decoder value

        Args:
            x (tuple of Tensor): (diffraction, spectrum) (training data)

       Returns:
            embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
        """
        
        embedding = self.enc_1D(x) # (batchsize, emb1dsize)
        embedding = self.relu(embedding)
        if self.get_embeddings: 
            return embedding
        
        predicted_1D = self.dec_1D(embedding)
        
        # if we do not apply affine transforms, return either the embedding or predicted, 
        # based on whether we are writing to embedding file directly during training
        if self.training: # writing embedding in real time while training
            return embedding,predicted_1D
        if not self.training:
            return predicted_1D


class AutoEncoder_2D(nn.Module):
    def __init__(self, enc_2D, dec_2D,
                 emb_size, device,
                 get_embeddings=False,
                 training=True,
                 generator_mode='affine'):
        """AutoEncoder model

        Args:
            enc (nn.Module): Encoder block
            dec (nn.Module): Decoder block
            emb_size ():
            device():
            training (bool): If we are training, return embeddings for writing. If generating, return predicted only
            mode (str): {'affine','no_affine'} whether or not you want to apply affine transform at end
        """
        super().__init__()
        self.device = device
        self.enc_2D = enc_2D
        self.dec_2D = dec_2D
        self.temp_affines = None # for filling embedding info
        self.get_embeddings=get_embeddings
        self.training = training
        self.generator_mode = generator_mode
        self.relu = nn.ReLU()
        self.temp_affines=None
        self.transformer = Transformer(self.device)
        
        @property
        def get_embeddings(self):
            return self._compute_embeddings

        @get_embeddings.setter
        def get_embeddings(self, value):
            self._compute_embeddings = value

    def forward(self, x):
        """Forward pass of the autoencoder applies and affine grid to the decoder value

        Args:
            x (tuple of Tensor): (diffraction, spectrum) (training data)

       Returns:
            embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
        """
        diff = x
        
        embedding,scaler_shear,rotation,translation = self.enc_2D(diff) # (batchsize, emb2dsize)
        embedding = self.relu(embedding)
        if self.get_embeddings: 
            return embedding,scaler_shear,rotation,translation
        
        predicted_2D = self.dec_2D(embedding).unsqueeze(1)
        self.temp_affines = scaler_shear,rotation,translation # for writing to embedding during training
        
        # if we do not apply affine transforms, return either the embedding or predicted, 
        # based on whether we are writing to embedding file directly during training
        if self.generator_mode=='no_affine':
            if self.training: # writing embedding in real time while training
                return embedding, predicted_2D # .squeeze() ?
            if not self.training:
                return predicted_2D

# ### make Transform class? ##TODO: figure out how to do this if we use kwargs
#         # sample and apply affine grids
#         size_grid = torch.ones([predicted_2D.shape[0],1,predicted_2D.shape[-2],predicted_2D.shape[-1]])

#         grid_1 = F.affine_grid(scaler_shear.to(self.device), size_grid.size()).to(self.device) # scale shear
#         grid_2 = F.affine_grid(rotation.to(self.device), size_grid.size()).to(self.device) # rotation
#         grid_3 = F.affine_grid(translation.to(self.device), size_grid.size()).to(self.device) # translation

#         predicted_2D = F.grid_sample(predicted_2D,grid_3) # translation first (ideally to center)
#         predicted_2D = F.grid_sample(predicted_2D,grid_1) # scale shear
#         predicted_2D = F.grid_sample(predicted_2D,grid_2) # rotation
        
        predicted_2D = self.transformer(predicted_2D,scaler_shear,rotation,translation)
### 
        if self.generator_mode=='affine':
            if self.training: 
                return embedding, predicted_2D, scaler_shear,rotation,translation
            if not self.training: 
                return predicted_2D 
        
        assert False, 'set train to True/False, and generator_mode to affine/no_affine'


def db_show_im(tensor):
    import matplotlib.pyplot as plt
    plt.imshow(tensor.squeeze().detach().cpu())
    plt.show()