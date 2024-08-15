from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from m3_learning.viz.layout import layout_fig, imagemap, labelfigs, find_nearest, add_scalebar
from os.path import join as pjoin
from m3_learning.viz.nn import embeddings as embeddings_
from m3_learning.viz.nn import affines as affines_
import glob
import os
import h5py
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.image as mpimg
import panel as pn


# TODO: make interactive clickable visualization using datashader, holoview, and Panel
## TODO: get original Viz in here too
class Viz_Multimodal:

    """Visualization class for the STEM_AE class
    """

    def __init__(self,
                 channels=None,
                 color_map='viridis',
                 printer=None,
                 labelfigs_=False,
                 scalebar_=None,
                 ):
        """Initialization of the Viz class
        """

        self.printer = printer
        self.labelfigs_ = labelfigs_
        self.scalebar_ = scalebar_
        self.cmap = plt.get_cmap(color_map)
        self.channels = channels

    @property
    def model(self):
        """model getter

        Returns:
            obj: neural network model
        """
        return self._model

    @model.setter
    def model(self, model):
        """Model setter

        Args:
            model (object): neural network model
        """
        self._model = model

    @property
    def channels(self):
        """channel that are visualized getter

        Returns:
            list: channels that are visualized
        """

        return self._channels

    @channels.setter
    def channels(self, channels):
        """channel that are visualized setter

        Args:
            channels (list): channels that are visualized
        """

        if channels == None:
            # if none is given, all channels are visualized
            try:
                self._channels = range(self.model.embedding.shape[1])
            except:
                self._channels = None
        else:
            self._channels = channels

    def STEM_raw_and_virtual(self, dataset,ind,
                             bright_field_=True,
                             dark_field_=True,
                             scalebar_=True,
                             save_folder='bf_df_spec',
                             **kwargs):
        """visualizes the raw STEM data and the virtual STEM data

        Args:
            data (np.array): raw data to visualize
            bright_field_ (list, optional): bounding box for the bright field diffraction spot. Defaults to None.
            dark_field_ (list, optional): bounding box for the dark field diffraction spot. Defaults to None.
            scalebar_ (bool, optional): determines if the scalebar is shown. Defaults to True.
            filename (string, optional): Name of the file to save. Defaults to None.
            shape_ (list, optional): shape of the original data structure. Defaults to [265, 256, 256, 256].
        """
        # ind = dataset.meta['particle_list'].where(particle)
        with h5py.File(dataset.h5_name,'a') as h:
            diff = h['processed_data/diff'][dataset.meta['particle_inds'][ind]:
                                    dataset.meta['particle_inds'][ind+1]]
            
            eels = h['processed_data/eels'][dataset.meta['particle_inds'][ind]:
                                    dataset.meta['particle_inds'][ind+1]]
            
            # sets the number of figures based on how many plots are shown
            fig_num = 1
            if bright_field_:
                fig_num += 1
            if dark_field_:
                fig_num += 1

            # Make grid
            fig = plt.figure(figsize=(9,9))
            gs = fig.add_gridspec(3, 3)
            axs = [] ##TODO: make plots flexible for diff number of bf/df images
            axs.append( fig.add_subplot(gs[0,0]) ) # particle 0
            axs.append( fig.add_subplot(gs[0,1]) ) # bf 1
            axs.append( fig.add_subplot(gs[0,2]) ) # df 2
            axs.append( fig.add_subplot(gs[1,0]) ) # ll 3
            axs.append( fig.add_subplot(gs[1,1:]) ) # ll 4
            axs.append( fig.add_subplot(gs[2,0]) ) # hl 5
            axs.append( fig.add_subplot(gs[2,1:]) ) # hl 6
            
            # # creates the figure
            # fig, axs = layout_fig(fig_num, fig_num, figsize=(
            #                         1.5*fig_num, 1.25))
            a=0
            # plots the raw STEM data
            imagemap(axs[a], np.mean(diff, axis=(0,1))*dataset.BF_mask[ind], 
                     divider_=False)
            a+=1

            # plots the virtual bright field image
            if bright_field_ is not None:
                bright_field = diff[:,:,dataset.BF_inds[ind][0],
                                      dataset.BF_inds[ind][1]]
                bright_field = bright_field.mean(axis=(1,2))
                bright_field = bright_field.reshape(dataset.meta['shape_list'][ind][0][0],
                                                    dataset.meta['shape_list'][ind][0][1])
                imagemap(axs[a], bright_field, divider_=False)
                a+=1

            # plots the virtual dark field image
            if dark_field_ is not None:
                dark_field = diff
                dark_field[:,:,dataset.BF_inds[ind][0],
                                dataset.BF_inds[ind][1]] = 0
                                
                dark_field = np.mean(dark_field,axis=(2,3))
                dark_field = dark_field.reshape(dataset.meta['shape_list'][ind][0][0],
                                                dataset.meta['shape_list'][ind][0][1])
                imagemap(axs[a], dark_field, divider_=False)
                a+=1
                
            for i in range(eels.shape[1]):
                s = dataset.meta['shape_list'][ind][1]
                data =  eels[:,i]
                
                # plots image
                mean_img = np.mean(data,axis=(-1)).reshape(s[0],s[1])
                imagemap(axs[a], mean_img, divider_=False)
                a+=1
                
                # plots the spectrum
                mean_spec = np.mean(data,axis=(0))
                spec_inds = dataset.meta['eels_axis_labels'][i]
                x = dataset.get_raw_spectral_axis()
                axs[a].plot(x[spec_inds[2]][spec_inds[0]-1:spec_inds[1]], 
                            mean_spec - dataset.bkgs[ind][i])
                a+=1
                
            # adds labels to the figure
            if self.labelfigs_:
                for j, ax in enumerate(axs):
                    labelfigs(ax, j)

            if scalebar_:
                # adds a scalebar to the figure
                add_scalebar(axs[2], self.scalebar_)

            make_folder(f'{self.printer.basepath}{save_folder}/')
            # saves the figure
            if self.printer is not None:
                self.printer.savefig(fig, f'{save_folder}/{dataset.meta["particle_list"][ind]}', tight_layout=True)

    def find_nearest(self, array, value, averaging_number):
        """Finds the nearest value in an array

        This is useful when generating data from the embedding space.

        Args:
            array (array): embedding values
            value (array): current value
            averaging_number (int): how many spectra to use for averaging in the embedding space

        Returns:
            list : list of indexes to use for averaging
        """

        idx = (np.abs(array-value)).argsort()[0:averaging_number]
        return idx

    def predictor(self, values):
        """Computes the forward pass of the autoencoder

        Args:
            values (array): input values to predict

        Returns:
            array: predicted output values
        """
        with torch.no_grad():
            values = torch.from_numpy(np.atleast_2d(values))
            values = self.model(values.float())
            values = values.detach().numpy()
            return values
        
    def embeddings(self,meta,savefolder,overwrite=False, **kwargs):
        """function to plot the embeddings of the data
        """        
        # if savefolder
        for i,p_name in enumerate(tqdm(meta['particle_list'])): # each sample
            if not overwrite: 
                existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
                if '*'+p_name+'_embedding_maps.png' in existing:
                    print('skipping',savefolder+'*'+p_name+'_embedding_maps.png')
                    continue

            with self.model.open_embedding_h() as h:
                data=self.model.embedding[ meta['particle_inds'][i]:meta['particle_inds'][i+1]]
                
            make_folder(self.printer.basepath+savefolder)
        
            embeddings_(data, 
                    channels=self.channels[:self.model.embedding_size], 
                    labelfigs_ = self.labelfigs_,
                    printer = self.printer,
                    shape_=meta['shape_list'][i][0],
                    name = savefolder+p_name,
                    figsize=(5, 6),
                    **kwargs)
            
            plt.clf();
            plt.close();
        
    def affines(self,meta,savefolder,overwrite=False, **kwargs):
        """function to plot the embeddings of the data
        """        
        # if savefolder
        for i,p_name in enumerate(tqdm(meta['particle_list'])): # each sample
            if overwrite: 
                existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
                if p_name+'_affine_maps.png' in existing:
                    print('skipping',savefolder+p_name+'_embedding_maps.png')
                    continue
                
            with self.model.open_embedding_h() as h:
                affines_data=(self.model.scale_shear[ meta['particle_inds'][i]:\
                                            meta['particle_inds'][i+1]],
                            self.model.scale_shear[ meta['particle_inds'][i]:\
                                            meta['particle_inds'][i+1]],
                            self.model.scale_shear[ meta['particle_inds'][i]:\
                                            meta['particle_inds'][i+1]] )
                
            make_folder(self.printer.basepath+savefolder)
            affines_(affines_data, 
                    labelfigs_ = self.labelfigs_,
                    printer = self.printer,
                    shape_=meta['shape_list'][i][0],
                    name = savefolder+p_name,
                    # clim=(0,data.max()),
                    **kwargs)
            plt.clf();
            plt.close();
            
    def clustered_images(self, dset, cluster_list, labels,
                         save_folder = './',
                         labelfigs_ = False,
                         scalebar_ = None,
                         printer = None,
                         **kwargs):
        
        # colors = iter([plt.cm.tab20(i) for i in range(20)])
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'pink', 'gray', 'olive', 'cyan', 'navy', 'teal', 'maroon', 'silver', 'tan', 'gold', 'purple', 'moccasin', 'bisque', 'wheat', 'peachpuff', 'navajowhite', 'salmon', 'crimson', 'palevioletred', 'darksalmon', 'lightcoral', 'hotpink', 'palegoldenrod', 'plum', 'darkkhaki', 'orchid', 'thistle', 'lightgray', 'lightgreen', 'lightblue', 'lightskyblue', 'lightyellow', 'lavender', 'linen']
        n_clusters = len(np.unique(labels))
        colors = colors[:n_clusters]
        self.cmap = ListedColormap(colors,'segmented',N=n_clusters)
        
        for i,particle in enumerate(tqdm(dset.meta['particle_list'])):
            # builds the figure
            fig, axs = layout_fig(1, **kwargs)

            # loops around the channels to plot
            imagemap(axs[0], cluster_list[i],clim=(0,n_clusters-1),colormap=self.cmap,
                     cbar_number_format='%2.0e', divider_=False, **kwargs)

            # adds labels to the figure
            if labelfigs_:
                for i, ax in enumerate(axs):
                    labelfigs(ax, i)

            # adds the scalebar
            if scalebar_ is not None:
                add_scalebar(axs.flatten()[-1], scalebar_)
            
            # prints the image
            if printer is not None:
                make_folder(self.printer.basepath+save_folder)
                self.printer.savefig(fig,
                    f'./{save_folder}/{particle}', tight_layout=False)
                
            plt.close(fig)
            
            # plt.clf()
            # plt.imshow(cluster_list[i],vmin=0,vmax=19,cmap='tab20')
            # cbar = plt.colorbar()
            # cbar.set_ticks(np.arange(0,20))
            # cbar.set_ticklabels(np.arange(20))
            # plt.title(particle)
            # plt.savefig(f'./clustering_emb_affine_epoch_13/{particle}',facecolor='white',dpi=200)
             
    def generator_images(self,meta,generated,
                         embedding=None,
                         folder_name='',
                         ranges=None,
                         graph_layout=[2, 2],
                         shape_=[256, 256, 128, 128],
                         clim=(0.0,1.0),
                         scaler=None,
                         overwrite=False,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            self.channels = kwargs["channels"]

        he = self.model.open_embedding_h()
        hg = self.model.open_generated_h()

        check = self.model.checkpoint.split('/')[-1][:-4]
        self.model.generated = hg[check]
        self.model.embedding = he[f'embedding_{check}']
        num_particles,generator_iters,emb_ch,_,_ = self.model.generated.shape

        # # gets the embedding if a specific embedding is not provided
        # if embedding is None: embedding = self.model.embedding
        make_folder(f'{self.printer.basepath}{folder_name}/')

        for p,p_name in enumerate(meta['particle_list']): # each sample
            existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{folder_name}/{p_name}/*')]
            
            print(p, p_name)
            emb = self.model.embedding[ meta['particle_inds'][p]:\
                                        meta['particle_inds'][p+1]]
            make_folder(f'{self.printer.basepath}{folder_name}/{p_name}/')

            # loops around the number of iterations to generate
            for i in tqdm(range(generator_iters)):
                if not overwrite:
                    if f'{i:04d}_maps.png' in existing:
                        print('skipping',f'{folder_name}/{p_name}/{i:04d}_maps')
                        continue

                # builds the figure
                fig, ax = layout_fig(graph_layout[0], graph_layout[1], **kwargs)
                ax = ax.reshape(-1)
                data = self.model.generated[p][i]

                try:
                    data=scaler.inverse_transform(
                        data.reshape(-1,128*128) ).reshape(-1,128,128)
                except: pass

                # loops around all of the embeddings
                for j, channel in enumerate(self.channels):

                    if ranges is None: # span this range when generating
                        ranges = np.stack((np.min(data, axis=(1,2)),
                                        np.max(data, axis=(1,2))), axis=1)
                
                    # plot the generated image
                    imagemap(ax[j], data[j], **kwargs)
                    
                    pt = int(shape_[-1]*0.1)
                    ax[j].plot(shape_[-1]-pt-1,pt, marker='o', markeredgewidth=0.0, markersize=3,
                               markerfacecolor=self.cmap((i+1)/generator_iters))

                    axes_in = ax[j].inset_axes([0.55, 0.02, 0.43, 0.43])

                    # plots the embedding and formats embedding
                    imagemap(axes_in, 
                             emb[:,channel].reshape(meta['shape_list'][p][-4], meta['shape_list'][p][-3]), 
                             colorbars=False)
                    
                    # adds labels to the figure
                    if self.labelfigs_: labelfigs(ax[j], j, size=4, text_pos="center")

                if self.printer is not None:
                    self.printer.savefig(fig,
                                    f'{folder_name}/{p_name}/{i:04d}_maps', 
                                    tight_layout=False, 
                                    )
                plt.close(fig);
        he.close()
        hg.close()
              
    def generator_images_1D(self,dset,
                         folder_name='',
                         ranges=None,
                         graph_layout=[2, 2],
                         shape_=[256, 256, 128, 128],
                         clim=(0.0,1.0),
                         scaler=None,
                         overwrite=True,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            self.channels = kwargs["channels"]

        # he = self.model.open_embedding_h()
        # hg = self.model.open_generated_h()

        check = self.model.checkpoint.split('/')[-1][:-4]
        # self.model.generated = hg[check]
        # self.model.embedding = he[f'embedding_{check}']

        # # gets the embedding if a specific embedding is not provided
        # if embedding is None: embedding = self.model.embedding
        make_folder(f'{self.printer.basepath}{folder_name}/')

        for p,p_name in enumerate(dset.meta['particle_list']): # each sample
            existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{folder_name}/{p_name}/*')]
            
            print(p, p_name)
            with self.model.open_embedding_h() as he:
                emb = self.model.embedding[ dset.meta['particle_inds'][p]:\
                                            dset.meta['particle_inds'][p+1]]
            make_folder(f'{self.printer.basepath}{folder_name}/{p_name}/')
            
            with self.model.open_generated_h() as hg:            
                num_particles,generator_iters,emb_ch,spec_ch,spec_len = self.model.generated.shape

                # loops around the number of iterations to generate
                for i in tqdm(range(generator_iters)):
                    if not overwrite:
                        if f'{i:04d}_maps.png' in existing:
                            print('skipping',f'{folder_name}/{p_name}/{i:04d}_maps')
                            continue

                    # builds the figure
                    fig, ax = layout_fig(graph_layout[0], graph_layout[1], **kwargs)
                    ax = ax.reshape(-1)
                    
                    data = self.model.generated[p][i]

                    try:
                        data=dset.scaler.inverse_transform(
                            data.reshape(-1,x*y) ).reshape(-1,x,y)
                    except: pass

                    # loops around all of the embeddings
                    for j, channel in enumerate(self.channels):

                        if ranges is None: # span this range when generating
                            ranges = np.stack((np.min(data, axis=(1,2)),
                                            np.max(data, axis=(1,2))), axis=1)
                    
                        # plot each spec channel
                        for c in range(dset.eels_chs):
                            color=self.cmap((i+1)/generator_iters)
                            ind = dset.eels_chs*j + c
                            spec_i = dset.meta['eels_axis_labels'][c]
                            x_labels = dset.raw_x_labels[spec_i[0]:spec_i[1]+1,spec_i[2]]
                            ax[ind].plot(x_labels, data[j,c],color=color)
                            

                        axes_in = ax[dset.eels_chs*j].inset_axes([0.55, 0.55, 0.43, 0.43])

                        # plots the embedding and formats embedding
                        imagemap(axes_in, 
                                emb[:,channel].reshape(dset.meta['shape_list'][p][0][-4], 
                                                        dset.meta['shape_list'][p][0][-3]), 
                                colorbars=False)
                        
                        # adds labels to the figure
                        if self.labelfigs_: labelfigs(ax[j], j, size=4, text_pos="center")

                    if self.printer is not None:
                        self.printer.savefig(fig,
                                        f'{folder_name}/{p_name}/{i:04d}_maps', 
                                        tight_layout=False, 
                                        )
                    plt.close(fig);
        # he.close()
        # hg.close()
              
    def generator_images_2D(self,dset,
                         folder_name='',
                         ranges=None,
                         graph_layout=[2, 2],
                         shape_=[256, 256, 128, 128],
                         clim=(0.0,1.0),
                         scaler=None,
                         overwrite=True,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        # sets the channels to use in the object
        if "channels" in kwargs:
            self.channels = kwargs["channels"]

        # he = self.model.open_embedding_h()
        # hg = self.model.open_generated_h()

        check = self.model.checkpoint.split('/')[-1][:-4]
        # self.model.generated = hg[check]
        # self.model.embedding = he[f'embedding_{check}']

        # # gets the embedding if a specific embedding is not provided
        # if embedding is None: embedding = self.model.embedding
        make_folder(f'{self.printer.basepath}{folder_name}/')

        for p,p_name in enumerate(dset.meta['particle_list']): # each sample
            existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{folder_name}/{p_name}/*')]
            
            print(p, p_name)
            with self.model.open_embedding_h() as he:
                emb = self.model.embedding[ dset.meta['particle_inds'][p]:\
                                            dset.meta['particle_inds'][p+1]]
            make_folder(f'{self.printer.basepath}{folder_name}/{p_name}/')
            
            with self.model.open_generated_h() as hg:            
                num_particles,generator_iters,emb_ch,x,y = self.model.generated.shape

                # loops around the number of iterations to generate
                for i in tqdm(range(generator_iters)):
                    if not overwrite:
                        if f'{i:04d}_maps.png' in existing:
                            print('skipping',f'{folder_name}/{p_name}/{i:04d}_maps')
                            continue

                    # builds the figure
                    fig, ax = layout_fig(graph_layout[0], graph_layout[1], **kwargs)
                    ax = ax.reshape(-1)
                    
                    data = self.model.generated[p][i]

                    try:
                        data=dset.scaler.inverse_transform(
                            data.reshape(-1,x*y) ).reshape(-1,x,y)
                    except: pass

                    # loops around all of the embeddings
                    for j, channel in enumerate(self.channels):

                        if ranges is None: # span this range when generating
                            ranges = np.stack((np.min(data, axis=(1,2)),
                                            np.max(data, axis=(1,2))), axis=1)
                    
                        # plot the generated image
                        imagemap(ax[j], data[j], **kwargs)
                        
                        pt = int(shape_[-1]*0.1)
                        ax[j].plot(shape_[-1]-pt-1,pt, marker='o', markeredgewidth=0.0, markersize=3,
                                markerfacecolor=self.cmap((i+1)/generator_iters))

                        axes_in = ax[j].inset_axes([0.55, 0.02, 0.43, 0.43])

                        # plots the embedding and formats embedding
                        imagemap(axes_in, 
                                emb[:,channel].reshape(dset.meta['shape_list'][p][0][-4], 
                                                        dset.meta['shape_list'][p][0][-3]), 
                                colorbars=False)
                        
                        # adds labels to the figure
                        if self.labelfigs_: labelfigs(ax[j], j, size=4, text_pos="center")

                    if self.printer is not None:
                        self.printer.savefig(fig,
                                        f'{folder_name}/{p_name}/{i:04d}_maps', 
                                        tight_layout=False, 
                                        )
                    plt.close(fig);
        # he.close()
        # hg.close()
        
    def fits_Fitter1D(self,model,dset,savefolder='embeddings',overwrite=True,
                   name="",
                   channels = None,
                   labelfigs_ = False,
                   scalebar_ = None,
                   printer = None,):
        # TODO: put parameter labels (a_g, a_l, x, sig, gam, nu)
        # TODO: display orig spectrum
        meta = {}
        with h5py.File(dset.h5_name, 'r+') as h5:
            for key, value in h5[f'{dset.mode[0]}'].attrs.items():
                meta[key] = value
        chnm = ['HL: ','LL: ']    
        nchs = dset.eels_chs
        nfits = model.num_fits
        x_vals = [dset.raw_x_labels[i0-1:i1+1,l] for i0,i1,l in dset.meta['eels_axis_labels']]
        for p,p_name in enumerate(tqdm(dset.meta['particle_list'])): # each sample
            x,y,_ = meta['shape_list'][p]
            if not overwrite: 
                existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
                if '*'+p_name+'_embedding_maps.png' in existing:
                    print('skipping',savefolder+'*'+p_name+'_embedding_maps.png')
                    continue

            with model.open_embedding_h() as h:
                check = model.checkpoint.split('/')[-1][:-4]
                emb = h[f'embedding_{check}'][ meta['particle_inds'][p]:meta['particle_inds'][p+1]]
                fits = h[f'fits_{check}'][ meta['particle_inds'][p]:meta['particle_inds'][p+1]]
                
            make_folder(self.printer.basepath+savefolder)
            
            # sets the channels to use in the object (emb channels, not orig channels)
            if channels is None:
                channels = range(nfits)

            # builds the figure
            fig = plt.figure(figsize=(6*nchs,len(channels)*2))
            r,c = nfits,3*nchs
            gs = gridspec.GridSpec(r,c) 
            axs=[]
            
            for i,c in enumerate(channels):
                for j in range(nchs):
                    axs.append(plt.subplot(gs[i,j*3]))
                    imagemap(axs[-1],fits[:,j,c].mean(axis=1).reshape((x,y)))
                    
                    axs.append(plt.subplot(gs[i,j*3+1:j*3+3]))
                    axs[-1].plot(x_vals[j], fits[:,j,c].mean(axis=0))
                    title_text = (r'$A_g$:{:.1e},  $A_l$:{:.1e},  x:{:.1e},  $\sigma$:{:.1e},  $\gamma$:{:.1e},  $\nu$:{:.1e}'.format(
                        emb[:, j, c, 0].mean(0),
                        emb[:, j, c, 1].mean(0),
                        emb[:, j, c, 2].mean(0),
                        emb[:, j, c, 3].mean(0),
                        emb[:, j, c, 4].mean(0),
                        emb[:, j, c, 5].mean(0)
                    ))
                    axs[-1].set_title(chnm[j]+title_text,fontsize=10,loc='right')
            
            fig.suptitle(p_name)    
            
            # adds labels to the figure
            if labelfigs_:
                for i, ax in enumerate(axs):
                    labelfigs(ax, i)

            # adds the scalebar
            if scalebar_ is not None:
                add_scalebar(axs[-2], scalebar_)

            # prints the image
            if printer is not None:
                printer.savefig(fig,
                    f'{savefolder}/{p_name}_embedding_maps', tight_layout=True)
            
            # plt.close(fig)
            # plt.clf();
            # plt.close();        
            
    def fits_Fitter1D_widget(self,p,e,a,b,w,model,dset,
                            savefolder='embeddings',
                            overwrite=True,
                            scalebar_ = None,
                            printer = None,
                            mean=False):
        """create images for viewing with ipywidget

        Args:
            model (_type_): _description_
            dset (_type_): _description_
            p (_type_): particle num
            i (_type_): embedding channel num
            j (_type_): eels channel num
            savefolder (str, optional): _description_. Defaults to 'embeddings'.
            overwrite (bool, optional): _description_. Defaults to True.
            name (str, optional): _description_. Defaults to "".
            channels (_type_, optional): _description_. Defaults to None.
            labelfigs_ (bool, optional): _description_. Defaults to False.
            scalebar_ (_type_, optional): _description_. Defaults to None.
            printer (_type_, optional): _description_. Defaults to None.
        """        
        # TODO: put parameter labels (a_g, a_l, x, sig, gam, nu)
        # TODO: display orig spectrum

        # existing = [item.split('/')[-1] for item in glob.glob(f'{self.printer.basepath}{savefolder}*')]
        # if f'{p}_{i}_embedding_maps.png' in existing:
        #     img = mpimg.imread('your_image.png')
        #     plt.imshow(img)
        #     plt.show()
        #     return 
        meta=dset.meta
        idx = dset.get_index(p,a,b)
        nchs=dset.eels_chs
        with model.open_embedding_h() as h:
            check = model.checkpoint.split('/')[-1][:-4]
            embs = h[f'embedding_{check}'][ meta['particle_inds'][p]:meta['particle_inds'][p+1]]
            emb = h[f'embedding_{check}'][idx]
            fits = h[f'fits_{check}'][ meta['particle_inds'][p]:meta['particle_inds'][p+1]]
            fit = h[f'fits_{check}'][idx]
        _,orig = dset[idx]
        
        x,y = meta['shape_list'][p]
        make_folder(self.printer.basepath+savefolder)

        # builds the figure
        fig = plt.figure(figsize=(model.num_params, 3*nchs))
        fig.suptitle(f"{dset.meta['particle_list'][p]}, emb channel {e}")
        
        rows,cols = model.num_params,3*nchs
        gs = gridspec.GridSpec(rows,cols) 
        axs=[]

        for j in range(nchs):
            plt.rc('text', usetex=True)
            
            # parameters
            par_labels = ['$A_g$', '$A_l$', 'x', '$\\sigma$', '$\\gamma$', '$\\nu$']
            for par in range(rows):
                axs.append(plt.subplot(gs[j*3,par]))
                imagemap(axs[-1],embs[:,j,e,par].reshape((x,y)),colorbars=False)
                axs[-1].scatter(a,b,s=5,c='r',marker='o')
                axs[-1].text(0.95, 0.95, f'{par_labels[par]}: {emb[j, e, par]:.1e}',
                            verticalalignment='top', horizontalalignment='right',
                            transform=axs[-1].transAxes,
                            color='white', fontsize=8)
                # axs[-1].set_title(f'{par_labels[par]}: {emb[j,e,par]:.1e}',fontsize=6)
    
            # real space img
            axs.append(plt.subplot(gs[j*3+1:j*3+3,:2])) 
            
            if mean: imagemap(axs[-1], fits[:,j,e].mean(axis=1).reshape((x,y)))
            else: imagemap(axs[-1], fits[:,j,e,w].reshape((x,y)))
            
            axs[-1].scatter(a,b,s=5,c='r',marker='o')
            axs[-1].text(0.95, 0.95, f'({a},{b})',
                            verticalalignment='top', horizontalalignment='right',
                            transform=axs[-1].transAxes,
                            color='white', fontsize=8)
            # axs[-1].set_title(f'({a},{b})',fontsize=8)
            
            # spectrum
            axs.append(plt.subplot(gs[j*3+1:j*3+3,2:]))
            axs[-1].tick_params(axis='both', which='both', direction='in')
            axs[-1].tick_params(axis='x', which='both', direction='in', pad=-10)  # Bottom axis labels
            axs[-1].tick_params(axis='y', which='both', direction='in', pad=-20)   # Left axis labels
            
            # axs[-1].plot(dset.meta['eels_axis_labels'][j], orig[j])
            if mean: axs[-1].plot(dset.meta['eels_axis_labels'][j], fits[:,j,e].mean(axis=0))
            else:  axs[-1].plot(dset.meta['eels_axis_labels'][j], fit[j,e])
            
            axs[-1].axvline(meta['eels_axis_labels'][j][w],color='r')
            axs[-1].text(0.95, 0.95, f'{meta["eels_axis_labels"][j][w]} eV',
                        verticalalignment='top', horizontalalignment='right',
                        transform=axs[-1].transAxes,
                        color='black', fontsize=8)
            # axs[-1].set_title(f'{meta["eels_axis_labels"][j][w]} eV',fontsize=6)
            # axs[-1].set_xlabel(f'loss (eV)',fontsize=6)
            
        fig.tight_layout(w_pad=-1,h_pad=-0.1)
        
        # adds the scalebar
        if scalebar_ is not None:
            add_scalebar(axs[0], scalebar_)

        # prints the image
        if printer is not None:
            printer.savefig(fig,
                f'{savefolder}/{p}_{e}_embedding_maps', tight_layout=False,verbose=True)
        
        return fig
        
    # TODO: use datashader    
    def interactive_fits_Fitter1d_widget(self, model, dset,
                                        savefolder='embeddings', overwrite=True,
                                        scalebar_=None, printer=None):
        """Wrapper to make fits_Fitter1D_widget interactive using Panel."""

        # Panel widgets
        p_dict = {p: i for i,p in enumerate(dset.meta['particle_list'])}
        p_selector = pn.widgets.Select(name='Particle Index', options=p_dict)
        e_slider = pn.widgets.IntSlider(name='Embedding Channel', start=0, end=model.num_fits - 1, value=0)
        w_slider = pn.widgets.IntSlider(name='Loss (eV)', start=0, end=len(dset.meta['eels_axis_labels'][0]) - 1, value=0)
        coord_text = pn.widgets.StaticText(name='Selected Coordinates', value='(a, b)')
        
        # Store coordinates as panel parameters
        coords = pn.widgets.StaticText(name='Coords', value=(0, 0))
        a_slider = pn.widgets.IntSlider(name='a', start=0, end=len(dset.meta['diff_dims'][0]) - 1, value=0)
        b_slider = pn.widgets.IntSlider(name='b', start=0, end=len(dset.meta['diff_dims'][1]) - 1, value=0)

        # # Interactive update function
        # def update_image(event=None):
        #     p = p_selector.value
        #     e = e_slider.value
        #     a, b = np.random.randint(0, dset.meta['shape_list'][p])  # Random start, user will select later
        #     w = w_slider.value

        #     # Call the original function with the selected parameters
        #     layout = self.fits_Fitter1D_widget(p, e, a, b, w, model, dset, savefolder, overwrite, scalebar_, printer)
        #     return layout

        # # Image selection callback
        # def on_select(event):
        #     nonlocal coord_text
        #     a, b = int(event.xdata), int(event.ydata)
            # coord_text.value = f'Selected Coordinates: ({a}, {b})'
            
            # p = p_selector.value
            # e = e_slider.value
            # w = w_slider.value
            # pn.state.curdoc().add_next_tick_callback(lambda: update_image())  # Redraw after selection


        def update_image(p, e, w, a,b):
            # a, b = coords
            return self.fits_Fitter1D_widget(p, e, a, b, w, model, dset, savefolder, overwrite, scalebar_, printer)
       
        # Bind the update_image function to the widgets
        interactive_image = pn.bind(update_image, p_selector, e_slider, w_slider, a_slider, b_slider)
        
        # Initial plot
        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        ax.imshow(np.zeros((100, 100)))  # Placeholder
        cid = fig.canvas.mpl_connect('button_press_event', on_select)

        # Layout for the Panel application
        widgets = pn.Row(p_selector, e_slider, w_slider, coord_text)
        interactive_layout = pn.Column(widgets, interactive_image).servable()

        return interactive_layout

        
def imshow_tensor(x):
    import matplotlib.pyplot as plt
    plt.imshow(x.detach().cpu().numpy());
    plt.colorbar()
    plt.show()