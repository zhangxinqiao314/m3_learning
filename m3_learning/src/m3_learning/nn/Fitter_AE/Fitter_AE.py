import torch
import torch.nn as nn
import torch.nn.functional as F

from ..STEM_AE.STEM_AE import ConvBlock, IdentityBlock

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

    def __init__(self, original_step_size, pooling_list, embedding_size, conv_size):
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
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
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
                out1 = torch.flatten(out1, start_dim=1)
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = torch.cat((out, out1), dim=1)
        out = self.dense(out)
        selection = self.relu_1(out)
        return selection


class PV_AE(nn.Module):
    """AE block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, encoder, fitter_function, 
                 ):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(PV_AE, self).__init__(original_step_size, pooling_list, embedding_size, conv_size)


