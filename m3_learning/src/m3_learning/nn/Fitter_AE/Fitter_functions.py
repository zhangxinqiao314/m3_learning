from os.path import join as pjoin
import torch
import torch.nn as nn

import warnings 
warnings.filterwarnings("ignore")

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/

# TODO: fix these to match the article
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
    gamma_x = torch.clamp(m*nn.Tanh()(embedding[:,:,0]) + m, min=0, max=limits[1])
    eta = (0.5*nn.Tanh()(embedding[:,:,2]) + 0.5)
    return amplitude,gamma_x, eta # look at limits after activations

def generate_pseudovoigt_1D(embedding, dset, limits=[1,975,25,1,25,1], device='cpu',return_params=False):
    '''embedding is A_g, x, sigma, A_l, gamma, nu. shape should be (batch*eels_ch, num_peaks, spec_len)'''
    
    a_g,mean_x,cov_x, = get_gaussian_parameters_1D(embedding, limits, dset.spec_len)
    a_l,gamma_x, eta = get_lorentzian_parameters_1D(embedding[:,:,-3:],limits[-2:], dset.spec_len)
    
    s = mean_x.shape
    x = torch.arange(dset.spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)
    
    # Gaussian component
    gaussian = a_g.unsqueeze(-1)* torch.exp(
        -0.5 * ((x-mean_x.unsqueeze(-1)) / cov_x.view(s[0],s[1],1))**2 )

    # Lorentzian component (simplified version)
    lorentzian = a_l.unsqueeze(-1) *(
            gamma_x.view(s[0],s[1],1)/ ((x-mean_x.unsqueeze(-1))**2+gamma_x.view(s[0],s[1],1)**2) )
    
    # Pseudo-Voigt profile
    pseudovoigt = eta.unsqueeze(-1) * lorentzian + \
                    (1 - eta.unsqueeze(-1)) * gaussian

    if return_params: return pseudovoigt.to(torch.float32), torch.stack([a_g,mean_x,cov_x,a_l,gamma_x,eta],axis=2)
    return pseudovoigt.to(torch.float32)


def generate_pseudovoigt_2D(embedding, out_shape, limits=[1, 10, 10, 10, 10, 0.5], device='cpu', return_params=False):
    '''embedding is: 
        A: Area under curve
        I_b: baseline intensity
        x: mean x of the distributions
        y: mean y of the distributions
        wx: x FWHM
        wy: y FWHM
        nu: lorentzian character fraction
        t: rotation angle
       
       shape should be (_, num_fits, x_, y_)'''
    
    A = limits[0] * nn.ReLU()(embedding[..., 0])
    x = torch.clamp(limits[1]/2 * nn.Tanh()(embedding[..., 1]) + limits[1]/2, min=1e-3)
    y = torch.clamp(limits[2]/2 * nn.Tanh()(embedding[..., 2]) + limits[2]/2, min=1e-3)
    wx = torch.clamp(limits[3]/2 * nn.Tanh()(embedding[..., 3]) + limits[3]/2, min=1e-3)
    wy = torch.clamp(limits[4]/2 * nn.Tanh()(embedding[..., 4]) + limits[4]/2, min=1e-3)
    nu = 0.5 * nn.Tanh()(embedding[..., 5]) + 0.5
    t = torch.pi / 2 * nn.Tanh()(embedding[..., 6])

    s = x.shape  # (_, num_fits)

    # Generate grid
    x_ = torch.arange(out_shape[0], dtype=torch.float32)
    y_ = torch.arange(out_shape[1], dtype=torch.float32)
    x_, y_ = torch.meshgrid(x_, y_)
    x_ = x_.repeat(s[0], s[1], 1, 1).to(device) 
    y_ = y_.repeat(s[0], s[1], 1, 1).to(device)

    # Apply rotation
    cos_t = torch.cos(t).unsqueeze(-1).unsqueeze(-1)
    sin_t = torch.sin(t).unsqueeze(-1).unsqueeze(-1)
    x_rot = cos_t * x_ + sin_t * y_
    y_rot = -sin_t * x_ + cos_t * y_

    # Compute 2D Gaussian component
    gauss_exp_x = (x_rot - x.unsqueeze(-1).unsqueeze(-1))**2 / (2 * wx.unsqueeze(-1).unsqueeze(-1)**2)
    gauss_exp_y = (y_rot - y.unsqueeze(-1).unsqueeze(-1))**2 / (2 * wy.unsqueeze(-1).unsqueeze(-1)**2)
    gaussian = A.unsqueeze(-1).unsqueeze(-1) * torch.exp(-(gauss_exp_x + gauss_exp_y))

    # Compute 2D Lorentzian component
    lorentzian_x = wx.unsqueeze(-1).unsqueeze(-1) / ((x_rot - x.unsqueeze(-1).unsqueeze(-1))**2 + wx.unsqueeze(-1).unsqueeze(-1)**2)
    lorentzian_y = wy.unsqueeze(-1).unsqueeze(-1) / ((y_rot - y.unsqueeze(-1).unsqueeze(-1))**2 + wy.unsqueeze(-1).unsqueeze(-1)**2)
    lorentzian = A.unsqueeze(-1).unsqueeze(-1) * lorentzian_x * lorentzian_y

    # Combine Gaussian and Lorentzian components
    pseudovoigt = nu.unsqueeze(-1).unsqueeze(-1) * lorentzian + (1 - nu.unsqueeze(-1).unsqueeze(-1)) * gaussian

    if not return_params:
        return pseudovoigt.to(torch.float32)

    params = torch.stack([A, x, y, wx, wy, t, nu], dim=2)
    return pseudovoigt.to(torch.float32), params.to(torch.float32)