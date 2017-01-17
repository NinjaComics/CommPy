#!/usr/bin/python

#Authors: Ravi Sharan B A G <bhagavathula.ravisharan@gmail.com>

"""
===============================================
Optimal Filters (:mod:`commpy.optimal_filters`)
===============================================

.. autosummary::
   :toctree: generated/

   matched_filter            	 -- Matched Filter receiver.
   correlation_filter            -- Correlation Filter receiver.
"""

import numpy as np
from numpy import linspace, pi, exp, convolve, conjugate, sqrt, sum

__all__ = ['matched_filter', 'correlation_filter']

def matched_filter(received_data, channel_response):
    """
    Performs Matched filter operation given a priori channel response.

    Parameters
    ----------
    received_data : Variable length ndarray 
        Received data to be matched with the given channel response.

    channel_response : 1D ndarray (float)
        Impulse response of the channel. In case of AWGN, channel response
        is the impulse response of the transmit pulse.

    Returns
    -------
    post_filt : 1D ndarray (floats)
        Matched filter output sampled at t=kT, the pulse width(if AWGN).
        Refer http://dsp.stackexchange.com/a/9389/3637, for 
        quick reference.
        
    """
    
    T = len(channel_response)
    channel_bar = conjugate(channel_response[::-1])/sum(channel_response**2)
    match_filt = np.array([convolve(received_data[i], channel_bar) for i in\
            xrange(len(received_data))])
    post_filt = np.array([match_filt[i][T-1] for i in xrange(len(match_filt))])
    
    return post_filt

def correlation_filter(received_data, channel_response):
    """
    Performs correlation filter operation on received data

    Parameters
    ----------
    received_data : Variable length ndarray 
        Received data to be matched with the given channel response.

    channel_response : 1D ndarray (float)
        Impulse response of the channel. In case of AWGN, channel response
        is the impulse response of the transmit pulse.

    Returns
    -------
    post_filt : 1D ndarray (floats)
        Correlation filter output

    """

    corr_filt = received_data*channel_response
    post_filt = np.array([sum(corr_filt[i]) for i in range(received_data.size/\
            channel_response.size)])

    return post_filt	
