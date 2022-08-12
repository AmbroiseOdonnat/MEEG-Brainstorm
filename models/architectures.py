#!/usr/bin/env python

"""
This script contains a model to detect spikes and a model
to count the number of spikes inspired by:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
<https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from models import <class>" to use one class.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import math
import numpy as np

from copy import deepcopy

import torch

import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch import Tensor
from utils.utils_ import xavier_initialization, normal_initialization
from utils.utils_ import out_size_CNN

""" ********** Mish activation ********** """


class Mish(nn.Module):

    """ Activation function inspired by:
        `<https://www.bmvc2020-conference.com/assets/papers/0928.pdf>`.
    """

    def __init__(self):

        super().__init__()

    def forward(self,
                x: Tensor):

        return x*torch.tanh(F.softplus(x))


""" ********** Spatial transforming ********** """


class ChannelAttention(nn.Module):

    def __init__(self,
                 emb_size,
                 num_heads,
                 dropout):

        """ Multi-head attention inspired by:
            `"Attention Is All You Need"
            <https://arxiv.org/pdf/1606.08415v3.pdf>`_.

        Args:
            emb_size (int): Size of embedding vectors (here: n_time_points).
                            Warning -> num_heads must be a
                                       dividor of emb_size !
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(emb_size,
                                               num_heads,
                                               dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)

        # Weight initialization
        self.attention.apply(xavier_initialization)

    def forward(self,
                x: Tensor):

        """ Apply spatial transforming.
            Trials can be padded with zeros channels for same sequence length.

        Args:
            x (torch tensor): Batches of trials of dimension
                              [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Batches of trials of dimension
                          [batch_size x 1 x n_channels x n_time_points].
        """

        temp = torch.squeeze(x, dim=1)

        # padded channels are ignored in self-attention
        mask = (temp.mean(dim=-1) == 0) & (temp.std(dim=-1) == 0)
        temp = rearrange(temp, 'b s e -> s b e')
        temp, attention_weights = self.attention(temp,
                                                 temp,
                                                 temp,
                                                 key_padding_mask=mask)
        temp = rearrange(temp, 's b e -> b s e')
        x_attention = self.dropout(temp).unsqueeze(1)
        out = self.norm(x + x_attention)

        return out, attention_weights


""" ********** Embedding and positional encoding ********** """


class PatchEmbedding(nn.Module):

    def __init__(self,
                 seq_len,
                 emb_size,
                 n_maps,
                 position_kernel,
                 channels_kernel,
                 channels_stride,
                 time_kernel,
                 time_stride,
                 dropout):

        """Positional encoding and embedding. Inspired by:
            `"EEGNet: a compact convolutional neural network for EEG-based
            brain–computer interfaces"
            <https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/pdf>`_.

        Args:
            seq_len (int): Sequence length (here: n_time_points).
            emb_size (int): Size of embedding vectors.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            dropout (float): Dropout value.
        """

        super().__init__()

        # Padding values to preserve seq_len
        position_padding = position_kernel-1
        position_padding = int(position_padding / 2) + 1
        new_seq_len = int(seq_len + 2*position_padding
                          - position_kernel + 1)
        time_padding = ((time_stride-1) * new_seq_len
                        + time_kernel) - time_stride
        if (time_kernel % 2 == 0) & (time_stride % 2 == 0):
            time_padding = int(time_padding / 2) - 1
        elif (time_kernel % 2 != 0) & (time_stride % 2 != 0):
            time_padding = int(time_padding / 2) - 1
        else:
            time_padding = int(time_padding / 2)

        # Embedding and positional encoding
        self.embedding = nn.Sequential(
                            nn.Conv2d(1,
                                      n_maps,
                                      (1, position_kernel),
                                      stride=(1, 1),
                                      padding=(0, position_padding)),
                            nn.BatchNorm2d(n_maps),
                            nn.AdaptiveAvgPool2d(((channels_kernel,
                                                   new_seq_len))),
                            nn.Conv2d(n_maps,
                                      n_maps,
                                      (channels_kernel, 1),
                                      stride=(channels_stride, 1),
                                      groups=n_maps),
                            nn.BatchNorm2d(n_maps),
                            Mish(),
                            nn.Dropout(dropout),
                            nn.Conv2d(n_maps,
                                      emb_size,
                                      (1, time_kernel),
                                      stride=(1, time_stride),
                                      padding=(0, time_padding)),
                            nn.BatchNorm2d(emb_size),
                            Mish(),
                            nn.Dropout(dropout),
                            Rearrange('b o c t -> b (c t) o')
                            )
        self.embedding.apply(normal_initialization)

    def forward(self,
                x: Tensor):

        """ Create embeddings with positional encoding.

        Args:
            x (tensor): Batch of trials of dimension
                        [batch_size x 1 x n_channels x seq_len].

        Returns:
            x (tensor): Batches of embeddings of dimension
                        [batch_size x new_seq_len x emb_size].
                        If padding, maintain seq_len value.
        """

        # Create embeddings with positional encoding
        x = self.embedding(x)
        return x


""" ********** Transformer Encoder ********** """


class TransformerEncoder(nn.Sequential):

    """ Multi-head attention inspired by:
        `"Attention Is All You Need"
        <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
    """

    def __init__(self,
                 depth,
                 emb_size,
                 num_heads,
                 expansion,
                 dropout,
                 src_mask=False):

        """
        Args:
            depth (int): Number of Transformer layers.
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient in FF block.
            dropout (float): Dropout value.
            scr_mask (bool): If True, use self-attention
                             with mask in MultiHeadAttention.
        """

        super().__init__()
        self.src_mask = src_mask
        dim = expansion * emb_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim,
                                                   dropout=dropout,
                                                   activation='gelu')
        norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=depth,
                                             norm=norm)

        # Weight initialization
        self.encoder.apply(xavier_initialization)

    def forward(self,
                x: Tensor):

        """ Apply Transformer Encoder.

        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x seq_len x emb_size].

        Returns:
             out (tensor): Batch of trials with dimension
                           [batch_size x seq_len x emb_size].
        """
        x = rearrange(x, 'b s e -> s b e')
        if self.src_mask:

            # Create tensor of size [b x b]
            artifact = torch.einsum('b c , l d -> b l',
                                    x[:, 0],
                                    x[:, 0])

            mask = torch.ones_like(artifact)
            mask = torch.tril(mask, diagonal=0)
            out = self.encoder(x, mask=mask)
        else:
            out = self.encoder(x)
        out = rearrange(out, 's b e -> b s e')

        return out


""" ********** EEGNet ********** """


class EEGNet(nn.Module):

    """ EEGNet inspired by:
        `"EEGNet: A Compact Convolutional Neural Network
        for EEG-based Brain-Computer Interfaces"
        <https://arxiv.org/pdf/1611.08024.pdf>`_.
        Implementation inspired by:
        `<https://github.com/Tammie-Li/RSVP-EEGNet/blob/master/models/eegnet.py>`_
        Predicts probability of spike occurence in a trial.

    Input (tensor): Batch of trials of dimension
                    [batch_size x n_channels x n_time_points].
    Output (tensor): Logits of dimension [batch_size x 1].
    """

    def __init__(self):

        super().__init__()

        # Block 1: conv2d
        self.block1 = nn.Sequential(
                        nn.Conv2d(in_channels=1,
                                  out_channels=8,
                                  kernel_size=(1, 64),
                                  padding=(0, 32),
                                  bias=False),
                        nn.BatchNorm2d(8)
                        )

        # Block 2: depthwiseconv2d
        self.block2 = nn.Sequential(
                        nn.Conv2d(in_channels=8,
                                  out_channels=16,
                                  kernel_size=(20, 1),
                                  groups=2,
                                  bias=False),
                        nn.ELU(),
                        nn.AdaptiveAvgPool2d(output_size=(1, 4)),
                        nn.Dropout()
                        )

        # Block 3: separableconv2d
        self.block3 = nn.Sequential(
                        nn.Conv2d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=(1, 16),
                                  padding=(0, 8),
                                  groups=16,
                                  bias=False),
                        nn.Conv2d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=(1, 1),
                                  bias=False),
                        nn.ELU(),
                        nn.AdaptiveAvgPool2d(output_size=(1, 8)),
                        nn.Dropout()
                        )

        # Block 4: classifier
        self.classifier = nn.Linear(128, 1)

    def forward(self,
                x: Tensor):

        """ Apply EEGNet model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Artificial attention weights
                                        to match other models' outputs.
        """

        # Conv2d
        x = self.block1(x)

        # Depthwise Conv2d
        x = self.block2(x)

        # Separable Conv2d
        x = self.block3(x)

        # Classifier
        x = x.view(x.size(0), -1)
        out, attention_weights = self.classifier(x).squeeze(1), torch.zeros(1)

        return out, attention_weights


""" ********** EEGNet-1D ********** """


class EEGNet_1D(nn.Module):

    """ EEGNet inspired by:
        `"EEGNet: A Compact Convolutional Neural Network
        for EEG-based Brain-Computer Interfaces"
        <https://arxiv.org/pdf/1611.08024.pdf>`_.
        Implementation inspired by:
        `<https://github.com/Tammie-Li/RSVP-EEGNet/blob/master/models/eegnet.py>`_
        Predicts probability of spike occurence in a trial.
        Takes single-channel trial as input.
    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_time_points]
    Output (tensor): Logits of dimension [batch_size].
    """

    def __init__(self):

        super().__init__()

        # Block 1: conv1d
        self.block1 = nn.Sequential(
                        nn.Conv1d(in_channels=1,
                                  out_channels=8,
                                  kernel_size=64,
                                  padding=32,
                                  bias=False),
                        nn.BatchNorm1d(8)
                        )

        # Block 2: depthwiseconv1d
        self.block2 = nn.Sequential(
                        nn.Conv1d(in_channels=8,
                                  out_channels=16,
                                  kernel_size=1,
                                  groups=2,
                                  bias=False),
                        nn.ELU(),
                        nn.AdaptiveAvgPool1d(output_size=4),
                        nn.Dropout()
                        )

        # Block 3: separableconv1d
        self.block3 = nn.Sequential(
                        nn.Conv1d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=16,
                                  padding=8,
                                  groups=16,
                                  bias=False),
                        nn.Conv1d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=1,
                                  bias=False),
                        nn.ELU(),
                        nn.AdaptiveAvgPool1d(output_size=8),
                        nn.Dropout()
                        )

        # Block 4: classifier
        self.classifier = nn.Linear(128, 1)

    def forward(self,
                x: Tensor):

        """ Apply EEGNet model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_time_points].
        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Artificial attention weights
                                        to match other models' outputs.
        """
        x = x.flatten(start_dim=2)

        # Conv1d
        x = self.block1(x)

        # Depthwise Conv1d
        x = self.block2(x)

        # Separable Conv1d
        x = self.block3(x)

        # Classifier
        x = x.view(x.size(0), -1)
        out, attention_weights = self.classifier(x).squeeze(1), torch.zeros(1)

        return out, attention_weights


""" ********** Gated Transformer Network ********** """


class GTN(nn.Module):

    """ Gated Transformer Network inspired by:
        `"Gated Transformer Networks for Multivariate
        Time Series Classification"
        <https://arxiv.org/pdf/2103.14438.pdf>`_.
        Predicts probability of spike occurence in a trial.

    Input (tensor): Batch of trials of dimension
                    [batch_size x n_channels x n_time_points].
    Output (tensor): Logits of dimension [batch_size x 1].
    """

    def __init__(self,
                 n_time_points=201,
                 channel_num_heads=1,
                 channel_dropout=0.1,
                 emb_size=32,
                 positional_encoding=True,
                 channels_kernel=20,
                 depth=3,
                 num_heads=8,
                 expansion=4,
                 transformer_dropout=0.25):

        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
            channel_num_heads (int): Number of heads in ChannelAttention.
            channel_dropout (float): Dropout value in ChannelAttention.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            positional_encoding (bool): If True, add positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Transformer.
        """

        super().__init__()
        self.n_time_points = n_time_points
        self.emb_size = emb_size
        self.positional_encoding = positional_encoding
        self.channel_attention = ChannelAttention(n_time_points,
                                                  channel_num_heads,
                                                  channel_dropout)

        # Step-wise transformer
        self.embedding_1 = nn.Sequential(
                                nn.AdaptiveAvgPool2d((channels_kernel,
                                                      n_time_points)),
                                Rearrange('b o c t -> b (o t) c'),
                                nn.Linear(channels_kernel, emb_size)
                                )
        self.transformer_1 = TransformerEncoder(depth,
                                                emb_size,
                                                num_heads,
                                                expansion,
                                                transformer_dropout,
                                                True)

        # Channel-wise transformer
        self.embedding_2 = nn.Sequential(
                                nn.AdaptiveAvgPool2d((channels_kernel,
                                                      n_time_points)),
                                Rearrange('b o c t -> b (o c) t'),
                                nn.Linear(n_time_points, emb_size)
                                )
        self.transformer_2 = TransformerEncoder(depth,
                                                emb_size,
                                                num_heads,
                                                expansion,
                                                transformer_dropout)

        # Gate
        in_features = emb_size * (channels_kernel + n_time_points)
        self.gate = nn.Linear(in_features, 2)

        # Classifier
        self.classifier = nn.Linear(in_features, 1)

    def forward(self,
                x: Tensor):

        """ Apply GTN model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Attention weights of channel attention.
        """

        # Focus on relevant channels
        attention, attention_weights = self.channel_attention(x)

        # Step-wise encoder
        embedding_1 = self.embedding_1(attention)

        if self.positional_encoding:

            # Add Positional encoding
            pe = torch.ones_like(embedding_1[0])
            position = torch.arange(0, self.n_time_points).unsqueeze(-1)
            temp = torch.Tensor(range(0, self.emb_size, 2))
            temp = temp * -(math.log(10000) / self.emb_size)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)
            embedding_1 += pe
        encoder_1 = self.transformer_1(embedding_1)
        encoder_1 = encoder_1.reshape(encoder_1.shape[0], -1)

        # Channel-wise encoder
        embedding_2 = self.embedding_2(x)
        encoder_2 = self.transformer_2(embedding_2)
        encoder_2 = encoder_2.reshape(encoder_2.shape[0], -1)

        # Merge encoders
        encoder = torch.cat([encoder_1, encoder_2], dim=-1)

        # gate
        gate = F.softmax(self.gate(encoder), dim=-1)
        encoding = torch.cat([encoder_1 * gate[:, 0:1],
                              encoder_2 * gate[:, 1:2]], dim=-1)

        # Classifier
        out = self.classifier(encoding).squeeze(1)

        return out, attention_weights


""" ********** RNN self-attention ********** """


class RNN_self_attention(nn.Module):

    """ RNN self-attention inspired by:
        `"Epileptic spike detection by recurrent neural
        networks with self-attention mechanism"
        <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747560>`_.

    Input (tensor): Batch of trials of dimension
                    [batch_size x n_time_points x 1].
    Output (tensor): Logits of dimension [batch_size].
    """

    def __init__(self,
                 n_time_points):
        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
        """

        super().__init__()
        self.n_time_points = n_time_points
        self.LSTM_1 = nn.LSTM(input_size=1,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.attention = nn.MultiheadAttention(num_heads=1,
                                               embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=8,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True)
        self.classifier = nn.Linear(int(n_time_points/2), 1)

        # Weight initialization
        self.classifier.apply(normal_initialization)

    def forward(self,
                x: Tensor):

        """ Apply 1D-RNN with self-attention model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x n_time_points x 1].

        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Attention weights of channel attention.
        """

        # First LSTM
        self.LSTM_1.flatten_parameters()
        x = x.flatten(start_dim=2)
        x, (_, _) = self.LSTM_1(x.transpose(1, 2))
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)

        # Self-attention Layer
        x_attention, attention_weights = self.attention(x, x, x)

        x = x + x_attention
        x = x.transpose(0, 1)

        # Second LSTM
        self.LSTM_2.flatten_parameters()
        x, (_, _) = self.LSTM_2(x)
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Classifier
        out = self.classifier(x.flatten(1)).squeeze(1)

        return out, attention_weights


""" ********** Spatial Temporal Transformers ********** """


class STT(nn.Module):

    """ Spatial Temporal Transformer inspired by:
        `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
        <https://arxiv.org/pdf/2106.11170.pdf>`_.
        Predicts probability of spike occurence in a trial.

    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Logits of dimension [batch_size x 1].
    """

    def __init__(self,
                 n_time_points=201,
                 channel_num_heads=1,
                 channel_dropout=0.1,
                 emb_size=30,
                 n_maps=5,
                 position_kernel=50,
                 channels_kernel=20,
                 channels_stride=1,
                 time_kernel=20,
                 time_stride=1,
                 positional_dropout=0.25,
                 depth=3,
                 num_heads=10,
                 expansion=4,
                 transformer_dropout=0.25):

        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
            channel_num_heads (int): Number of heads in ChannelAttention.
            channel_dropout (float): Dropout value in ChannelAttention.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            positional_dropout (float): Dropout value for positional encoding.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Transformer.
        """

        super().__init__()
        self.spatial_transforming = ChannelAttention(n_time_points,
                                                     channel_num_heads,
                                                     channel_dropout)
        self.embedding = PatchEmbedding(n_time_points,
                                        emb_size,
                                        n_maps,
                                        position_kernel,
                                        channels_kernel,
                                        channels_stride,
                                        time_kernel,
                                        time_stride,
                                        positional_dropout)
        self.encoder = TransformerEncoder(depth,
                                          emb_size,
                                          num_heads,
                                          expansion,
                                          transformer_dropout)
        flatten_size = emb_size * n_time_points
        self.classifier = nn.Linear(flatten_size, 1)

        # Weight initialization
        self.classifier.apply(normal_initialization)

    def forward(self,
                x: Tensor):

        """ Apply STT model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Attention weights of channel attention.
        """

        # Spatial Transforming
        attention, attention_weights = self.spatial_transforming(x)

        # Embedding
        embedding = self.embedding(attention)

        # Temporal Transforming
        code = self.encoder(embedding)

        # Classifier
        out = self.classifier(code.flatten(1)).squeeze(1)

        return out, attention_weights


""" ********** Variational Auto-Encoder ********** """


class CNN(nn.Module):
    # TODO: add stride
    def __init__(
        self,
        filters,
        kernel_size,
        activation,
        norm_layer=None,
        pool_layer=None,
        pool_size=None,
        pool_every=None,
        pool_offset=0,
        transpose=False,
        padding=0,
    ) -> None:
        # if padding != 0 and len(padding) != len(filters):
        #     raise Exception("Must either have as many padding sets as filters or none.")
        super().__init__()
        kernel_dim = len(kernel_size)
        if transpose:
            conv_fun = eval(f"nn.LazyConvTranspose{kernel_dim}d")
        else:
            conv_fun = eval(f"nn.LazyConv{kernel_dim}d")
        if pool_layer and pool_size is None:
            pool_size = [2] * kernel_dim
        if pool_layer and pool_every is None:
            pool_every = 1

        modules = []
        for i, out_channels in enumerate(filters):
            if padding:
                pad = padding[i]
            else:
                pad = padding
            modules.append(conv_fun(out_channels, kernel_size, padding=pad))
            if norm_layer:
                modules.append(norm_layer)
            modules.append(activation)
            if pool_layer and not (pool_offset + i + 1) % pool_every:
                modules.append(pool_layer)

        self.cnn = nn.Sequential(*modules)

    def forward(self, x) -> Tensor:
        return self.cnn(x)


class MLP(nn.Module):
    def __init__(self, neurons, activation, norm_layer=None) -> None:
        super().__init__()
        modules = []
        for out_features in neurons:
            modules.append(nn.LazyLinear(out_features))
            if norm_layer:
                modules.append(norm_layer)
            modules.append(activation)


""" ********** Variational Auto-Encoder ********** """


class CNN(nn.Module):
    # TODO: add stride
    def __init__(
        self,
        filters,
        kernel_size,
        activation,
        norm_layer=None,
        pool_layer=None,
        pool_size=None,
        pool_every=None,
        pool_offset=0,
        transpose=False,
        padding=0,
    ) -> None:
        # if padding != 0 and len(padding) != len(filters):
        #     raise Exception("Must either have as many padding sets as filters or none.")
        super().__init__()
        kernel_dim = len(kernel_size)
        if transpose:
            conv_fun = eval(f"nn.LazyConvTranspose{kernel_dim}d")
        else:
            conv_fun = eval(f"nn.LazyConv{kernel_dim}d")
        if pool_layer and pool_size is None:
            pool_size = [2] * kernel_dim
        if pool_layer and pool_every is None:
            pool_every = 1

        modules = []
        for i, out_channels in enumerate(filters):
            if padding:
                pad = padding[i]
            else:
                pad = padding
            modules.append(conv_fun(out_channels, kernel_size, padding=pad))
            if norm_layer:
                modules.append(norm_layer)
            modules.append(activation)
            if pool_layer and not (pool_offset + i + 1) % pool_every:
                modules.append(pool_layer)

        self.cnn = nn.Sequential(*modules)

    def forward(self, x) -> Tensor:
        return self.cnn(x)


class MLP(nn.Module):
    def __init__(self, neurons, activation, norm_layer=None) -> None:
        super().__init__()
        modules = []
        for out_features in neurons:
            modules.append(nn.LazyLinear(out_features))
            if norm_layer:
                modules.append(norm_layer)
            modules.append(activation)

        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


def instantiate_config(config):
    config_local = deepcopy(config)
    if config_local.get("norm_layer", None):
        print(
            "warning: normalization won't be supported"
            "until ivadomed upgrades its pytorch"
            "dependancy to torch>=1.9.0"
        )
        config_local["norm_layer"] = None
    # Load required config parameters
    kernel_size = config_local.get("kernel_size", None)
    pool_size = config_local.get("pool_size", None)
    # Deduce dimension of kernels
    norm_dim = 1 if kernel_size is None else len(kernel_size)
    pool_dim = norm_dim if pool_size is None else len(pool_size)

    # Supported activations, normalizations and poolings
    activations = {
        "leaky_relu": f"nn.LeakyReLU",
        "relu": f"nn.ReLU",
        "hard_swish": f"nn.Hardswish",
    }
    normalizations = {
        "batch": f"nn.LazyBatchNorm{norm_dim}d",
        "layer": f"nn.LayerNorm",
        "instance": f"nn.LazyInstanceNorm{norm_dim}d",
    }
    poolings = {
        "max": f"MaxPool{pool_dim}d",
        # "avg": f"AvgPool{pool_dim}d",
        # "adaptive_max": f"AdaptiveMaxPool{pool_dim}d",
        # "adaptive_avg": f"AdaptiveAvgPool{pool_dim}d",
    }
    unpoolings = {
        "max": f"MaxUnpool{pool_dim}d",
    }

    # Dev note: only instantiate the classes actually used in config
    activation = config_local.get("activation", None)
    if activation:
        config_local["activation"] = eval(activations[activation])()
    norm_layer = config_local.get("norm_layer", None)
    if norm_layer:
        config_local["norm_layer"] = eval(normalizations[norm_layer])()
    pool_layer = config_local.get("pool_layer", None)
    if pool_layer:
        if config_local.get("transpose", False):
            config_local["pool_layer"] = eval(unpoolings[pool_layer])()
        else:
            config_local["pool_layer"] = eval(poolings[pool_layer])()

    return config_local


class VAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: validate CNN and MLP config once loaded
        config_cnn = config.get("CNN", None)
        config_mlp = config.get("MLP", None)
        latent_dim = config.get("latent_dim", 2)

        # Instantiate activation, normalization, pooling
        config_cnn = instantiate_config(config_cnn)
        config_mlp = instantiate_config(config_mlp)

        modules = []
        if config_cnn is not None:
            modules.append(CNN(**config_cnn))
        modules.append(nn.Flatten())
        if config_mlp is not None:
            modules.append(MLP(**config_mlp))
        self.encoder = nn.Sequential(*modules)

        # Creation of the latent space mean and variance layers
        self.mu = nn.Sequential(nn.LazyLinear(latent_dim))
        self.log_var = nn.Sequential(nn.LazyLinear(latent_dim))

    def forward(self, x):
        # Encode the example
        x = self.encoder(x)
        # Get mean and variance of the latent variables for the example
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, config, shape, decoder_config=False):
        super().__init__()
        # TODO: validate CNN and MLP config once loaded
        config_cnn = config.get("CNN", None)
        config_mlp = config.get("MLP", None)

        # Change CNN and MLP config to be reversed encoder
        if not decoder_config:
            # CNN
            if config_cnn:
                # Load config parameters
                filters = config_cnn["filters"]
                kernel = config_cnn["kernel_size"]
                stride = config_cnn.get("stride", 1)
                # Compute target size to match in output
                cnn_out_dims = out_size_CNN(shape[1:], kernel, stride, n=len(filters))
                target_ctnn_out_dims = np.flip(cnn_out_dims, 0)
                # Compute output size without padding
                ctnn_out_dims = np.apply_along_axis(
                    out_size_CNN,
                    1,
                    target_ctnn_out_dims,
                    kernel,
                    stride,
                    n=1,
                    transpose=True,
                )[:, 1]
                # Compute padding required to match target size
                padding = list(
                    map(tuple, target_ctnn_out_dims[1:] - ctnn_out_dims[:-1])
                )
                # Save flattened dimension to be use when unflattening
                preflattened_dim = tuple(map(
                    int, np.insert(cnn_out_dims[-1], 0, filters[-1], axis=0))
                )
                # Compute flattened dimension to be used in MLP input
                flattened_dim = filters[-1] * np.prod(cnn_out_dims[-1])

                # Update CNN config
                config_cnn["filters"] = reversed(config_cnn["filters"])
                config_cnn["padding"] = padding
                config_cnn["transpose"] = True
                if config_cnn.get("pool_every", None):
                    config_cnn["pool_offset"] = len(filters) % config_cnn["pool_every"]
            else:
                preflattened_dim = shape
                flattened_dim = np.prod(np.asarray(shape))

            # Update MLP config
            if config_mlp:
                config_mlp["neurons"] = reversed(config_mlp["neurons"])

        # Instantiate activation, normalization, pooling
        config_cnn = instantiate_config(config_cnn)
        config_mlp = instantiate_config(config_mlp)

        modules = []
        if config_mlp is not None:
            modules.append(MLP(**config_mlp))
        modules.append(nn.LazyLinear(flattened_dim))
        modules.append(nn.Unflatten(1, preflattened_dim))
        if config_cnn is not None:
            modules.append(CNN(**config_cnn))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Computing the dims required by the flattening and unflattening ops
        self.encoder = VAEEncoder(config["encoder"])
        self.decoder = VAEDecoder(config["encoder"], config["shape"])

    def forward(self, x):
        mu, log_var = self.encoder(x)
        # Sample from the latent space
        z = self.sample_latent_space(mu, log_var)
        # Decode the sample
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def sample_latent_space(self, mu, log_var):
        return mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))

    def encode(self, x, sample=True):
        # Get mean and variance of the latent variables
        # from the encoded example
        mu, log_var = self.encoder(x)

        if sample:
            # Sample from the latent space
            z = self.sample_latent_space(mu, log_var)
            return z
        else:
            return mu, log_var

    def decode(self, z, sample=False):
        # Decode the sample
        decoded = self.decoder(z)
        if sample:
            """# Get mean and variance of the output
            values from the decoded sample
            mus = decoded[:, 0::2, :, :]
            log_vars = decoded[:, 1::2, :, :]
            # Sample from the parameters
            sampled = self.sample_latent_space(mus, log_vars)
            return sampled"""
            pass
        else:
            return decoded
