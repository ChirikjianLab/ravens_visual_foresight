#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

from operator import concat
from tensorflow import keras as K

def UNet(imgs_shape, 
         output_channel,
         fms=16,
         use_upsampling=False,
         use_dropout=True,
         dropout=0.2):
    """
    U-Net Model
    ===========
    Based on https://arxiv.org/abs/1505.04597
    The default uses UpSampling2D (nearest neighbors interpolation) in
    the decoder path. The alternative is to use Transposed
    Convolution.
    """

    if use_upsampling:
        print("Using UpSampling2D")
    else:
        print("Using Transposed Convolution")

    input_shape = imgs_shape
    inputs = K.layers.Input(input_shape, name="inputImage")

    # Convolution parameters
    params = dict(kernel_size=(3, 3), activation="relu",
                    padding="same",
                    kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                        padding="same")

    encodeA = K.layers.Conv2D(
        name="encodeAa", filters=fms, **params)(inputs)
    encodeA = K.layers.Conv2D(
        name="encodeAb", filters=fms, **params)(encodeA)
    poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

    encodeB = K.layers.Conv2D(
        name="encodeBa", filters=fms*2, **params)(poolA)
    encodeB = K.layers.Conv2D(
        name="encodeBb", filters=fms*2, **params)(encodeB)
    poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

    encodeC = K.layers.Conv2D(
        name="encodeCa", filters=fms*4, **params)(poolB)
    if use_dropout:
        encodeC = K.layers.SpatialDropout2D(dropout)(encodeC)
    encodeC = K.layers.Conv2D(
        name="encodeCb", filters=fms*4, **params)(encodeC)

    poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

    encodeD = K.layers.Conv2D(
        name="encodeDa", filters=fms*8, **params)(poolC)
    if use_dropout:
        encodeD = K.layers.SpatialDropout2D(dropout)(encodeD)
    encodeD = K.layers.Conv2D(
        name="encodeDb", filters=fms*8, **params)(encodeD)

    poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

    encodeE = K.layers.Conv2D(
        name="encodeEa", filters=fms*16, **params)(poolD)
    encodeE = K.layers.Conv2D(
        name="encodeEb", filters=fms*16, **params)(encodeE)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="upE", size=(2, 2))(encodeE)
    else:
        up = K.layers.Conv2DTranspose(name="transconvE", filters=fms*8,
                                        **params_trans)(encodeE)
    concatD = K.layers.concatenate(
        [up, encodeD], axis=-1, name="concatD")

    decodeC = K.layers.Conv2D(
        name="decodeCa", filters=fms*8, **params)(concatD)
    decodeC = K.layers.Conv2D(
        name="decodeCb", filters=fms*8, **params)(decodeC)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="upC", size=(2, 2))(decodeC)
    else:
        up = K.layers.Conv2DTranspose(name="transconvC", filters=fms*4,
                                        **params_trans)(decodeC)
    concatC = K.layers.concatenate(
        [up, encodeC], axis=-1, name="concatC")

    decodeB = K.layers.Conv2D(
        name="decodeBa", filters=fms*4, **params)(concatC)
    decodeB = K.layers.Conv2D(
        name="decodeBb", filters=fms*4, **params)(decodeB)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="upB", size=(2, 2))(decodeB)
    else:
        up = K.layers.Conv2DTranspose(name="transconvB", filters=fms*2,
                                        **params_trans)(decodeB)
    concatB = K.layers.concatenate(
        [up, encodeB], axis=-1, name="concatB")

    decodeA = K.layers.Conv2D(
        name="decodeAa", filters=fms*2, **params)(concatB)
    decodeA = K.layers.Conv2D(
        name="decodeAb", filters=fms*2, **params)(decodeA)

    if use_upsampling:
        up = K.layers.UpSampling2D(name="upA", size=(2, 2))(decodeA)
    else:
        up = K.layers.Conv2DTranspose(name="transconvA", filters=fms,
                                        **params_trans)(decodeA)
    concatA = K.layers.concatenate(
        [up, encodeA], axis=-1, name="concatA")

    convOut = K.layers.Conv2D(
        name="convOuta", filters=fms, **params)(concatA)
    convOut = K.layers.Conv2D(
        name="convOutb", filters=fms, **params)(convOut)

    prediction = K.layers.Conv2D(name="outputData",
                                    filters=output_channel, kernel_size=(1, 1),
                                    kernel_initializer="he_uniform",
                                    activation="relu")(convOut)

    return inputs, prediction