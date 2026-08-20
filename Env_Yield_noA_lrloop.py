import os
import pickle
import time
import copy
import math
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import Sequence


os.chdir('projects/def-aali/Thesis_PhD/')





# =============================================================================
# Loop over different batch and epoch configurations.
# =============================================================================
for batch_loop, epoch_loop, lr_loop in [(64, 4, 0.00005), (128, 4, 0.00005)]:
    print('batch_loop:', batch_loop, 'lr_loop:', lr_loop)
    
    # For building the dataset we fix a maximum length (for padding TFRecord data).
    length_max = 40000

    # =========================================================================
    # TFRecord Parsing Functions
    # =========================================================================
    def parse_combined_tfr_element(element):
        data = {
            'area' : tf.io.FixedLenFeature([], tf.string),
            'longitude' : tf.io.FixedLenFeature([], tf.string),
            'latitude' : tf.io.FixedLenFeature([], tf.string),
            
            'time_dependent' : tf.io.FixedLenFeature([], tf.string),
            'soil' : tf.io.FixedLenFeature([], tf.string),
            'texture_v' : tf.io.FixedLenFeature([], tf.string),
            'singles' : tf.io.FixedLenFeature([], tf.string),
            'landcover_v' : tf.io.FixedLenFeature([], tf.string),
            
            'one_hot_encodings' : tf.io.FixedLenFeature([], tf.string),
            'name' : tf.io.FixedLenFeature([], tf.string),
            'cyield' : tf.io.FixedLenFeature([], tf.float32)
        }
        content = tf.io.parse_single_example(element, data)

        # Process and pad the 'area' tensor.
        raw_area = content['area']
        area = tf.io.parse_tensor(raw_area, out_type=tf.float32)
        height = tf.shape(area)[0]
        zeros = tf.zeros([length_max - height], tf.float32)
        area = tf.concat([area, zeros], axis=0)

        # Process and pad the 'time_dependent' tensor.
        raw_time_dependent = content['time_dependent']
        time_dependent = tf.io.parse_tensor(raw_time_dependent, out_type=tf.float32)[:, 2:8, :]
        zeros = tf.zeros([length_max - height, 6, 48], tf.float32)
        time_dependent = tf.concat([time_dependent, zeros], axis=0)

        # Process and pad the 'soil' tensor.
        raw_soil = content['soil']
        soil = tf.io.parse_tensor(raw_soil, out_type=tf.float32)
        zeros = tf.zeros([length_max - height, 4, 6], tf.float32)
        soil = tf.concat([soil, zeros], axis=0)

        # Process and pad the 'texture_v' tensor.
        raw_texture_v = content['texture_v']
        texture_v = tf.io.parse_tensor(raw_texture_v, out_type=tf.float32)
        zeros = tf.zeros([length_max - height, 12, 6], tf.float32)
        texture_v = tf.concat([texture_v, zeros], axis=0)

        # Process and pad the 'singles' tensor.
        raw_singles = content['singles']
        singles = tf.io.parse_tensor(raw_singles, out_type=tf.float32)
        zeros = tf.zeros([length_max - height, 3], tf.float32)
        singles = tf.concat([singles, zeros], axis=0)

        # Process and pad the 'landcover_v' tensor.
        raw_landcover_v = content['landcover_v']
        landcover_v = tf.io.parse_tensor(raw_landcover_v, out_type=tf.float32)
        zeros = tf.zeros([length_max - height, 25, 8], tf.float32)
        landcover_v = tf.concat([landcover_v, zeros], axis=0)

        # Process one-hot encodings.
        onehot = content['one_hot_encodings']
        one_hot_encoding = tf.io.parse_tensor(onehot, out_type=tf.int16)

        # Process the name.
        name = content['name']
        name_feature = tf.io.parse_tensor(name, out_type=tf.string)

        cyield = content['cyield']

        # Process and pad the 'latitude' tensor.
        raw_latitude = content['latitude']
        latitude = tf.io.parse_tensor(raw_latitude, out_type=tf.float32)

        # Process and pad the 'longitude' tensor.
        raw_longitude = content['longitude']
        longitude = tf.io.parse_tensor(raw_longitude, out_type=tf.float32)

        # Cemb ------------------------------------------------------------------------------------------------------
        frequency_num = 64  # Number of sinusoidal frequencies
        min_radius = ((2*np.pi) / 360.0)  # Earth's circumference
        max_radius = ((2*np.pi) / 0.000001) # 0.5 km a part

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num*1.0 - 1))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = timescales
        freq_mat = tf.expand_dims(freq_list, axis=1)
        freq_mat = tf.repeat(freq_mat, 2, axis=1)
        freq_mat = tf.cast(freq_mat, tf.float32)  # Ensure data type consistency


        latitude = tf.expand_dims(latitude, axis=-1)
        longitude = tf.expand_dims(longitude, axis=-1)
        coords = tf.concat([latitude, longitude], axis=-1)

        coords_mat = tf.reshape(coords, (height, 2, 1, 1))
        coords_mat = tf.repeat(coords_mat, frequency_num, axis=2)
        coords_mat = tf.repeat(coords_mat, 2, axis=3)
        spr_embeds = coords_mat * freq_mat
        sin_part = tf.sin(spr_embeds[..., 0::2])
        cos_part = tf.cos(spr_embeds[..., 1::2])
        spr_embeds = tf.concat([tf.reshape(sin_part, tf.shape(sin_part)), 
                                tf.reshape(cos_part, tf.shape(cos_part))], axis=-1)
        spr_embeds = tf.reshape(spr_embeds, (height, -1))

        zeros = tf.zeros([length_max-height, tf.shape(spr_embeds)[1]], tf.float32)
        spr_embeds = tf.concat([spr_embeds, zeros], axis=0)
        # -----------------------------------------------------------------------------

        return (time_dependent, soil, texture_v, singles, landcover_v, one_hot_encoding, area, spr_embeds), cyield


    def get_combined_dataset(filenames):
        options = tf.data.Options()
        options.deterministic = False  # For TF 2.17, use .deterministic
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        dataset = dataset.with_options(options)
        dataset = dataset.map(parse_combined_tfr_element, num_parallel_calls=AUTOTUNE)
        return dataset

    def get_dataset(filenames):
        BATCH_SIZE = batch_loop
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = get_combined_dataset(filenames)
        dataset = dataset.shuffle(100)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    # =========================================================================
    # Define training and validation TFRecord filenames.
    # =========================================================================
    training_names = ['./tfrecords_env_noA/train_36117.tfrecords',
                     './tfrecords_env_noA/train_28091.tfrecords',
                     './tfrecords_env_noA/train_40130.tfrecords',
                     './tfrecords_env_noA/train_0.tfrecords',
                     './tfrecords_env_noA/train_20065.tfrecords',
                     './tfrecords_env_noA/train_4013.tfrecords',
                     './tfrecords_env_noA/train_16052.tfrecords',
                     './tfrecords_env_noA/train_32104.tfrecords',
                     './tfrecords_env_noA/train_8026.tfrecords',
                     './tfrecords_env_noA/train_24078.tfrecords',
                     './tfrecords_env_noA/train_12039.tfrecords']

    validation_names = ['./tfrecords_env_noA/val_4470.tfrecords',
                         './tfrecords_env_noA/val_3576.tfrecords',
                         './tfrecords_env_noA/val_2235.tfrecords',
                         './tfrecords_env_noA/val_447.tfrecords',
                         './tfrecords_env_noA/val_0.tfrecords',
                         './tfrecords_env_noA/val_1341.tfrecords',
                         './tfrecords_env_noA/val_4023.tfrecords',
                         './tfrecords_env_noA/val_894.tfrecords',
                         './tfrecords_env_noA/val_2682.tfrecords',
                         './tfrecords_env_noA/val_3129.tfrecords',
                         './tfrecords_env_noA/val_1788.tfrecords']

    # training_names = ['./tfrecords_env_noA/val_1788.tfrecords']

    # validation_names = ['./tfrecords_env_noA/val_1788.tfrecords']

    training_set = get_dataset(training_names)
    validation_set = get_dataset(validation_names)

    # Count records in each set.
    total_records = 0
    for filename in training_names:
        dataset = tf.data.TFRecordDataset(filename, compression_type=None)
        num_records = sum(1 for _ in dataset)
        total_records += num_records
    print(f"Total number of records in training TFRecords files: {total_records}")

    total_records_val = 0
    for filename in validation_names:
        dataset = tf.data.TFRecordDataset(filename, compression_type=None)
        num_records = sum(1 for _ in dataset)
        total_records_val += num_records
    print(f"Total number of records in validation TFRecords files: {total_records_val}")

    # =========================================================================
    # Distribution Strategy
    # =========================================================================
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Now, we allow variable length for district-level inputs.
    length_max = None

    with strategy.scope():
        crop_classes = 19

        # ----------------------------------------------------------------------
        # Helper Functions: Create Processing Block Layers (with names)
        # ----------------------------------------------------------------------
        def get_process_block_time(num_repeats=3, prefix="time"):
            layers_list = [
                layers.Conv2D(filters=4, kernel_size=(1, 3), strides=(1, 1),
                              padding="valid", activation=None, name=f"{prefix}_conv_initial"),
                layers.BatchNormalization(name=f"{prefix}_bn_initial"),
                layers.Activation('relu', name=f"{prefix}_act_initial")
            ]
            for i in range(num_repeats):
                layers_list.extend([
                    layers.DepthwiseConv2D(kernel_size=(1, 3), strides=(2, 2),
                                           padding='valid', name=f"{prefix}_depthwise_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_depthwise_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_depthwise_{i}"),
                    layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),
                                  padding="valid", activation=None, name=f"{prefix}_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_conv_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_conv_{i}")
                ])
            return tf.keras.Sequential(layers_list, name=f"{prefix}_process_block")

        def get_process_block_soil(num_repeats=3, prefix="soil"):
            layers_list = [
                layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
                              padding="valid", activation=None, name=f"{prefix}_conv_initial"),
                layers.BatchNormalization(name=f"{prefix}_bn_initial"),
                layers.Activation('relu', name=f"{prefix}_act_initial")
            ]
            for i in range(num_repeats):
                layers_list.extend([
                    layers.DepthwiseConv2D(kernel_size=(1, 2), strides=(1, 1),
                                           padding='valid', name=f"{prefix}_depthwise_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_depthwise_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_depthwise_{i}"),
                    layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
                                  padding="valid", activation=None, name=f"{prefix}_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_conv_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_conv_{i}")
                ])
            return tf.keras.Sequential(layers_list, name=f"{prefix}_process_block")

        def get_process_block_texture(num_repeats=3, prefix="texture"):
            layers_list = [
                layers.Conv2D(filters=8, kernel_size=(12, 1), strides=(1, 1),
                              padding="valid", activation=None, name=f"{prefix}_conv_initial"),
                layers.BatchNormalization(name=f"{prefix}_bn_initial"),
                layers.Activation('relu', name=f"{prefix}_act_initial")
            ]
            for i in range(num_repeats):
                layers_list.extend([
                    layers.DepthwiseConv2D(kernel_size=(1, 2), strides=(1, 1),
                                           padding='valid', name=f"{prefix}_depthwise_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_depthwise_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_depthwise_{i}"),
                    layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
                                  padding="valid", activation=None, name=f"{prefix}_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_conv_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_conv_{i}")
                ])
            return tf.keras.Sequential(layers_list, name=f"{prefix}_process_block")

        def get_process_block_landcover(num_repeats=4, prefix="landcover"):
            layers_list = [
                layers.Conv2D(filters=8, kernel_size=(25, 1), strides=(1, 1),
                              padding="valid", activation=None, name=f"{prefix}_conv_initial"),
                layers.BatchNormalization(name=f"{prefix}_bn_initial"),
                layers.Activation('relu', name=f"{prefix}_act_initial")
            ]
            for i in range(num_repeats):
                layers_list.extend([
                    layers.DepthwiseConv2D(kernel_size=(1, 2), strides=(1, 1),
                                           padding='valid', name=f"{prefix}_depthwise_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_depthwise_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_depthwise_{i}"),
                    layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1),
                                  padding="valid", activation=None, name=f"{prefix}_conv_{i}"),
                    layers.BatchNormalization(name=f"{prefix}_bn_conv_{i}"),
                    layers.Activation('relu', name=f"{prefix}_act_conv_{i}")
                ])
            return tf.keras.Sequential(layers_list, name=f"{prefix}_process_block")




        # Corrects non-spatial yield with spatial embeddings
        class ProcessBlockYieldSpa(tf.keras.layers.Layer):
            def __init__(self, index, **kwargs):
                super(ProcessBlockYieldSpa, self).__init__(**kwargs)
                self.index = index
                
                # First block.
                self.fc1 = layers.Dense(16, activation=None, kernel_initializer=glorot_uniform(), name='fc1_ys_' + str(index))
                self.bn1 = layers.BatchNormalization(name='batch_norm1_' + str(index))
                self.act1 = layers.Activation('relu')
                self.dropout1 = layers.Dropout(0.5, name='dropout1_' + str(index))
                
                # Second block.
                self.fc2 = layers.Dense(32, activation=None, kernel_initializer=glorot_uniform(), name='fc2_ys_' + str(index))
                self.bn2 = layers.BatchNormalization(name='batch_norm2_' + str(index))
                self.act2 = layers.Activation('relu')
                self.dropout2 = layers.Dropout(0.5, name='dropout2_' + str(index))
                
                # Third block.
                self.fc3 = layers.Dense(16, activation=None, kernel_initializer=glorot_uniform(), name='fc3_ys_' + str(index))
                self.bn3 = layers.BatchNormalization(name='batch_norm3_' + str(index))
                self.act3 = layers.Activation('relu')
                self.dropout3 = layers.Dropout(0.5, name='dropout3_' + str(index))
                
                # Final dense layer.
                self.fc4 = layers.Dense(1, activation=None, kernel_initializer=glorot_uniform(), name='fc4_ys_' + str(index))
                
            def call(self, inputs, training=None):
                X = self.fc1(inputs)
                X = self.bn1(X, training=training)
                X = self.act1(X)
                X = self.dropout1(X, training=training)
                
                X = self.fc2(X)
                X = self.bn2(X, training=training)
                X = self.act2(X)
                X = self.dropout2(X, training=training)
                
                X = self.fc3(X)
                X = self.bn3(X, training=training)
                X = self.act3(X)
                X = self.dropout3(X, training=training)
                
                X = self.fc4(X)
                return X
            

        # ----------------------------------------------------------------------
        # Custom Production Estimation Layer
        # ----------------------------------------------------------------------
        class ProductionEstimateLayer(tf.keras.layers.Layer):
            def __init__(self, crop_classes, **kwargs):
                super(ProductionEstimateLayer, self).__init__(**kwargs)
                self.crop_classes = crop_classes

                # Create processing blocks for each time and soil channel.
                self.process_block_time_list = [
                    get_process_block_time(prefix=f"time_channel_{i}") for i in range(6)
                ]
                self.process_block_soil_list = [
                    get_process_block_soil(prefix=f"soil_channel_{i}") for i in range(4)
                ]
                self.process_block_texture = get_process_block_texture(prefix="texture")
                self.process_block_landcover = get_process_block_landcover(prefix="landcover")

                # Final dense layers with names.
                self.fc1 = layers.Dense(32, activation=None, kernel_initializer=glorot_uniform(), name='fc1')
                self.bn1 = layers.BatchNormalization(name='batch_norm1')
                self.act1 = layers.Activation('relu', name='act1')
                self.dropout1 = layers.Dropout(0.5, name='dropout1')

                self.fc2 = layers.Dense(64, activation=None, kernel_initializer=glorot_uniform(), name='fc2')
                self.bn2 = layers.BatchNormalization(name='batch_norm2')
                self.act2 = layers.Activation('relu', name='act2')
                self.dropout2 = layers.Dropout(0.5, name='dropout2')

                self.fc3 = layers.Dense(32, activation=None, kernel_initializer=glorot_uniform(), name='fc3')
                self.bn3 = layers.BatchNormalization(name='batch_norm3')
                self.act3 = layers.Activation('relu', name='act3')
                self.dropout3 = layers.Dropout(0.5, name='dropout3')

                self.fc4 = layers.Dense(crop_classes, activation=None, kernel_initializer=glorot_uniform(), name='fc4')

            def call(self, inputs, training=None):
                # Unpack the inputs.
                X_time, X_soil, X_texture, X_singles, X_landcover = inputs
                batch_size = tf.shape(X_time)[0]

                # Determine which samples are all zeros.
                is_zero_sample = tf.reduce_all(tf.equal(X_time, 0), axis=[1, 2])
                non_zero_indices = tf.where(~is_zero_sample)[:, 0]
                output_tensor = tf.zeros((batch_size, self.crop_classes), dtype=tf.float32)

                # Gather non-zero samples.
                non_zero_X_time = tf.gather(X_time, non_zero_indices)
                non_zero_X_soil = tf.gather(X_soil, non_zero_indices)
                non_zero_X_texture = tf.gather(X_texture, non_zero_indices)
                non_zero_X_singles = tf.gather(X_singles, non_zero_indices)
                non_zero_X_landcover = tf.gather(X_landcover, non_zero_indices)

                outputs = []

                # Append max and min values along axis=2.
                max_values = tf.reshape(tf.reduce_max(non_zero_X_time, axis=2), [-1, 6])
                min_values = tf.reshape(tf.reduce_min(non_zero_X_time, axis=2), [-1, 6])
                outputs.append(max_values)
                outputs.append(min_values)

                # Process each of the 6 time channels.
                for i in range(6):
                    channel_data = non_zero_X_time[:, i, :]  # shape: (num_nonzero, 48)
                    reshaped = tf.reshape(channel_data, [-1, 1, 48, 1])
                    processed = self.process_block_time_list[i](reshaped, training=training)
                    flat = layers.Flatten(name=f"time_channel_{i}_flatten")(processed)
                    outputs.append(flat)

                # Process each of the 4 soil channels.
                for i in range(4):
                    channel_data = non_zero_X_soil[:, i, :]  # shape: (num_nonzero, 6)
                    reshaped = tf.reshape(channel_data, [-1, 1, 6, 1])
                    processed = self.process_block_soil_list[i](reshaped, training=training)
                    flat = layers.Flatten(name=f"soil_channel_{i}_flatten")(processed)
                    outputs.append(flat)

                # Process texture.
                reshaped_texture = tf.reshape(non_zero_X_texture, [-1, 12, 6, 1])
                processed_texture = self.process_block_texture(reshaped_texture, training=training)
                flat_texture = layers.Flatten(name="texture_flatten")(processed_texture)
                outputs.append(flat_texture)

                # Process landcover.
                reshaped_landcover = tf.reshape(non_zero_X_landcover, [-1, 25, 8, 1])
                processed_landcover = self.process_block_landcover(reshaped_landcover, training=training)
                flat_landcover = layers.Flatten(name="landcover_flatten")(processed_landcover)
                outputs.append(flat_landcover)

                # Process singles.
                flat_singles = layers.Flatten(name="singles_flatten")(non_zero_X_singles)
                outputs.append(flat_singles)

                # Concatenate all features.
                concat_out = layers.Concatenate(axis=1, name="concat_features")(outputs)

                # Final dense layers.
                X = self.fc1(concat_out)
                X = self.bn1(X, training=training)
                X = self.act1(X)
                X = self.dropout1(X, training=training)

                X = self.fc2(X)
                X = self.bn2(X, training=training)
                X = self.act2(X)
                X = self.dropout2(X, training=training)

                X = self.fc3(X)
                X = self.bn3(X, training=training)
                X = self.act3(X)
                X = self.dropout3(X, training=training)

                X = self.fc4(X)

                # Scatter the non-zero outputs back to their original positions.
                output_tensor = tf.tensor_scatter_nd_update(
                    output_tensor, tf.expand_dims(non_zero_indices, axis=1), X
                )
                return output_tensor



        # Refine spatial embeddings
        class SpatialContextEstimator(tf.keras.layers.Layer):
            def __init__(self, sinus=32, **kwargs):
                super(SpatialContextEstimator, self).__init__(**kwargs)
                self.sinus = sinus
        
                # Define the layers for processing non-zero samples.
                self.fc1 = layers.Dense(64, activation=None, kernel_initializer=glorot_uniform(), name='fc1_sp')
                self.bn1 = layers.BatchNormalization(name='batch_norm1_sp')
                self.act1 = layers.Activation('tanh')
                self.dropout1 = layers.Dropout(0.5, name='dropout1_sp')
        
                self.fc2 = layers.Dense(32, activation=None, kernel_initializer=glorot_uniform(), name='fc2_sp')
                self.bn2 = layers.BatchNormalization(name='batch_norm2_sp')
                self.act2 = layers.Activation('tanh')
                self.dropout2 = layers.Dropout(0.5, name='dropout2_sp')
        
                self.fc3 = layers.Dense(64, activation=None, kernel_initializer=glorot_uniform(), name='fc3_sp')
                self.bn3 = layers.BatchNormalization(name='batch_norm3_sp')
                self.act3 = layers.Activation('tanh')
                self.dropout3 = layers.Dropout(0.5, name='dropout3_sp')
        
                self.fc4 = layers.Dense(sinus, activation='tanh', kernel_initializer=glorot_uniform(), name='fc4_sp')
        
            def call(self, inputs, training=None):
                # Unpack the inputs.
                cemb, X_time = inputs
        
                # Check if each sample in X_time is all zeros.
                is_zero_sample = tf.reduce_all(tf.equal(X_time, 0), axis=[1, 2])
                non_zero_indices = tf.where(~is_zero_sample)[:, 0]
                batch_size = tf.shape(X_time)[0]
                output_tensor = tf.zeros((batch_size, self.sinus), dtype=tf.float32)
        
                # Gather non-zero samples from cemb.
                non_zero_cemb = tf.gather(cemb, non_zero_indices)
                X = non_zero_cemb
        
                # Process through the dense blocks.
                X = self.fc1(X)
                X = self.bn1(X, training=training)
                X = self.act1(X)
                X = self.dropout1(X, training=training)
        
                X = self.fc2(X)
                X = self.bn2(X, training=training)
                X = self.act2(X)
                X = self.dropout2(X, training=training)
        
                X = self.fc3(X)
                X = self.bn3(X, training=training)
                X = self.act3(X)
                X = self.dropout3(X, training=training)
        
                X = self.fc4(X)
        
                # Scatter the non-zero outputs back to their original positions.
                output_tensor = tf.tensor_scatter_nd_update(
                    output_tensor, tf.expand_dims(non_zero_indices, axis=1), X
                )
                return output_tensor





        class YieldAdjuster(tf.keras.layers.Layer):
            def __init__(self, classes=19, **kwargs):
                super(YieldAdjuster, self).__init__(**kwargs)
                self.classes = classes
                # Create one processing block per crop class.
                self.blocks = [ProcessBlockYieldSpa(index=i) for i in range(classes)]
                
            def call(self, inputs, training=None):
                # Unpack the inputs.
                cemb, yieldd, X_time = inputs
                
                # Identify samples in which X_time is entirely zeros.
                is_zero_sample = tf.reduce_all(tf.equal(X_time, 0), axis=[1, 2])
                non_zero_indices = tf.where(~is_zero_sample)[:, 0]
                batch_size = tf.shape(X_time)[0]
                output_tensor = tf.zeros((batch_size, self.classes), dtype=tf.float32)
                
                # Gather non-zero samples.
                non_zero_cemb = tf.gather(cemb, non_zero_indices)
                non_zero_yield = tf.gather(yieldd, non_zero_indices)
                
                outputs = []
                # For each crop class, process the concatenated features.
                for i in range(self.classes):
                    # Concatenate along axis=1 the crop-agnostic features with the crop-specific yield.
                    crop_specific_input = tf.concat([non_zero_cemb, non_zero_yield[:, i:i+1]], axis=1)
                    processed_output = self.blocks[i](crop_specific_input, training=training)
                    outputs.append(processed_output)
                
                # Concatenate the outputs along the class dimension.
                X = tf.concat(outputs, axis=1)
                # Scatter the processed outputs back to their original batch positions.
                output_tensor = tf.tensor_scatter_nd_update(
                    output_tensor, tf.expand_dims(non_zero_indices, axis=1), X
                )
                return output_tensor



            

        # ----------------------------------------------------------------------
        # Custom Layer to Reshape Production Estimates into District Layout
        # ----------------------------------------------------------------------
        class ReshapeDistrict(layers.Layer):
            def __init__(self, crop_classes, **kwargs):
                super(ReshapeDistrict, self).__init__(**kwargs)
                self.crop_classes = crop_classes

            def call(self, inputs, **kwargs):
                # inputs is a list: [p_estimate, X_time_district]
                p_estimate, X_time_district = inputs
                district_length = tf.shape(X_time_district)[1]
                return tf.reshape(p_estimate, [-1, district_length, self.crop_classes])


                

        # ----------------------------------------------------------------------
        # Build the Production Estimate Model (Per-Sample)
        # ----------------------------------------------------------------------
        # Define inputs for the per-sample (e.g., pixel) production estimation.
        X_time = layers.Input(shape=(6, 48), name='input_time')
        X_soil = layers.Input(shape=(4, 6), name='input_soil')
        X_texture = layers.Input(shape=(12, 6), name='input_texture')
        X_singles = layers.Input(shape=(3,), name='input_singles')
        X_landcover = layers.Input(shape=(25, 8), name='input_landcover')
        X_cemb = layers.Input(shape=(256,), name='input_cemb')

        prod_est_layer = ProductionEstimateLayer(crop_classes=crop_classes, name="production_estimate_layer")
        production_estimate_output = prod_est_layer([X_time, X_soil, X_texture, X_singles, X_landcover])
        production_estimate_model = Model(
            [X_time, X_soil, X_texture, X_singles, X_landcover],
            production_estimate_output,
            name="production_estimate_model"
        )


        spatial_est_layer = SpatialContextEstimator(sinus=32, name="spatial_estimate_layer")
        spatial_estimate_output = spatial_est_layer([X_cemb, X_time])
        spatial_estimate_model = Model(
            [X_cemb, X_time],
            spatial_estimate_output,
            name="spatial_estimate_model"
        )



        X_reduced_cemb = layers.Input(shape=(32,))
        X_unadjusted_yield = layers.Input(shape=(19,))
        
        spatial_adjuster_layer = YieldAdjuster(classes=19, name="spatial_adjuster_layer")
        spatial_adjuster_output = spatial_adjuster_layer([X_reduced_cemb, X_unadjusted_yield, X_time])
        spatial_adjuster_model = Model(
            [X_reduced_cemb, X_unadjusted_yield, X_time],
            spatial_adjuster_output,
            name="spatial_adjuster_model"
        )


        # ----------------------------------------------------------------------
        # Build the Overall Model for Yield Estimation
        # ----------------------------------------------------------------------
        # District-level inputs (with variable length along the first axis).
        X_time_district = layers.Input(shape=(length_max, 6, 48), name='X_time_district')
        X_time_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 6, 48]),
                                         name='X_time_district2')(X_time_district)

        X_soil_district = layers.Input(shape=(length_max, 4, 6), name='X_soil_district')
        X_soil_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 4, 6]),
                                         name='X_soil_district2')(X_soil_district)

        X_texture_district = layers.Input(shape=(length_max, 12, 6), name='X_texture_district')
        X_texture_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 12, 6]),
                                            name='X_texture_district2')(X_texture_district)

        X_singles_district = layers.Input(shape=(length_max, 3), name='X_singles_district')
        X_singles_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 3]),
                                            name='X_singles_district2')(X_singles_district)

        X_landcover_district = layers.Input(shape=(length_max, 25, 8), name='X_landcover_district')
        X_landcover_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 25, 8]),
                                              name='X_landcover_district2')(X_landcover_district)

        X_cemb_district = layers.Input(shape=(length_max, 256), name='X_cemb_district')
        X_cemb_district2 = layers.Lambda(lambda x: tf.reshape(x, [-1, 256]),
                                              name='X_cemb_district2')(X_cemb_district)

        # Compute production estimate at the district level.
        p_estimate = production_estimate_model([X_time_district2, X_soil_district2, X_texture_district2, 
                                                X_singles_district2, X_landcover_district2])


        s_estimate = spatial_estimate_model([X_cemb_district2, X_time_district2])

        p_estimate = spatial_adjuster_model([s_estimate, p_estimate, X_time_district2])

        # Use the custom ReshapeDistrict layer to reshape p_estimate.
        p_estimate = ReshapeDistrict(crop_classes, name="reshape_district")([p_estimate, X_time_district])

        

        # Input for pixel area (used to calculate yield).
        input_pixel_area = layers.Input(shape=(length_max,), name='input_pixel_area')
        input_pixel_area_reshaped = layers.Lambda(lambda x: tf.reshape(x, [-1, tf.shape(x)[1], 1]),
                                                  name='reshape_pixel_area')(input_pixel_area)
        pixel_area = layers.Lambda(lambda x: tf.tile(x, [1, 1, crop_classes]) * 0.000247105,
                                   name='tile_pixel_area')(input_pixel_area_reshaped)

        # Calculate yield as an area-weighted sum.
        area_weighted_production = layers.Lambda(lambda args: tf.math.multiply(*args),
                                                 name='area_weighted_production')([p_estimate, pixel_area])
        p_estimate_sum = layers.Lambda(lambda x: K.sum(x, axis=1), name='p_estimate_sum')(area_weighted_production)
        total_area = layers.Lambda(lambda x: K.sum(x, axis=1), name='total_area')(pixel_area)
        yield_estimate = layers.Lambda(lambda args: tf.math.divide(*args),
                                       name='yield_estimate')([p_estimate_sum, total_area])

        # Only learn from known crops.
        input_onehot_encoded_crops = layers.Input((crop_classes,), name='input_onehot_encoded_crops')
        yield_estimate_weighted = layers.Lambda(lambda args: tf.math.multiply(*args),
                                                name='yield_estimate_weighted')([yield_estimate, input_onehot_encoded_crops])
        yield_estimate_weighted = layers.Lambda(lambda x: K.sum(x, axis=1),
                                                name='final_yield_estimate')(yield_estimate_weighted)

        newmodel = Model(
            [X_time_district, X_soil_district, X_texture_district, X_singles_district, 
             X_landcover_district, input_onehot_encoded_crops, input_pixel_area, X_cemb_district],
            [yield_estimate_weighted],
            name="yield_estimation_model"
        )

        # ----------------------------------------------------------------------
        # Compile the Model
        # ----------------------------------------------------------------------
        # opt = keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
        opt = keras.optimizers.RMSprop(learning_rate=lr_loop)
        newmodel.compile(optimizer=opt, loss='mean_absolute_error')

        # Load best validation loss if available.
        best_val_path = './sbatch/Env_Yield_noA_lrloop.pickle'
        if os.path.isfile(best_val_path):
            with open(best_val_path, 'rb') as handle:
                best_val = pickle.load(handle)
            # best_val = 10
        else:
            best_val = 10

        # ----------------------------------------------------------------------
        # Custom Callback to Save Best Model Weights
        # ----------------------------------------------------------------------
        class CustomCallback(keras.callbacks.Callback):
            def __init__(self, best_val_path, initial_best):
                super(CustomCallback, self).__init__()
                self.best_val_path = best_val_path
                self.best_val = initial_best

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val = logs.get('val_loss')
                if val is not None and val < self.best_val:
                    self.best_val = val
                    print(f"\nValidation loss improved to {self.best_val:.4f} -- saving weights.")
                    with open(self.best_val_path, 'wb') as handle:
                        pickle.dump(self.best_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    self.model.save_weights('./sbatch/Env_Yield_noA_lrloop.weights.h5')
                    # self.best_val = 10

        # ----------------------------------------------------------------------
        # Custom Data Generator
        # ----------------------------------------------------------------------
        class CustomDataGenerator(Sequence):
            def __init__(self, tfrecords_dataset, batch_size_, total_records):
                self.tfrecords_dataset = tfrecords_dataset
                self.iterator = iter(self.tfrecords_dataset)
                self.batch_size_ = batch_size_
                self.total_records = total_records 

            def __len__(self):
                return int(np.ceil(self.total_records / self.batch_size_))

            # def generate_mask(self, batch_size):
            #     modify_indices = np.random.rand(batch_size) < 0.1
            #     landcover_mask = np.ones((batch_size, 25, 8))
            #     landcover_mask[modify_indices, :, :] = 0  
            #     return landcover_mask

            def generate_mask(self, batch_size):
                return np.random.choice([0, 1], size=(batch_size, 40000, 6, 48), p=[0.05, 0.95])
            
            def on_epoch_end(self):
                self.iterator = iter(self.tfrecords_dataset)

            def compute_sample_weights(self, one_hot_encoding):
                one_hot_encoding = tf.cast(one_hot_encoding, tf.float32)
                sample_weights = tf.constant([
                    15.0, 40.0, 1.0, 5.0, 50.0, 
                    3.0, 50.0, 5.0, 50.0, 50.0, 
                    25.0, 10.0, 10.0, 40.0, 10.0, 
                    50.0, 2.0, 15.0, 50.0
                ]) * 0.02 * one_hot_encoding
                sample_weights = K.sum(sample_weights, axis=[1])
                sample_weights = tf.reshape(sample_weights, [-1, 1])
                return sample_weights

            def __getitem__(self, index):
                try:
                    self.thebatch = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.tfrecords_dataset)
                    self.thebatch = next(self.iterator)

                self.X_time = self.thebatch[0][0]
                self.X_soil = self.thebatch[0][1]
                self.X_texture = self.thebatch[0][2]
                self.X_singles = self.thebatch[0][3]
                self.X_landcover = self.thebatch[0][4]
                self.input_onehot_encoded_crops = self.thebatch[0][5]
                self.area = self.thebatch[0][6]
                self.cemb = self.thebatch[0][7]
                self.cyield = self.thebatch[1]

                # mask_batch = self.generate_mask(self.batch_size_)
                # self.X_time = self.X_time*mask_batch

                inputs = (
                    self.X_time, self.X_soil, self.X_texture,
                    self.X_singles, self.X_landcover, self.input_onehot_encoded_crops, self.area, self.cemb
                )
                targets = self.cyield
                sample_weights = self.compute_sample_weights(self.input_onehot_encoded_crops)
                return inputs, targets, sample_weights

        # Load previously saved weights if available.
        model_weights_path = './sbatch/Env_Yield_noA_lrloop.weights.h5'
        if os.path.isfile(model_weights_path):
            newmodel.load_weights(model_weights_path)

        custom_callback = CustomCallback(best_val_path=best_val_path, initial_best=best_val)

        # ----------------------------------------------------------------------
        # Train the Model
        # ----------------------------------------------------------------------
        newmodel.fit(
            CustomDataGenerator(training_set, batch_size_=batch_loop, total_records=total_records),
            epochs=epoch_loop,
            validation_data=CustomDataGenerator(validation_set, batch_size_=batch_loop, total_records=total_records_val),
            callbacks=[custom_callback]
        )
