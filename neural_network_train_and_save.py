import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image


def subersample(x, pattern):
    n_samples = pattern.size
    x_undersampled = np.zeros((x.shape[0], n_samples, x.shape[3]))

    for img in range(x.shape[0]):
        x_vec_real = np.reshape(x[img,:,:,0], newshape=(x.shape[1]*x.shape[2]))
        x_vec_imag = np.reshape(x[img,:,:,1], newshape=(x.shape[1]*x.shape[2]))
        x_undersampled[img,:,0] = x_vec_real[pattern]
        x_undersampled[img,:,1] = x_vec_imag[pattern]

    return x_undersampled

setup_couter = 0

# lists of options
# every combination of options will be tested during training
batch_normalisation_list = [False] #[False, True]
momentum_list = [0.8] #[0.0,0.2,0.4,0.6,0.8]
learning_rate_list = [0.00002]  #[0.0002, 0.00002, 0.000002]:
noise_list = [False] #[False, True]
drop_out_list = [True] #[False, True]
optimizer_list = ['rmsprop']

for pattern_info in [
    'pattern_N64_red2.0_varDens',
]:
    for batch_normalisation in batch_normalisation_list:
        for momentum in momentum_list:
            for learning_rate in learning_rate_list:
                for noise in noise_list:
                    for drop_out in drop_out_list:
                        for optimizer in optimizer_list:

                            setup_couter += 1

                            info = 'mom:{}_lr:{}_noise:{}_bn:{}_drop:{}'.format(momentum, learning_rate, noise, batch_normalisation, drop_out)

                            if optimizer == 'adam' and momentum != 0:
                                print('skip {}'.format(info))
                                continue


                            # img width and height
                            n = 64
                            # normalize whole input and output to  -1 to 1



                            pattern = np.loadtxt('pattern/{}'.format(pattern_info)).astype('int16')
                            pattern -= 1
                            n_samples = pattern.size

                            # tf.reset_default_graph()
                            # paths
                            path_data = ''
                            path_saving = 'saved_model_{}/'.format(n)
                            path_logs = os.path.join('logs', str(n))

                            # Create folders if they dont exist
                            if not os.path.exists(path_logs):
                                os.makedirs(path_logs)

                            dir_to_save_train = os.path.join(path_logs, 'train')
                            dir_to_save_val = os.path.join(path_logs, 'validation')
                            dir_to_save_test = os.path.join(path_logs, 'test')

                            # Create folders for saving summary stuff for tensorboard if they dont exist
                            if not os.path.exists(dir_to_save_train):
                                os.makedirs(dir_to_save_train)
                            if not os.path.exists(dir_to_save_val):
                                os.makedirs(dir_to_save_val)
                            if not os.path.exists(dir_to_save_test):
                                os.makedirs(dir_to_save_test)

                            dir_to_save_train = os.path.join(dir_to_save_train, info)
                            dir_to_save_val = os.path.join(dir_to_save_val, info)
                            dir_to_save_test = os.path.join(dir_to_save_test, info)

                            if not os.path.exists(dir_to_save_train):
                                os.makedirs(dir_to_save_train)
                            if not os.path.exists(dir_to_save_val):
                                os.makedirs(dir_to_save_val)
                            if not os.path.exists(dir_to_save_test):
                                os.makedirs(dir_to_save_test)

                            # load input and output data
                            x = np.load(path_data + 'x_{}_{}.npy'.format(n, 'lin_pix_min_max'))
                            y = np.load(path_data + 'y_{}.npy'.format(n))

                            # devide train_temp into val and train
                            # 15% test
                            # 15% val (0.85*x = 0.15 => x=0.15/0.85)=0.1765
                            # 70% train
                            x_train_temp, x_test, y_train_temp, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
                            x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1765, random_state=1)
                            del x_train_temp, y_train_temp, x, y # delete not needed data

                            x_train = subersample(x=x_train, pattern=pattern)
                            x_val = subersample(x=x_val, pattern=pattern)
                            x_test = subersample(x=x_test, pattern=pattern)

                            # make results reproducible
                            seed = 14
                            np.random.seed(seed)

                            def generateBatchIndeces(x_train, batch_size):

                                n_samples = x_train.shape[0] # e.g. 950 samples
                                n_batches = n_samples // batch_size # 9 n_batches each 100 samples

                                # draw 900 random samples
                                all_drawn_indeces = np.random.choice(x_train.shape[0], n_batches * batch_size, replace=False)

                                # print('number of samples: {}'.format(n_samples))
                                # print('number of samples per batch: {}'.format(batch_size))
                                # print('number of batches: {}'.format(n_batches))
                                # print('number of drawn samples: {}'.format(n_batches * batch_size))

                                list_of_batches_ind = []
                                batch_ind = []

                                for ind in all_drawn_indeces:
                                    batch_ind.append(ind)

                                    if len(batch_ind) == batch_size:
                                        list_of_batches_ind.append(batch_ind)
                                        batch_ind = []

                                return list_of_batches_ind, n_batches


                            # with tf.name_scope('Input'):
                            X = tf.placeholder(tf.float32, [None, n_samples, 2], 'X') # 2 for real/imag, None for variable size of samples
                            # with tf.name_scope('Output'):
                            Y = tf.placeholder(tf.float32, [None, n, n], 'Y')
                            Is_training = tf.placeholder(tf.bool)

                            keep_prob = tf.placeholder(tf.float32) # dropout (keep probability) - not used right now

                            # functions for the network model
                            def multiply_noise(x):
                                return x[:, :, :] * np.random.normal(1, 0.01, (x.shape[0], x.shape[1], x.shape[2])).astype('float32')


                            def automap_network(x, is_training):

                                n_out = np.int(n*n)

                                with tf.name_scope('Reshape'):
                                    # (n_im, n, n, 2) -> (n_im, n*n*2)
                                    input_flat = tf.contrib.layers.flatten(x)
                                with tf.device('/gpu:0'):
                                #  (n_im, n*n*2) -> (n_im, n*n)

                                    FC1 = tf.contrib.layers.fully_connected(
                                        input_flat,
                                        n_out,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=None,
                                        biases_initializer=tf.zeros_initializer(),
                                        # biases_initializer=None,
                                        biases_regularizer=None,
                                        reuse=tf.AUTO_REUSE,  #
                                        variables_collections=None,
                                        outputs_collections=None,
                                        trainable=True,
                                        scope='fc1'
                                    )

                                    # Activation
                                    FC1 = tf.nn.tanh(FC1)
                                    # Batch Norm
                                    if batch_normalisation:
                                        FC1 = tf.layers.batch_normalization(
                                            inputs=FC1,
                                            training=is_training,
                                            name='FC1_bn'
                                        )
                                    # Dropout
                                    if drop_out:
                                        FC1 = tf.layers.dropout(
                                            inputs=FC1,
                                            rate=0.01,
                                            training=is_training,
                                            name='FC1_dropout'
                                        )

                                    FC2 = tf.contrib.layers.fully_connected(
                                        FC1,
                                        n_out,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=None,
                                        biases_initializer=tf.zeros_initializer(),
                                        #biases_initializer=None,
                                        biases_regularizer=None,
                                        reuse=tf.AUTO_REUSE,#
                                        variables_collections=None,
                                        outputs_collections=None,
                                        trainable=True,
                                        scope='fc2')
                                    # Activation
                                    FC2 = tf.nn.tanh(FC2)
                                    # Batch Norm
                                    if batch_normalisation:
                                        FC2 = tf.layers.batch_normalization(
                                            inputs=FC2,
                                            training=is_training,
                                            name='FC2_bn'
                                        )

                                    # Dropout
                                    if drop_out:
                                        FC2 = tf.layers.dropout(
                                            inputs=FC2,
                                            rate=0.01,
                                            training=is_training,
                                            name='FC2_dropout'
                                        )

                                with tf.name_scope('Reshape'):
                                    # (n*n)x1 -> nxn
                                    FC_M = tf.reshape(FC2, shape=[tf.shape(X)[0], n, n, 1])


                                CONV1 = tf.layers.conv2d(
                                    FC_M,
                                    filters=64,
                                    kernel_size=5,
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    dilation_rate=(1, 1),
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    bias_initializer=None,
                                    #bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='Conv_1',
                                    reuse=tf.AUTO_REUSE
                                )
                                # Activation
                                CONV1 = tf.nn.relu(CONV1)
                                # Batch Norm
                                if batch_normalisation:
                                    CONV1 = tf.layers.batch_normalization(
                                        inputs=CONV1,
                                        training=is_training,
                                        name='CONV1_bn'
                                    )

                                # Dropout
                                if drop_out:
                                    CONV1 = tf.layers.dropout(
                                        inputs=CONV1,
                                        rate=0.01,
                                        training=is_training,
                                        name='CONV1_dropout'
                                    )

                                # (n_im, n, n, 64_results_final_bn_drop_first_layer) -> (n_im, n, n, 64_results_final_bn_drop_first_layer)

                                CONV2 = tf.layers.conv2d(
                                    CONV1,
                                    filters=64,
                                    kernel_size=5,
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    dilation_rate=(1, 1),
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l1_regularizer(0.0001),
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='Conv_2',
                                    reuse=tf.AUTO_REUSE
                                )
                                # Activation
                                CONV2 = tf.nn.relu(CONV2)
                                # Batch Norm
                                if batch_normalisation:
                                    CONV2 = tf.layers.batch_normalization(
                                        inputs=CONV2,
                                        training=is_training,
                                        name='CONV2_bn'
                                    )

                                # Dropout
                                if drop_out:
                                    CONV2 = tf.layers.dropout(
                                        inputs=CONV2,
                                        rate=0.01,
                                        training=is_training,
                                        name='CONV2_dropout'
                                    )

                                DECONV = tf.layers.conv2d_transpose(
                                    inputs=CONV2,
                                    filters=1,
                                    kernel_size=7,
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    trainable=True,
                                    name='Deconv',
                                    reuse=tf.AUTO_REUSE
                                )

                                with tf.name_scope('Y_pred'):
                                    # Removes dimensions of size 1 from the shape of a tensor
                                    DECONV = tf.squeeze(DECONV)

                                return DECONV

                            # Construct model
                            # with tf.name_scope('Automap_Network'):
                            y_recon = automap_network(X, Is_training)

                            # (for tensorboard)
                            W_fc1_summary = tf.summary.histogram("W_fc1", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc1/weights'))
                            W_fc2_summary = tf.summary.histogram("W_fc2", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc2/weights'))
                            W_conv1_summary = tf.summary.histogram("W_conv1", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv_1/kernel'))
                            W_conv2_summary = tf.summary.histogram("W_conv2", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv_2/kernel'))
                            W_deconv_summary = tf.summary.histogram("W_deconv", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Deconv/kernel'))
                            # (for tensorboard)
                            b_fc1_summary = tf.summary.histogram("b_fc1", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc1/bias'))
                            b_fc2_summary = tf.summary.histogram("b_fc2", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc2/bias'))
                            b_conv1_summary = tf.summary.histogram("b_conv1", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv_1/bias'))
                            b_conv2_summary = tf.summary.histogram("b_conv2", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Conv_2/bias'))
                            b_deconv_summary = tf.summary.histogram("b_deconv", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Deconv/bias'))

                            with tf.name_scope('MSE'):
                                mse_op = tf.losses.mean_squared_error(Y, y_recon)

                            with tf.name_scope('Optimizer'):
                                if optimizer != 'rmsprop':
                                    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse_op)
                                else:
                                    if batch_normalisation:
                                        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                                        with tf.control_dependencies(extra_update_ops):
                                            train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=momentum).minimize(mse_op, global_step=tf.train.get_global_step())
                                    else:
                                        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=momentum).minimize(mse_op)

                            # Create a summary to monitor cost tensor (for tensorboard)
                            mse_summary = tf.summary.scalar("mse", mse_op)

                            # Merge all summaries into a single op (for tensorboard)
                            merged_summary_op = tf.summary.merge_all()

                            # For saving the model
                            saver = tf.train.Saver(max_to_keep=1)

                            # For memory
                            config = tf.ConfigProto(log_device_placement=True)
                            config.gpu_options.allow_growth = True
                            #config.gpu_options.allocator_type = 'BFC'

                            # Start training
                            #with tf.device('/gpu:1'):
                             #   ops.reset_default_graph()
                            with tf.Session(config=config) as sess:

                                # Run the initializer
                                sess.run(tf.global_variables_initializer())

                                # (for tensorboard)
                                summary_writer_train = tf.summary.FileWriter(dir_to_save_train, graph=tf.get_default_graph())
                                summary_writer_val = tf.summary.FileWriter(dir_to_save_val, graph=tf.get_default_graph())
                                summary_writer_test = tf.summary.FileWriter(dir_to_save_test, graph=tf.get_default_graph())

                                # (for tensorboard)
                                train_merged = tf.summary.merge([
                                    mse_summary,
                                    W_fc1_summary,
                                    W_fc2_summary,
                                    W_conv1_summary,
                                    W_conv2_summary,
                                    W_deconv_summary,
                                    b_fc1_summary,
                                    b_fc2_summary,
                                    b_conv1_summary,
                                    b_conv2_summary,
                                    b_deconv_summary
                                ])
                                val_merged = tf.summary.merge([
                                    mse_summary
                                ])
                                test_merged = tf.summary.merge([
                                    mse_summary
                                ])

                                epoch = 0
                                is_training = True
                                patience = 20               # how many more epochs are checked after val mse got worse
                                mse_val_lowest = np.inf     # set best mse to infinity at first
                                mse_test_final = 0          # mse of test set of chosen epoch with lowest val mse
                                epoch_with_lowest_val_mse = 0

                                while is_training:
                                    epoch = epoch + 1
                                    batch_counter = 0

                                    # generate a list of random batches that are drawn without replacement
                                    batch_ind_list_train, number_of_batches_train = generateBatchIndeces(x_train, batch_size=32)
                                    batch_ind_list_val, number_of_batches_val = generateBatchIndeces(x_val, batch_size=32)
                                    batch_ind_list_test, number_of_batches_test = generateBatchIndeces(x_test, batch_size=32)

                                    batch_mse_train = []
                                    batch_mse_val = []
                                    batch_mse_test = []

                                    #for indeces_train in batch_ind_list_train:
                                    for i in range(number_of_batches_train):
                                        # batch_x: ?, n, n, 2
                                        # batch_y: ?, n, n
                                        indeces_train = batch_ind_list_train[i]
                                        if noise:
                                            sess.run(train_op, feed_dict={X: multiply_noise(x_train[indeces_train]), Y: y_train[indeces_train], Is_training:True})
                                        else:
                                            sess.run(train_op, feed_dict={X: x_train[indeces_train], Y: y_train[indeces_train], Is_training:True})

                                        batch_counter = batch_counter + 1

                                        batch_mse_train.append(sess.run(mse_op, feed_dict={X: x_train[indeces_train], Y: y_train[indeces_train], Is_training:False}))

                                        #print("Minibatch {} of {}".format(batch_counter, len(batch_ind_list_train)), end="\r")

                                    for i in range(number_of_batches_val):
                                        indeces_val = batch_ind_list_val[i]
                                        batch_mse_val.append(sess.run(mse_op, feed_dict={X: x_val[indeces_val], Y: y_val[indeces_val], Is_training:False}))

                                    for i in range(number_of_batches_test):
                                        indeces_test = batch_ind_list_test[i]
                                        batch_mse_test.append(sess.run(mse_op, feed_dict={X: x_test[indeces_test], Y: y_test[indeces_test], Is_training:False}))

                                    # Get mse Values for printing

                                    mse_train = np.mean(batch_mse_train)
                                    mse_val = np.mean(batch_mse_val)
                                    mse_test = np.mean(batch_mse_test)

                                    print("Setup {} of {}, Epoch {}, train_mse: {}, val_mse: {}, test_mse: {}".format(
                                        setup_couter,
                                        len(batch_normalisation_list)*len(momentum_list)*len(learning_rate_list)*len(noise_list)*len(drop_out_list)*len(optimizer_list),
                                        epoch,
                                        mse_train,
                                        mse_val,
                                        mse_test
                                    ))

                                    # Get Data for Memory Info in Tensorboard
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()

                                    # Get Summaries (for tensorboard)
                                    summary_batch_mse_train = tf.Summary(value=[
                                        tf.Summary.Value(tag="mean_batch_mse", simple_value=np.mean(batch_mse_train)),
                                    ])
                                    summary_batch_mse_val = tf.Summary(value=[
                                        tf.Summary.Value(tag="mean_batch_mse", simple_value=np.mean(batch_mse_val)),
                                    ])
                                    summary_batch_mse_test = tf.Summary(value=[
                                        tf.Summary.Value(tag="mean_batch_mse", simple_value=np.mean(batch_mse_test)),
                                    ])
                                    #train_summary = sess.run(train_merged, feed_dict={X: x_train[0:10], Y: y_train[0:10]}, options=run_options, run_metadata=run_metadata)
                                    #val_summary = sess.run(val_merged, feed_dict={X: x_val[0:10], Y: y_val[0:10]})
                                    #test_summary = sess.run(test_merged, feed_dict={X: x_test[0:10], Y: y_test[0:10]})

                                    # Write Summaries (for tensorboard)
                                    # if epoch % 10 == 0:
                                    summary_writer_train.add_run_metadata(run_metadata, 'step%d' % epoch)

                                    summary_writer_train.add_summary(summary_batch_mse_train, epoch)
                                    summary_writer_val.add_summary(summary_batch_mse_val, epoch)
                                    summary_writer_test.add_summary(summary_batch_mse_test, epoch)

                                    # Early Stopping
                                    if (mse_val_lowest <= mse_val):
                                        # no improvement
                                        print('No Improvement since {} of {} epochs'.format(epoch - epoch_with_lowest_val_mse, patience))
                                        print('')

                                        if epoch >= (epoch_with_lowest_val_mse + patience):
                                            # stop training
                                            is_training = False

                                            # save images
                                            reconstructions_test = sess.run(y_recon, {X: x_test[0:200], Y: y_test[0:200], Is_training:False})

                                            if not os.path.exists('recon_{}'.format(n)):
                                                os.makedirs('recon_{}'.format(n))
                                            if not os.path.exists('recon_{}/{}_{}'.format(n, info, mse_test_final)):
                                                os.makedirs('recon_{}/{}_{}'.format(n, info, mse_test_final))

                                            for k in range(np.minimum(200, y_test.shape[0])):
                                                denormalized_y_true = (y_test[k] + 1) * 255 / 2
                                                Image.fromarray(np.uint8(denormalized_y_true)).save(
                                                    'recon_{}/{}_{}/{}_true.jpg'.format(n, info, mse_test_final, k))

                                                img = Image.new('L', (n, n))
                                                pixel = img.load()
                                                sample = reconstructions_test[k]

                                                for y in range(sample.shape[0]):
                                                    for x in range(sample.shape[1]):
                                                        denormalized_y_pred = (sample[y, x] + 1) * 255 / 2
                                                        pixel[x, y] = int(denormalized_y_pred)

                                                img.save('recon_{}/{}_{}/{}_recon.jpg'.format(n, info, mse_test_final, k))


                                    else:
                                        # improvement
                                        epoch_with_lowest_val_mse = epoch
                                        mse_val_lowest = mse_val
                                        mse_test_final = mse_test
                                        saver.save(sess, os.path.join(path_saving, 'model_{}'.format(info)), global_step=epoch)

                                print("Optimization Finished!")
                                print("Epochs: {}, mse test: {}".format(epoch_with_lowest_val_mse, mse_test_final))

