import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import time

def undersample(x, pattern):
    n_samples = pattern.size
    x_undersampled = np.zeros((x.shape[0], n_samples, x.shape[3]))

    for img in range(x.shape[0]):
        #print('subsample {} of {}'.format(img+1, x.shape[0]))
        x_vec_real = np.reshape(x[img,:,:,0], newshape=(x.shape[1]*x.shape[2]))
        x_vec_imag = np.reshape(x[img,:,:,1], newshape=(x.shape[1]*x.shape[2]))
        x_undersampled[img,:,0] = x_vec_real[pattern]
        x_undersampled[img,:,1] = x_vec_imag[pattern]

    return x_undersampled

# img width and height
n = 64

# directory containing features and labels
path_data = ''
path_logs = 'logs/'
path_saving = 'saved_model_{}/'.format(n)

# use the file name of the meta file in saved model directory
saved_model_meta_info = 'model_mom:0.8_lr:2e-05_noise:False_bn:False_drop:True-14.meta'
normalization_info = 'lin_pix_min_max'
pattern_info = 'pattern_N64_red2.0_varDens'
pattern = np.loadtxt('pattern/{}'.format(pattern_info)).astype('int16')
pattern -= 1

# load input and output data
x = np.load(path_data + 'x_{}_{}.npy'.format(n, normalization_info))
y = np.load(path_data + 'y_{}.npy'.format(n))

# devide train_temp into val and train
# 15% test
# 15% val (0.85*x = 0.15 => x=0.15/0.85)=17.65
# 70% train

# identical random seed is set to get the same train test split
x_train_temp, x_test, y_train_temp, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1765, random_state=1)

x_train = undersample(x=x_train, pattern=pattern)
x_val = undersample(x=x_val, pattern=pattern)
x_test = undersample(x=x_test, pattern=pattern)

del x_train_temp, y_train_temp, x, y

# with tf.device('/gpu:0'):
#     ops.reset_default_graph()
with tf.Session() as sess:

    #new_saver = tf.train.import_meta_graph(os.path.join(path_saving, 'model_{}-{}.meta'.format(pattern_info, epoch)))
    new_saver = tf.train.import_meta_graph(os.path.join(path_saving, saved_model_meta_info),clear_devices=True)
    new_saver.restore(sess, tf.train.latest_checkpoint(path_saving))

    graph = tf.get_default_graph()

    # restore placeholders
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    Is_training = graph.get_tensor_by_name("Placeholder:0")

    # Now, access the op that you want to run.
    y_recon = graph.get_tensor_by_name("Y_pred/Squeeze:0")

    mse_op = tf.losses.mean_squared_error(Y, y_recon)
    mse_train_op = tf.losses.mean_squared_error(y_train, y_recon)
    mse_val_op = tf.losses.mean_squared_error(y_val, y_recon)
    mse_test_op = tf.losses.mean_squared_error(y_test, y_recon)

    # reconstructions_test = sess.run(y_recon, {X: x_test, Y: y_test, Is_training: False})
    print("mse train: {}".format(sess.run(mse_train_op, {X: x_train, Y: y_train, Is_training: False})))
    # print("mse val: {}".format(sess.run(mse_val_op, {X: x_val, Y: y_val})))
    # print("mse test: {}".format(sess.run(mse_test_op, {X: x_test, Y: y_test})))

    start = time.time()
    reconstructions_train = sess.run(y_recon, {X: x_train, Y: y_train, Is_training: False})
    end = time.time()
    print(end - start)

    start = time.time()
    reconstructions_val = sess.run(y_recon, {X: x_val, Y: y_val, Is_training: False})
    end = time.time()
    print(end - start)

    start = time.time()
    reconstructions_test = sess.run(y_recon, {X: x_test, Y: y_test, Is_training: False})
    end = time.time()
    print(end - start)

if not os.path.exists('recon_{}_{}'.format(n,pattern_info)):
    os.makedirs('recon_{}_{}'.format(n,pattern_info))


def save_true_and_recon_imgs(y_true, y_recon, name):

    for k in range(y_true.shape[0]):
        denormalized_y_true = (y_true[k] + 1) * 255 / 2
        Image.fromarray(np.uint8(denormalized_y_true)).save('recon_{}_{}/{}_{}_true.jpg'.format(n, pattern_info, name, k))

        img = Image.new('L', (n, n))

        pixel = img.load()

        sample = y_recon[k]

        for y in range(sample.shape[0]):
            for x in range(sample.shape[1]):
                denormalized_y_pred = (sample[y, x] + 1) * 255 / 2
                pixel[x, y] = int(denormalized_y_pred)

        img.save('recon_{}_{}/{}_{}_recon.jpg'.format(n, pattern_info, name, k))

save_true_and_recon_imgs(y_true=y_train, y_recon=reconstructions_train, name='train')
save_true_and_recon_imgs(y_true=y_val, y_recon=reconstructions_val, name='val')
save_true_and_recon_imgs(y_true=y_test, y_recon=reconstructions_test, name='test')
