import os
from PIL import Image
import numpy as np

# groundtruth size  n = img.w = img.h
n = 64

normalization_info = 'lin_min_max'
normalization_info = 'log_min_max'

normalization_info = 'lin_mean_std'
normalization_info = 'log_mean_std'

normalization_info = 'lin_pix_min_max'
normalization_info = 'log_pix_min_max'

normalization_info = 'lin_pix_mean_std'
normalization_info = 'log_pix_mean_std'

normalization_info = 'lin_min_max_sep'
normalization_info = 'log_min_max_sep'

normalization_info = 'lin_mean_std_sep'
normalization_info = 'log_mean_std_sep'

normalization_info = 'lin_pix_min_max_sep'
normalization_info = 'log_pix_min_max_sep'

normalization_info = 'lin_pix_mean_std_sep'
normalization_info = 'log_pix_mean_std_sep'

# paths
path_raw = 'imgs_raw'

def get_n_img(path_raw):
    # counts number of images and number of images that have w and h >= 256
    n_img = 0

    for dir in os.listdir(path_raw):
        dir_count = 0
        if not dir.startswith(".") and not dir.startswith("_"):
            for file_name in os.listdir(os.path.join(path_raw, dir)):
                if file_name.endswith(".jpg"):
                    img = Image.open(os.path.join(os.path.join(path_raw, dir), file_name)).convert('L')
                    if img.width >= n and img.height >= n:
                        n_img += 1
                        dir_count += 1
        print('{} in {}'.format(dir_count, dir))
    return n_img

def get_process_input_and_output(n_img, path_raw):

    angles = [0, 90, 180, 270] # angles for augmenting the dataset

    # outputs ground truth y
    y = np.empty((n_img * len(angles), n, n), dtype='float32')
    # inputs k-space x
    x_lin = np.empty((n_img * len(angles), n, n, 2), dtype='float32') # 2 for real/imag
    x_log = np.empty((n_img * len(angles), n, n, 2), dtype='float32') # 2 for real/imag

    image_counter = 0
    for dir in os.listdir(path_raw):
        if not dir.startswith(".") and not dir.startswith("_"):
            for file_name in os.listdir(os.path.join(path_raw, dir)):
                if file_name.endswith(".jpg"):

                    # open image with gray scale intensities from 0-255
                    img = Image.open(os.path.join(os.path.join(path_raw, dir), file_name)).convert('L')

                    # only consider images that are big enough
                    if img.width >= n and img.height >= n:

                        smallest_side = np.minimum(img.width, img.height)

                        # size 0 = width, size 1 = height
                        box_left = ((img.size[0] / 2) - (smallest_side / 2))
                        box_upper = ((img.size[1] / 2) - (smallest_side / 2))
                        box_right = ((img.size[0] / 2) + (smallest_side / 2))
                        box_lower = ((img.size[1] / 2) + (smallest_side / 2))

                        img = img.crop([
                            box_left if box_left % 1 == 0 else box_left + 0.5,
                            box_upper if box_upper % 1 == 0 else box_upper + 0.5,
                            box_right if box_right % 1 == 0 else box_right + 0.5,
                            box_lower if box_lower % 1 == 0 else box_lower + 0.5,
                        ])

                        for angle in angles:
                            new_img = img.copy()

                            # rotate image
                            new_img = new_img.rotate(angle)

                            # subsample image from 256x256 to nxn
                            new_img = new_img.resize((n, n), Image.ANTIALIAS)

                            # image to numpy array
                            pixel = np.array(new_img).astype('float32')

                            # normalize ground truth
                            pixel_normalized = pixel * 2 / 255 - 1

                            # generate output y[image_counter, :, :]
                            y[image_counter, :, :] = pixel_normalized

                            # path for saving ground truth
                            dir_to_save = 'ground_truth_{}'.format(n)

                            # create folder if none exists
                            if not os.path.exists(dir_to_save):
                                os.makedirs(dir_to_save)

                            # save ground truth image
                            new_img.save(os.path.join(dir_to_save, '{}_{}_{}'.format(image_counter, angle, file_name)))

                            # generate input x
                            img_f = np.fft.fft2(pixel)   # FFT
                            img_f_log = img_f / np.power(abs(img_f)+1, 1/np.exp(1))
                            if np.isinf(img_f_log).any():
                                print()
                            if np.isnan(img_f_log).any():
                                print()

                            # FFT shift
                            img_fshift = np.fft.fftshift(img_f)
                            img_fshift_log = np.fft.fftshift(img_f_log)

                            img_real = img_fshift.real          # Real part: (im_size1, im_size2)
                            img_imag = img_fshift.imag          # Imaginary part: (im_size1, im_size2)
                            img_real_log = img_fshift_log.real
                            img_imag_log = img_fshift_log.imag

                            # Merge Real Imag
                            img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)
                            img_real_imag = np.expand_dims(img_real_imag, axis=0) # (1, im_size1, im_size2, 2)
                            img_real_imag_log = np.dstack((img_real_log, img_imag_log))  # (im_size1, im_size2, 2)
                            img_real_imag_log = np.expand_dims(img_real_imag_log, axis=0)  # (1, im_size1, im_size2, 2)

                            # x is k-space data for images
                            x_lin[image_counter, :, :, :] = img_real_imag
                            x_log[image_counter, :, :, :] = img_real_imag_log

                            if np.isinf(img_real_imag_log).any():
                                print()
                            if np.isnan(img_real_imag_log).any():
                                print()

                            image_counter += 1
                            print('{} of {}'.format(image_counter, len(angles)*n_img))

    np.save('y_{}'.format(n), y)

    def norm_min_max(x):
        return -1 + (x - np.amin(x)) * 2 / (np.amax(x) - np.amin(x))

    def norm_min_max_pixelwise(x):
        # avoid devision by zero
        if np.amax(x) - np.amin(x) == 0:
            return 0
        else:
            return -1 + (x - np.amin(x)) * 2 / (np.amax(x) - np.amin(x))

    def norm_mean_std(x):
        # avoid devision by zero
        std = np.maximum(np.std(x), 1 / np.sqrt(x.size))
        return (x - np.mean(x)) / std

    # MIN MAX - Normalize lin k-space input x to -1 to 1
    x_lin_min_max = np.copy(x_lin)
    x_lin_min_max = norm_min_max(x_lin)
    assert np.amax(x_lin_min_max) == 1 and np.amin(x_lin_min_max) == -1
    np.save('x_{}_lin_min_max'.format(n), x_lin_min_max)

    x_log_min_max = np.copy(x_log)
    x_log_min_max = norm_min_max(x_log)
    assert np.amax(x_log_min_max) == 1 and np.amin(x_log_min_max) == -1
    np.save('x_{}_log_min_max'.format(n), x_log_min_max)

    x_lin_min_max_sep = np.copy(x_lin)
    x_lin_min_max_sep[:,:,:,0] = norm_min_max(x_lin[:,:,:,0])
    x_lin_min_max_sep[:,:,:,1] = norm_min_max(x_lin[:,:,:,1])
    assert np.amax(x_lin_min_max_sep[:, :, :, 0]) == 1 and np.amin(x_lin_min_max_sep[:, :, :, 0]) == -1
    assert np.amax(x_lin_min_max_sep[:, :, :, 1]) == 1 and np.amin(x_lin_min_max_sep[:, :, :, 1]) == -1
    np.save('x_{}_lin_min_max_sep'.format(n), x_lin_min_max_sep)

    x_log_min_max_sep = np.copy(x_log)
    x_log_min_max_sep[:, :, :, 0] = norm_min_max(x_log[:, :, :, 0])
    x_log_min_max_sep[:, :, :, 1] = norm_min_max(x_log[:, :, :, 1])
    assert np.amax(x_log_min_max_sep[:, :, :, 0]) == 1 and np.amin(x_log_min_max_sep[:, :, :, 0]) == -1
    assert np.amax(x_log_min_max_sep[:, :, :, 1]) == 1 and np.amin(x_log_min_max_sep[:, :, :, 1]) == -1
    np.save('x_{}_log_min_max_sep'.format(n), x_log_min_max_sep)



    # MEAN STD - Normalize lin k-space input x to mean 0 std 1
    x_lin_mean_std = np.copy(x_lin)
    x_lin_mean_std = norm_mean_std(x_lin)
    assert np.isclose(np.mean(x_lin_mean_std), 0) and np.isclose(np.std(x_lin_mean_std), 1)
    np.save('x_{}_lin_mean_std'.format(n), x_lin_mean_std)

    x_log_mean_std = np.copy(x_log)
    x_log_mean_std = norm_mean_std(x_log)
    assert np.isclose(np.mean(x_log_mean_std), 0) and np.isclose(np.std(x_log_mean_std), 1)
    np.save('x_{}_log_mean_std'.format(n), x_log_mean_std)

    x_lin_mean_std_sep = np.copy(x_lin)
    x_lin_mean_std_sep[:,:,:,0] = norm_mean_std(x_lin[:,:,:,0])
    x_lin_mean_std_sep[:,:,:,1] = norm_mean_std(x_lin[:,:,:,1])
    assert np.isclose(np.mean(x_lin_mean_std_sep[:,:,:,0]), 0) and np.isclose(np.std(x_lin_mean_std_sep[:,:,:,0]), 1)
    assert np.isclose(np.mean(x_lin_mean_std_sep[:,:,:,1]), 0) and np.isclose(np.std(x_lin_mean_std_sep[:,:,:,1]), 1)
    np.save('x_{}_lin_mean_std_sep'.format(n), x_lin_mean_std_sep)

    x_log_mean_std_sep = np.copy(x_log)
    x_log_mean_std_sep[:,:,:,0] = norm_mean_std(x_log[:,:,:,0])
    x_log_mean_std_sep[:,:,:,1] = norm_mean_std(x_log[:,:,:,1])
    assert np.isclose(np.mean(x_log_mean_std_sep[:, :, :, 0]), 0) and np.isclose(np.std(x_log_mean_std_sep[:, :, :, 0]),1)
    assert np.isclose(np.mean(x_log_mean_std_sep[:, :, :, 1]), 0) and np.isclose(np.std(x_log_mean_std_sep[:, :, :, 1]),1)
    np.save('x_{}_log_mean_std_sep'.format(n), x_log_mean_std_sep)

    # Normalize lin/log k-space input x of each pixel to -1/1
    x_lin_pix_min_max = np.copy(x_lin)
    x_log_pix_min_max = np.copy(x_log)
    x_lin_pix_min_max_sep = np.copy(x_lin)
    x_log_pix_min_max_sep = np.copy(x_log)

    x_lin_pix_mean_std = np.copy(x_lin)
    x_log_pix_mean_std = np.copy(x_log)
    x_lin_pix_mean_std_sep = np.copy(x_lin)
    x_log_pix_mean_std_sep = np.copy(x_log)

    for w in range(n):
        print('{} of {}'.format(w+1, n))
        for h in range(n):

            x_lin_pix_min_max[:, w, h, :] = norm_min_max(x_lin[:, w, h, :])
            x_lin_pix_min_max_sep[:, w, h, 0] = norm_min_max_pixelwise(x_lin[:, w, h, 0])
            x_lin_pix_min_max_sep[:, w, h, 1] = norm_min_max_pixelwise(x_lin[:, w, h, 1])

            x_log_pix_min_max[:, w, h, :] = norm_min_max(x_log[:, w, h, :])
            x_log_pix_min_max_sep[:, w, h, 0] = norm_min_max_pixelwise(x_log[:, w, h, 0])
            x_log_pix_min_max_sep[:, w, h, 1] = norm_min_max_pixelwise(x_log[:, w, h, 1])

            x_lin_pix_mean_std[:, w, h, :] = norm_mean_std(x_lin[:, w, h, :])
            x_lin_pix_mean_std_sep[:, w, h, 0] = norm_mean_std(x_lin[:, w, h, 0])
            x_lin_pix_mean_std_sep[:, w, h, 1] = norm_mean_std(x_lin[:, w, h, 1])

            x_log_pix_mean_std[:, w, h, :] = norm_mean_std(x_log[:, w, h, :])
            x_log_pix_mean_std_sep[:, w, h, 0] = norm_mean_std(x_log[:, w, h, 0])
            x_log_pix_mean_std_sep[:, w, h, 1] = norm_mean_std(x_log[:, w, h, 1])


    assert np.amax(x_lin_pix_min_max) == 1 and np.amin(x_lin_pix_min_max) == -1
    assert np.amax(x_log_pix_min_max) == 1 and np.amin(x_log_pix_min_max) == -1
    assert np.amax(x_lin_pix_min_max_sep[:, :, :, 0]) == 1 and np.amin(x_lin_pix_min_max_sep[:, :, :, 0]) == -1
    assert np.amax(x_lin_pix_min_max_sep[:, :, :, 1]) == 1 and np.amin(x_lin_pix_min_max_sep[:, :, :, 1]) == -1
    assert np.amax(x_log_pix_min_max_sep[:, :, :, 0]) == 1 and np.amin(x_log_pix_min_max_sep[:, :, :, 0]) == -1
    assert np.amax(x_log_pix_min_max_sep[:, :, :, 1]) == 1 and np.amin(x_log_pix_min_max_sep[:, :, :, 1]) == -1

    np.save('x_{}_lin_pix_min_max'.format(n), x_lin_pix_min_max)
    np.save('x_{}_log_pix_min_max'.format(n), x_log_pix_min_max)
    np.save('x_{}_lin_pix_min_max_sep'.format(n), x_lin_pix_min_max_sep)
    np.save('x_{}_log_pix_min_max_sep'.format(n), x_log_pix_min_max_sep)


    assert np.isclose(np.mean(x_lin_pix_mean_std), 0, 1e-1) and np.isclose(np.std(x_lin_pix_mean_std),1, 1e-1)
    assert np.isclose(np.mean(x_log_pix_mean_std), 0, 1e-1) and np.isclose(np.std(x_log_pix_mean_std),1, 1e-1)
    assert np.isclose(np.mean(x_lin_pix_mean_std_sep[:, :, :, 0]), 0, 1e-1) and np.isclose(np.std(x_lin_pix_mean_std_sep[:, :, :, 0]),1, 1e-1)
    assert np.isclose(np.mean(x_lin_pix_mean_std_sep[:, :, :, 1]), 0, 1e-1) and np.isclose(np.std(x_lin_pix_mean_std_sep[:, :, :, 1]),1, 1e-1)
    assert np.isclose(np.mean(x_log_pix_mean_std_sep[:, :, :, 0]), 0, 1e-1) and np.isclose(np.std(x_log_pix_mean_std_sep[:, :, :, 0]),1, 1e-1)
    assert np.isclose(np.mean(x_log_pix_mean_std_sep[:, :, :, 1]), 0, 1e-1) and np.isclose(np.std(x_log_pix_mean_std_sep[:, :, :, 1]),1, 1e-1)

    np.save('x_{}_lin_pix_mean_std'.format(n), x_lin_pix_mean_std)
    np.save('x_{}_log_pix_mean_std'.format(n), x_log_pix_mean_std)
    np.save('x_{}_lin_pix_mean_std_sep'.format(n), x_lin_pix_mean_std_sep)
    np.save('x_{}_log_pix_mean_std_sep'.format(n), x_log_pix_mean_std_sep)

n_img = get_n_img(path_raw=path_raw)
print('{} raw images'.format(n_img))

get_process_input_and_output(n_img=n_img, path_raw=path_raw)
print('done')

def normalize_value(val, val_min, val_max, norm_min=0, norm_max=1):
    return norm_min + ((val - val_min) * (norm_max - norm_min)) / (val_max - val_min)

def denormalize_value(val, val_min, val_max, norm_min, norm_max):
    return (val - norm_min) * (val_max - val_min) / (norm_max - norm_min) + val_min