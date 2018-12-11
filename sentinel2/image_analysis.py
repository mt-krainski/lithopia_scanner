import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.misc import imresize, imsave
import imageio
import skimage
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import signal

WINDOW = 1
CHANNELS = 3

def find_vector_set(diff_image, new_size):
    i = 0
    j = 0
    vector_set = np.zeros((int((new_size[0] * new_size[1]) / WINDOW*WINDOW), WINDOW*WINDOW*CHANNELS))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block = diff_image[j:(j+WINDOW), k:(k+WINDOW)]
                feature = block.ravel()
                if block.shape != (WINDOW,WINDOW,CHANNELS):
                    # print("Skipped!")
                    k = k + WINDOW
                    continue
                # print(f"block.shape: {block.shape}")
                # print(f"feature.shape: {feature.shape}")
                vector_set[i, :] = feature
                k = k + WINDOW
            j = j + WINDOW
        i = i + 1

    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec


def find_FVS(EVS, diff_image, mean_vec, new):
    i = 2
    feature_vector_set = []
    LOWER = int(np.floor(WINDOW/2))
    UPPER = int(np.ceil(WINDOW/2))

    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[(i-LOWER):(i+UPPER), (j-LOWER):(j+UPPER)]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS


def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (int(new[0] - 4), int(new[1] - 4)))
    return least_index, change_map


def find_PCAKmeans(imagepath1, imagepath2):
    image1 = imageio.imread(imagepath1, pilmode='RGB')
    image2 = imageio.imread(imagepath2, pilmode='RGB')

    plt.imshow(image2)
    plt.show()
    new_size = np.asarray(image1.shape) / WINDOW * WINDOW
    image1 = skimage.transform.resize(image1, (new_size))#.astype(np.int16)
    image2 = skimage.transform.resize(image2, (new_size))#.astype(np.int16)


    diff_image = abs(image1 - image2)
    plt.imshow(diff_image)
    plt.show()
    imageio.imwrite('diff.png', diff_image)

    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    change_map = change_map.astype(np.uint8)
    # kernel = np.asarray(((0, 1, 0),
    #                      (1, 1, 1),
    #                      (0, 1, 1)), dtype=np.uint8)
    # cleanChangeMap = cv2.erode(change_map, kernel)
    imageio.imwrite("changemap.png", change_map)
    # imageio.imwrite("cleanchangemap.png", cleanChangeMap)


SCORE_TRAJECTORY = []
def score(x, *args):
    im_A = args[0]
    im_B = args[1]
    BORDER = 2
    print(x)
    M = np.float32([[1, 0, x[0]], [0, 1, x[1]]])
    rows, cols, ch = im_B.shape
    im_A = im_A[BORDER:rows-BORDER, BORDER:cols-BORDER]
    dst = cv2.warpAffine(im_B, M, (cols, rows))
    dst = dst[BORDER:rows-BORDER, BORDER:cols-BORDER]
    diff = cv2.absdiff(im_A, dst)
    res = np.sum(diff)
    print(res)
    SCORE_TRAJECTORY.append(res)
    return res


def get_diff_map(image_A, image_B, range=5):
    STEP = 0.1

    rows, cols, ch = image_A.shape
    im_A = image_A[range:rows - range, range:cols - range]
    # res_map = []
    res_dict = []
    for x in np.arange(-range, range, STEP):
        # res_map.append([])
        for y in np.arange(-range, range, STEP):
            M = np.float32([[1, 0, x], [0, 1, y]])
            transformed = cv2.warpAffine(image_B, M, (cols, rows))
            im_B = transformed[range:rows - range, range:cols - range]
            diff = cv2.absdiff(im_A, im_B)
            # res_map[-1].append(np.sum(diff))
            res_dict.append({
                'x': x,
                'y': y,
                'score': np.sum(diff)
            })

    return res_dict


def get_offset_minimize(orignal, secondary):
    res = optimize.minimize(
        score,
        np.array((0, 0)),
        args=(orignal, secondary),
        method='Powell'
    )
    return res.x

def get_offset(original, secondary):
    score_dict = get_diff_map(original, secondary)
    min_value = int(np.argmin([x['score'] for x in score_dict]))
    return score_dict[min_value]['x'], score_dict[min_value]['y']


def offset_image(image, offset):
    try:
        rows, cols = image.shape
    except ValueError:
        rows, cols, ch = image.shape

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    return cv2.warpAffine(image, M, (cols, rows))


def get_image_difference(imageA, imageB):
    return cv2.absdiff(imageA, imageB)


if __name__ == "__main__":
    a = r'C:\Users\pc1\Documents\Projects\lithopia_server\lithopia_server\request_images\S2A_MSIL1C_20181112T082151_N0207_R121_T36SXA_20181112T103700.png'
    b = r'C:\Users\pc1\Documents\Projects\lithopia_server\lithopia_server\request_images\S2A_MSIL1C_20181102T082101_N0206_R121_T36SXA_20181102T094559.png'
    # find_PCAKmeans(a, b)
    im_a = cv2.imread(a)
    im_b = cv2.imread(b)
    offset = get_offset(im_a, im_b)

    transformed = offset_image(im_b, offset)
    diff = get_image_difference(im_a, transformed)
    plt.imshow(diff)
    plt.show()

    print("Done")