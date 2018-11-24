import images
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def get_distance_to_closest(A, B_list):
    """
    :param A: Point
    :param B_list: list of Points
    :return: shortest distance from A to any element of B_list
    """

    distances = [((A[0]-B[0])**2 + (A[1]-B[1])**2)**0.5 for B in B_list]
    val, idx = min((val, idx) for (idx, val) in enumerate(distances))
    return val, idx


def evaluate_transform(x, *args):
    """
    Evaluates the score of the transformation x, transforming points
    from args[1] to match args[0].
    :param x: parameters of the transformation matrix. Transformation
        matrix created as follows:
            T = np.array([x[0:3], x[3:6], np.append(x[6:8], 1)])
    :param args:
        args[0] should be the target coordinates, np.array sized 3xN, N being
            number of points, first row being the X image coordinates,
            second row being Y image coordinates, third row being ones (1)
        args[1] should be the original coordinates, np. array sized 3xM, M
            being number of points, first row being longitude, second row
            being latitude, third row being ones (1)
    :return:
        score being related to distance from target coordinates to transformed
            original coordinates
    """
    image_pixel_ids = args[0]
    image_coordinates = args[1]

    T = np.array([x[0:3], x[3:6], np.append(x[6:8], 1)])

    print("T")
    print(T)

    transformed_coordinates = T.dot(image_coordinates)

    for col in range(transformed_coordinates.shape[1]):
        transformed_coordinates[:, col] = transformed_coordinates[:, col] / transformed_coordinates[-1, col]

    score = 0

    transformed_items = transformed_coordinates.transpose().tolist()

    for pix in image_pixel_ids.transpose().tolist():
        val, index = get_distance_to_closest(pix, transformed_items)
        score += val**3 # works best with **3
        del transformed_items[index]

    return score


ARCHIVE = "data/S2B_MSIL1C_20181018T100029_N0206_R122_T34UCF_20181018T120913.zip"

image = images.get_tci_image(ARCHIVE)
coords = images.get_coordinates(images.get_manifest(ARCHIVE))
vertices = np.array([coords[0][:-1],
                     coords[1][:-1],
                     (len(coords[0])-1)*[1]])

image_pixels = image.load()


def scan_by_column(scan_range, row_id, image_boundries, nan_pixel = (0, 0, 0)):

    for col in scan_range:
        if image_pixels[col, row_id] != nan_pixel:
            if (col, row_id) not in image_boundries:
                image_boundries.append((col, row_id))
            break


def get_image_boundries(image_pixels, image_size):
    image_boundries = []
    row_zero = 0
    row_last = image_size[0]-1
    nan_pixel = (0, 0, 0)

    for col in range(image_size[1]):
        if image_pixels[col, row_zero] != nan_pixel:
            if (col, row_zero) not in image_boundries:
                image_boundries.append((col, row_zero))
            break

    for col in range(image_size[1]):
        if image_pixels[col, row_last] != nan_pixel:
            if (col, row_last) not in image_boundries:
                image_boundries.append((col, row_last))
            break

    for col in range(image_size[1]-1, -1, -1):
        if image_pixels[col, row_zero] != nan_pixel:
            if (col, row_zero) not in image_boundries:
                image_boundries.append((col, row_zero))
            break

    for col in range(image_size[1]-1, -1, -1):
        if image_pixels[col, row_last] != nan_pixel:
            if (col, row_last) not in image_boundries:
                image_boundries.append((col, row_last))
            break

    col_zero = 0
    col_last = image_size[1]-1

    for row in range(image_size[0]):
        if image_pixels[col_zero, row] != nan_pixel:
            if (col_zero, row) not in image_boundries:
                image_boundries.append((col_zero, row))
            break

    for row in range(image_size[0]):
        if image_pixels[col_last, row] != nan_pixel:
            if (col_last, row) not in image_boundries:
                image_boundries.append((col_last, row))
            break

    for row in range(image_size[0]-1, -1, -1):
        if image_pixels[col_zero, row] != nan_pixel:
            if (col_zero, row) not in image_boundries:
                image_boundries.append((col_zero, row))
            break

    for row in range(image_size[0]-1, -1, -1):
        if image_pixels[col_last, row] != nan_pixel:
            if (col_last, row) not in image_boundries:
                image_boundries.append((col_last, row))
            break

    return np.vstack((np.array(image_boundries).transpose(), len(image_boundries)*[1]))


image_boundries = get_image_boundries(image_pixels, image.size)

plt.imshow(image)
plt.plot(image_boundries[0, :], image_boundries[1, :], 'o')
plt.show()

print(vertices)
print(image_boundries)

zero_node = [min(vertices[0, :]), max(vertices[1, :])]

T_zero = np.array([[1, 0, -zero_node[0]],
                   [0, -1, zero_node[1]],
                   [0, 0, 1]])

vert_zero = T_zero.dot(vertices)

vertices_range = np.max(vert_zero, axis=1) - np.min(vert_zero, axis=1)

scaling = [image.size[0] / vertices_range[0],
           image.size[1] / vertices_range[1]]

T_scale = np.array([[scaling[0], 0, 0],
                    [0, scaling[1], 0],
                    [0, 0, 1]])

plt.plot(T_scale.dot(vert_zero)[0, :], T_scale.dot(vert_zero)[1, :], 'o-')
plt.plot(image_boundries[0, :], image_boundries[1, :], 'o-')
plt.show()

T_initial = T_scale.dot(T_zero)

methods = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov'
]

results = {}

method = 'BFGS'

print("image_boundries")
print(image_boundries)

print("vertices")
print(vertices)


res = optimize.minimize(
    evaluate_transform,
    np.array([T_initial[0, 0], T_initial[0, 1], T_initial[0, 2],
              T_initial[1, 0], T_initial[1, 1], T_initial[1, 2],
              T_initial[2, 0], T_initial[2, 1]]),
    args=(image_boundries, vertices),
    method=method
)

        # results[method] = [res, SCORE[-1]]

    # except:
    #     pass

# min_score = float('inf')
# res = None
# for method in results:
#     print(f"{method}: {results[method][1]}")
#     if results[method][1] < min_score:
#         min_score = results[method][1]
#         res = results[method][0]


T_2 = np.array([res.x[0:3], res.x[3:6], np.append(res.x[6:8], 1)])

# print("T - res.x")
# print(T - T_2)

transformed_coordinates = T_2.dot(vertices)

print("transformed_coordinates:")
print(transformed_coordinates)

for col in range(transformed_coordinates.shape[1]):
    transformed_coordinates[:, col] = transformed_coordinates[:, col] / transformed_coordinates[-1, col]

print("transformed_coordinates (normalized):")
print(transformed_coordinates)

plt.plot(image_boundries[0, :], image_boundries[1, :], 'o-')
plt.plot(transformed_coordinates[0, :], transformed_coordinates[1, :], 'o-')
# plt.plot(transformed_vertices[0, :], transformed_vertices[1, :], 'o-')
# plt.plot(T.dot(vertices)[0, :], T.dot(vertices)[1, :], 'o-')
plt.show()





