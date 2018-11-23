import images
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

ARCHIVE = "data/S2B_MSIL1C_20181117T100259_N0207_R122_T33UVR_20181117T120601.zip"

image = images.get_tci_image(ARCHIVE)
coords = images.get_coordinates(images.get_manifest(ARCHIVE))
print(coords)

SAMPLE_LOCATION = (50.083333, 14.416667)

proper_image = None

if len(coords[0]) == 5:
    print("Proper image")
    proper_image = True
else:
    print("Improper image")
    proper_image = False


def get_distance_to_closest(A, B_list):
    """
    :param A: Point
    :param B_list: list of Points
    :return: shortest distance from A to any element of B_list
    """

    distances = [((A[0]-B[0])**2 + (A[1]-B[1])**2)**0.5 for B in B_list]

    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    return val, idx


SCORE = []

def transform_evaluate(x, *args):
    global SCORE
    image_pixel_ids = args[0]
    image_coordinates = args[1]

    T = np.array([x[0:3], x[3:6], np.append(x[6:8], 1)])

    print("T")
    print(T)

    transformed_coordinates = T.dot(image_coordinates)

    for col in range(transformed_coordinates.shape[1]):
        transformed_coordinates[:, col] = transformed_coordinates[:, col] / transformed_coordinates[-1, col]

    # print("image_pixel_ids")
    # print(image_pixel_ids)
    # print("image_coordinates")
    # print(image_coordinates)
    # print("transformed_coordinates")
    # print(transformed_coordinates)

    score = 0

    transformed_items = transformed_coordinates.transpose().tolist()

    for pix in image_pixel_ids.transpose().tolist():
        val, index = get_distance_to_closest(pix, transformed_items)
        score += val**3
        del transformed_items[index]

    # print(f"score: {score}")

    SCORE.append(score)

    return score


def initial_transform_evaluate(x, *args):
    image_pixel_ids = args[0]
    image_coordinates = args[1]

    T = np.array([[x[0], 0, x[1]], [0, x[2], x[3]], [0, 0, 1]])

    print("T")
    print(T)

    transformed_coordinates = T.dot(image_coordinates)

    print("image_pixel_ids")
    print(image_pixel_ids)
    print("image_coordinates")
    print(image_coordinates)
    print("transformed_coordinates")
    print(transformed_coordinates)

    score = 0

    transformed_items = transformed_coordinates.transpose().tolist()

    for pix in image_pixel_ids.transpose().tolist():
        val, index = get_distance_to_closest(pix, transformed_items)
        score += val
        del transformed_items[index]

    print(f"score: {score}")

    return 1000*score


vertices = np.array([coords[0][:-1], coords[1][:-1], (len(coords[0])-1)*[1]])

if proper_image:
    # plt.plot(coords[0][:-1], coords[1][:-1], 'o-')
    # plt.plot((bounds['west'], bounds['west'], bounds['east'], bounds['east'], bounds['west']),
    #          (bounds['north'], bounds['south'], bounds['south'], bounds['north'], bounds['north']), 'o-')
    #
    # plt.show()

    max_size = image.size[0]
    image_boundries = np.array(
        [[0, 0, max_size, max_size],
         [max_size, 0, 0, max_size],
         [1, 1, 1, 1]])
    # transformed_vertices = np.array((
    #     [bounds['west'], bounds['east'], bounds['east'], bounds['west']],
    #     [bounds['north'], bounds['north'], bounds['south'], bounds['south']],
    #     4*[1],
    # ))


    print(vertices)
    print(image_boundries)

    T = image_boundries.dot(np.linalg.pinv(vertices))
    # T = vertices.dot(np.linalg.pinv(transformed_vertices))

    print("T")
    print(T)

    print("T * vertices")
    print(T.dot(vertices))

    # T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    res = optimize.minimize(
        initial_transform_evaluate,
        np.array([5000, 0, -5000, 0]),
        args=(image_boundries, vertices)
    )

    # COBYLA yielded some different results
    # SLSQP yielded some different results

    print("res.x ")
    # T_2 = np.array([res.x[0:3], res.x[3:6], np.append(res.x[6:8], 1)])
    T_2 = np.array([[res.x[0], 0, res.x[1]], [0, res.x[2], res.x[3]], [0, 0, 1]])
    print(T_2)

    plt.plot(image_boundries[0, :], image_boundries[1, :], 'o-')
    plt.plot(T_2.dot(vertices)[0, :], T_2.dot(vertices)[1, :], 'o-')
    # plt.plot(transformed_vertices[0, :], transformed_vertices[1, :], 'o-')
    # plt.plot(T.dot(vertices)[0, :], T.dot(vertices)[1, :], 'o-')
    plt.show()

    res = optimize.minimize(
        transform_evaluate,
        np.array([T_2[0,0], T_2[0,1], T_2[0,2], T_2[1,0], T_2[1,1], T_2[1,2], T_2[2,0], T_2[2,1]]),
        args=(image_boundries, vertices)
    )

    T_2 = np.array([res.x[0:3], res.x[3:6], np.append(res.x[6:8], 1)])

    print("T - res.x")
    print(T-T_2)

    plt.plot(image_boundries[0, :], image_boundries[1, :], 'o-')
    plt.plot(T_2.dot(vertices)[0, :], T_2.dot(vertices)[1, :], 'o-')
    # plt.plot(transformed_vertices[0, :], transformed_vertices[1, :], 'o-')
    # plt.plot(T.dot(vertices)[0, :], T.dot(vertices)[1, :], 'o-')
    plt.show()

    # TEST_CUTOUT_BOX = {
    #         'east': SAMPLE_LOCATION[1] + 0.1,
    #         'north' : SAMPLE_LOCATION[0] + 0.1,
    #         'west' : SAMPLE_LOCATION[1] - 0.1,
    #         'south' : SAMPLE_LOCATION[0] - 0.1
    #     }
    #
    # crop_vertices = np.array((
    #     [TEST_CUTOUT_BOX['west'], TEST_CUTOUT_BOX['east'], TEST_CUTOUT_BOX['east'], TEST_CUTOUT_BOX['west']],
    #     [TEST_CUTOUT_BOX['north'], TEST_CUTOUT_BOX['north'], TEST_CUTOUT_BOX['south'], TEST_CUTOUT_BOX['south']],
    #     4*[1],
    # ))
    #
    # transformed_crop_vertices = T.dot(crop_vertices)
    #
    # transformed_crop_vertices_dict = {
    #     'east': min(transformed_crop_vertices[0, :]),
    #     'north' : max(transformed_crop_vertices[1, :]),
    #     'west' : max(transformed_crop_vertices[0, :]),
    #     'south' : min(transformed_crop_vertices[1, :])
    # }
    #
    # cropped_image = images.crop(image, bounds, transformed_crop_vertices_dict)
    #
    # plt.imshow(cropped_image)
    # plt.show()

else:
    image_pixels = image.load()
    image_boundries = []
    row_zero = 0
    row_last = image.size[0]-1
    nan_pixel = (0, 0, 0)

    for col in range(image.size[1]):
        if image_pixels[col, row_zero] != nan_pixel:
            if (col, row_zero) not in image_boundries:
                image_boundries.append((col, row_zero))
            break

    for col in range(image.size[1]):
        if image_pixels[col, row_last] != nan_pixel:
            if (col, row_last) not in image_boundries:
                image_boundries.append((col, row_last))
            break

    for col in range(image.size[1]-1, -1, -1):
        if image_pixels[col, row_zero] != nan_pixel:
            if (col, row_zero) not in image_boundries:
                image_boundries.append((col, row_zero))
            break

    for col in range(image.size[1]-1, -1, -1):
        if image_pixels[col, row_last] != nan_pixel:
            if (col, row_last) not in image_boundries:
                image_boundries.append((col, row_last))
            break

    col_zero = 0
    col_last = image.size[1]-1

    for row in range(image.size[0]):
        if image_pixels[col_zero, row] != nan_pixel:
            if (col_zero, row) not in image_boundries:
                image_boundries.append((col_zero, row))
            break

    for row in range(image.size[0]):
        if image_pixels[col_last, row] != nan_pixel:
            if (col_last, row) not in image_boundries:
                image_boundries.append((col_last, row))
            break

    for row in range(image.size[0]-1, -1, -1):
        if image_pixels[col_zero, row] != nan_pixel:
            if (col_zero, row) not in image_boundries:
                image_boundries.append((col_zero, row))
            break

    for row in range(image.size[0]-1, -1, -1):
        if image_pixels[col_last, row] != nan_pixel:
            if (col_last, row) not in image_boundries:
                image_boundries.append((col_last, row))
            break

    plt.imshow(image)
    plt.plot([x[0] for x in image_boundries], [x[1] for x in image_boundries], 'o')
    plt.show()

    image_boundries = np.vstack((np.array(image_boundries).transpose(), len(image_boundries)*[1]))

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

    # T = image_boundries.dot(np.linalg.pinv(vertices))
    # T = vertices.dot(np.linalg.pinv(transformed_vertices))

    # print("T")
    # print(T)

    # print("T * vertices")
    # print(T.dot(vertices))

    # T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # res = optimize.minimize(
    #     initial_transform_evaluate,
    #     np.array([5000, 0, -5000, 5000]),
    #     args=(image_boundries, vertices)
    # )

    # COBYLA yielded some different results
    # SLSQP yielded some different results

    # print("res.x ")
    # # T_2 = np.array([res.x[0:3], res.x[3:6], np.append(res.x[6:8], 1)])
    # T_2 = np.array([[res.x[0], 0, res.x[1]], [0, res.x[2], res.x[3]], [0, 0, 1]])
    # print(T_2)

    # plt.plot(image_boundries[0, :], image_boundries[1, :], 'o-')
    # plt.plot(T_2.dot(vertices)[0, :], T_2.dot(vertices)[1, :], 'o-')
    # plt.plot(transformed_vertices[0, :], transformed_vertices[1, :], 'o-')
    # plt.plot(T.dot(vertices)[0, :], T.dot(vertices)[1, :], 'o-')
    # plt.show()

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

    # for method in methods:
    #     try:
    res = optimize.minimize(
        transform_evaluate,
        np.array([T_initial[0, 0], T_initial[0, 1], T_initial[0, 2],
                  T_initial[1, 0], T_initial[1, 1], T_initial[1, 2],
                  T_initial[2, 0], T_initial[2, 1]]),
        args=(image_boundries, vertices),
        method=method
    )

    print(f"scode: {SCORE[-1]}")
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

    plt.plot(SCORE)
    plt.show()




