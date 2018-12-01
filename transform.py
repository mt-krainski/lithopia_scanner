import images
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import skimage
from skimage import feature


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


def scan_by(image_pixels,
            scan_range,
            row_col_id,
            matched_pixels,
            pixel = (0, 0, 0),
            by = 'row'):
    """
    scans by row or column in a single column or row to find a matching pixel
    :param image_pixels: pixels of an image as returned by pillows' load method
    :param scan_range: range on which to perform the search
    :param row_col_id: id of the row/column on which the search is to be performed
    :param matched_pixels: list to which matched pixels are appended
    :param pixel: pixel which is to be found
    :param by: 'row' or 'col', indicates whether the search should be performed
        by row, on a fixed column or by column on a fixed row
    :return: results are appended to the matched_pixels list
    """

    for row_col in scan_range:
        if by == 'row':
            indexes = (row_col_id, row_col)
        elif by == 'col':
            indexes = (row_col, row_col_id)
        else:
            raise ValueError("by can be either 'row' or 'col'!")

        if image_pixels[indexes] != pixel:
            if indexes not in matched_pixels:
                matched_pixels.append(indexes)
            break


def get_image_boundries(image, image_size):

    image_pixels = image.load()
    image_boundries = []

    row_zero = 0
    row_last = image_size[0]-1
    row_sequence = range(image_size[0])
    row_sequence_rev = range(image_size[0] - 1, -1, -1)

    col_zero = 0
    col_last = image_size[1] - 1
    col_sequence = range(image_size[1])
    col_sequence_rev = range(image_size[1] - 1, -1, -1)

    nan_pixel = (0, 0, 0)

    param_seq = [
        {'seq': col_sequence_rev, 'id': row_last, 'by': 'col'},
        {'seq': row_sequence_rev, 'id': col_last, 'by': 'row'},
        {'seq': row_sequence, 'id': col_last, 'by': 'row'},
        {'seq': col_sequence_rev, 'id': row_zero, 'by': 'col'},
        {'seq': col_sequence, 'id': row_zero, 'by': 'col'},
        {'seq': row_sequence, 'id': col_zero, 'by': 'row'},
        {'seq': row_sequence_rev, 'id': col_zero, 'by': 'row'},
        {'seq': col_sequence, 'id': row_last, 'by': 'col'},
    ]

    for param in param_seq:
        scan_by(image_pixels, param['seq'], param['id'],
                image_boundries, nan_pixel, param['by'])

    if len(image_boundries) < 4: ## not enough points
        IMAGE_SCALE_FACTOR = 5
        image_nd = skimage.io._plugins.pil_plugin.pil_to_ndarray(image)[0::IMAGE_SCALE_FACTOR,
                   0::IMAGE_SCALE_FACTOR, 0]
        image_nd[image_nd>0] = 1
        harris_score = feature.corner_harris(image_nd)
        corners = feature.corner_peaks(harris_score, min_distance=5)
        corners = corners*IMAGE_SCALE_FACTOR

        for corner in corners:
            image_boundries.append([corner[1], corner[0]])

    return np.array(image_boundries).transpose().tolist()


def find_transform(coords, target_coords):
    """
    Finds the best matching transformation from coords to
        image_boundaries
    :param coords: coordinates to be transformed. They should be
        in a format returned by images.get_coordinates, i.e. a list,
        first element of which is a list of longitudes, second element
        is a list of latitudes:
        [[lon1, lon2, ..., lonN], [lat1, lat2, ..., latN]]
    :param target_coords: target coordinates. a list, first
        element of which is a list of longitudes, second element
        is a list of latitudes:
        [[lon1, lon2, ..., lonN], [lat1, lat2, ..., latN]]
    :return: best match transformation matrix transforming from coords
        space to target_coords space
    """

    vertices = np.array([coords[0][:-1],
                         coords[1][:-1],
                         (len(coords[0]) - 1) * [1]])

    target_vertices = np.array([
        target_coords[0],
        target_coords[1],
        len(target_coords[0]) * [1]])

    zero_node = [min(vertices[0, :]), max(vertices[1, :])]

    t_zero = np.array([[1, 0, -zero_node[0]],
                       [0, -1, zero_node[1]],
                       [0, 0, 1]])

    vert_zero = t_zero.dot(vertices)

    vertices_range = np.max(vert_zero, axis=1) - np.min(vert_zero, axis=1)

    scale = [max(target_vertices[0, :]) - min(target_vertices[0, :]),
             max(target_vertices[1, :]) - min(target_vertices[1, :])]

    scaling = [scale[0] / vertices_range[0],
               scale[1] / vertices_range[1]]

    t_scale = np.array([[scaling[0], 0, 0],
                        [0, scaling[1], 0],
                        [0, 0, 1]])

    vert_scaled = t_scale.dot(vert_zero)

    t_translate = np.array([[1, 0, min(target_vertices[0, :]) - min(vert_scaled[0, :])],
                            [0, 1, min(target_vertices[1, :]) - min(vert_scaled[1, :])],
                            [0, 0, 1]])

    t_initial = t_translate.dot(t_scale.dot(t_zero))

    res = optimize.minimize(
        evaluate_transform,
        np.array([t_initial[0, 0], t_initial[0, 1], t_initial[0, 2],
                  t_initial[1, 0], t_initial[1, 1], t_initial[1, 2],
                  t_initial[2, 0], t_initial[2, 1]]),
        args=(target_vertices, vertices)
    )

    t_final = np.array([res.x[0:3], res.x[3:6], np.append(res.x[6:8], 1)])

    return t_final


def projection_transform(coords, transform):
    """
    Performds a transform transformation of coords. This might include
        projection transformation which requires additional scaling
        of the result
    :param coords: coordinates to be transformed. They should be
        in a format returned by images.get_coordinates, i.e. a list,
        first element of which is a list of longitudes, second element
        is a list of latitudes:
        [[lon1, lon2, ..., lonN], [lat1, lat2, ..., latN]]
    :param transform: transformation matrix
    :return: transformed coords in the same format as input
    """

    points = np.array([coords[0],
                         coords[1],
                         len(coords[0]) * [1]])

    result = transform.dot(points)

    for col in range(result.shape[1]):
        result[:, col] = result[:, col] / result[-1, col]

    result = result[0:2, :]

    return result.tolist()


def get_transform_function(image, coords):
    """
    Performs all necessary calculations to find a matching
        transformation from the geographic coordinate
        space to image pixel space. Returns a function
        which takes geographic coordinates as parameter
        and returns corresponding image coordinates
    :param image: image on which the coordinates
        should be projected
    :param coords: bounding box of the image (as defined
        in dataset manifest
    :return: a function which transforms coordinates
        from geographic coordinates to image coordinates
    """
    image_boundaries = get_image_boundries(image, image.size)
    t_matrix = find_transform(coords, image_boundaries)
    def transform(points):
        """
        Transforms points from geographical coordinates
            to pixel coordinates. This stores the transformation matrix
            in a closure
        :param points: coordinates to be transformed. They should be
            in a format returned by images.get_coordinates, i.e. a list,
            first element of which is a list of longitudes, second element
            is a list of latitudes:
            [[lon1, lon2, ..., lonN], [lat1, lat2, ..., latN]]
        :return: transformed points in the same format as input
        """
        return projection_transform(points, t_matrix)
    return transform


if __name__ == "__main__":
    ARCHIVE = "data/S2B_MSIL1C_20181117T100259_N0207_R122_T33UVR_20181117T120601.zip"

    image = images.get_tci_image(ARCHIVE)
    coords = images.get_coordinates(images.get_manifest(ARCHIVE))

    to_pixel_coords = get_transform_function(image, coords)

    transformed_coordinates = to_pixel_coords(coords)

    plt.imshow(image)
    plt.plot(transformed_coordinates[0],
             transformed_coordinates[1],
             'o-', color='red')
    plt.show()




