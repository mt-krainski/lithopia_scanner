import images
import matplotlib.pyplot as plt
import numpy as np

ARCHIVE = "data/S2A_MSIL1C_20181112T100241_N0207_R122_T33UVQ_20181112T121216.zip"

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


# plt.plot(coords[0][:-1], coords[1][:-1], 'o-')
# plt.plot((bounds['west'], bounds['west'], bounds['east'], bounds['east'], bounds['west']),
#          (bounds['north'], bounds['south'], bounds['south'], bounds['north'], bounds['north']), 'o-')
#
# plt.show()

vertices = np.array([coords[0][:-1], coords[1][:-1], 4*[1]])
transformed_vertices = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
# transformed_vertices = np.array((
#     [bounds['west'], bounds['east'], bounds['east'], bounds['west']],
#     [bounds['north'], bounds['north'], bounds['south'], bounds['south']],
#     4*[1],
# ))


print(vertices)
print(transformed_vertices)

T = transformed_vertices.dot(np.linalg.pinv(vertices))

print("T")
print(T)

# plt.plot(vertices[0, :], vertices[1, :], 'o-')
plt.plot(transformed_vertices[0, :], transformed_vertices[1, :], 'o-')
plt.plot(T.dot(vertices)[0, :], T.dot(vertices)[1, :], 'o-')
plt.show()

print("T * vertices")
print(T.dot(vertices))

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
