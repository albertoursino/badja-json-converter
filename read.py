import h5py

from write import IMAGE_SIZE

hf = h5py.File('h5/annotation_set_{}_{}.h5'.format(IMAGE_SIZE[0], IMAGE_SIZE[1]), 'r')

for x in hf['annotations']:
    print(x)