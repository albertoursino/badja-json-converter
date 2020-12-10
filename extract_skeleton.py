import h5py

IMAGE_SIZE = (512, 256)

hf = h5py.File('annotation_set_{}_{}.h5'.format(IMAGE_SIZE[0], IMAGE_SIZE[1]), 'r')
skeleton = h5py.File('skeleton.h5', 'w')

skeleton_dataset = hf['skeleton']
skeleton.create_dataset('skeleton', data=skeleton_dataset)
