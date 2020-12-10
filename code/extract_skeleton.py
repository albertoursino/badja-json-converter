import h5py

HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/Convert-BADJA-json/'

hf = h5py.File(HOME+'annotation_set_512_256.h5', 'a')
skeleton = h5py.File(HOME+'skeleton.h5', 'a')

skeleton_dataset = hf['skeleton']

skeleton.create_dataset('skeleton', data=skeleton_dataset)
