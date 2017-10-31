'''
THIS MODULE USES THE BASIC FUNCTION BUILD IN forward_reverse.py, IMPLEMENTING MORE COMPLEX TREATMENT WITH FORWARD_REVERSE LOGIC
'''
from numba import cuda
import numpy, math



@cuda.jit 
def map2NUM(data, _len, NUM):
	index = cuda.grid(1)
	if index<_len:
		if data[index]>0:
			cuda.atomic.add(NUM, data[index]-1, 1)


@cuda.jit
def map2PIVOT(data, _len, _len_y, PIVOT):
	index = cuda.grid(1)
	if index<_len:
		if(data[index]>0):
			index_x = int(index/_len_y)
			index_y = int(index%_len_y)
			cuda.atomic.min(PIVOT, (data[index]-1)*4, index_x)
			cuda.atomic.max(PIVOT, (data[index]-1)*4+1, index_x)
			cuda.atomic.min(PIVOT, (data[index]-1)*4+2, index_y)
			cuda.atomic.max(PIVOT, (data[index]-1)*4+3, index_y)


@cuda.jit
def PIVOT2level0(PIVOT, _len_index, num_level_0):
	index = cuda.grid(1)
	if index<_len_index:
		min_x = PIVOT[index*4]
		max_x = PIVOT[index*4+1]
		min_y = PIVOT[index*4+2]
		max_y = PIVOT[index*4+3]
		num_range = (max_x-min_x+1)*(max_y-min_y+1)
		num_level_0[index] = num_range



@cuda.jit
def label2map(data, PIVOT, _len_index, _labels, level_PIVOT):
	x = cuda.grid(1)
	if x<_len_index:
		label = x+1
		range_y = PIVOT[x*4+3]-PIVOT[x*4+2]+1
		for i in range(PIVOT[x*4], PIVOT[x*4+1]+1):
			for j in range(PIVOT[x*4+2], PIVOT[x*4+3]+1):
				if(data[i,j]==label):
					index_x = i-PIVOT[x*4]
					index_y = j-PIVOT[x*4+2]
					_labels[index_x*range_y+index_y+level_PIVOT[x]] = 1



@cuda.jit
def map2index(FINAL, single_final, INDEX, NUM, PIVOT, INDEX_PIVOT, level_PIVOT, _len_index):
	x = cuda.grid(1)
	if x<_len_index:
		min_x = PIVOT[x*4]
		min_y = PIVOT[x*4+2]	
		max_y = PIVOT[x*4+3]
		range_y = max_y-min_y+1
		for index in range(NUM[x]):
			index_x = (INDEX[INDEX_PIVOT[x]+index]-level_PIVOT[x])//range_y+min_x

			index_y = (INDEX[INDEX_PIVOT[x]+index]-level_PIVOT[x])%range_y+min_y

			FINAL[INDEX_PIVOT[x]+index, 0] = index_x
			FINAL[INDEX_PIVOT[x]+index, 1] = index_y

			single_final[INDEX_PIVOT[x]+index] = index_x*2048+index_y


from GPU.reduce_filter import forward_reverse as fr
def labels2matrix(data, shape=(2048,2048), device=0):
	'''
	this function merge labels with different value into a 2D binary matrix, where each row of the matrix representing mapping of each label value
	we truncated and remap the mapping to make the matrix small enough
	'''
	cuda.select_device(device)
	_len = data.shape[0]
	_len_index = data.max()
	_len_y = shape[1]

	NUM = cuda.to_device(numpy.zeros((_len_index)).astype('int32'))
	data_gpu = cuda.to_device(data)

	PIVOT = numpy.zeros((_len_index*4)).astype('int32')
	cond_list = numpy.array([x*4 for x in range(_len_index)])
	PIVOT[cond_list] = 2048*2048*2

	cond_list = numpy.array([x*4+2 for x in range(_len_index)])
	PIVOT[cond_list] = 2048*2048*2 


	PIVOT = cuda.to_device(PIVOT)


	thresperblock = 1024
	blockpergrid = (math.ceil(_len/thresperblock))

	map2NUM[blockpergrid, thresperblock](data_gpu, _len, NUM)
	cuda.synchronize()

	map2PIVOT[blockpergrid, thresperblock](data, _len, _len_y, PIVOT)
	cuda.synchronize()

	# NUM = NUM.copy_to_host()
	# PIVOT = PIVOT.copy_to_host()


	#num_level_0 is the array storing first level length of STORAGE array for each label
	num_level_0 = cuda.to_device(numpy.zeros((_len_index)).astype('int32'))

	blockpergrid = (math.ceil(_len_index/thresperblock))


	PIVOT2level0[blockpergrid, thresperblock](PIVOT, _len_index, num_level_0)
	cuda.synchronize()

	num_level_0 = num_level_0.copy_to_host()

	labels = cuda.to_device(numpy.zeros((num_level_0.sum())).astype('int32'))

	PIVOT = PIVOT.copy_to_host()


	level_PIVOT = numpy.zeros((_len_index)).astype('int32')

	for i in range(1, level_PIVOT.shape[0]):
		level_PIVOT[i] = level_PIVOT[i-1]+num_level_0[i-1]

	data.resize(shape)
	data = cuda.to_device(data)
	level_PIVOT = cuda.to_device(level_PIVOT)
	PIVOT = cuda.to_device(PIVOT)

	label2map[blockpergrid, thresperblock](data, PIVOT, _len_index, labels, level_PIVOT)

	INDEX = fr(labels.copy_to_host(), d2=False, device=device)

	# print(INDEX)
	NUM = NUM.copy_to_host()

	INDEX_PIVOT = numpy.zeros((_len_index)).astype('int32')
	for i in range(1, level_PIVOT.shape[0]):
		INDEX_PIVOT[i] = INDEX_PIVOT[i-1]+NUM[i-1]

	FINAL = cuda.to_device(numpy.zeros((NUM.sum(), 2)).astype('int32'))
	single_final = cuda.to_device(numpy.zeros((NUM.sum())).astype('int32'))

	INDEX = cuda.to_device(INDEX)
	INDEX_PIVOT = cuda.to_device(INDEX_PIVOT)
	NUM = cuda.to_device(NUM)

	map2index[blockpergrid, thresperblock](FINAL, single_final, INDEX, NUM, PIVOT, INDEX_PIVOT, level_PIVOT, _len_index)

	F, I, N, s = FINAL.copy_to_host(), INDEX_PIVOT.copy_to_host(), NUM.copy_to_host(), single_final.copy_to_host()

	del FINAL
	del INDEX
	del INDEX_PIVOT
	del NUM
	del single_final
	del data_gpu
	del PIVOT
	del num_level_0

	return F, I, N, s

from skimage import measure
from timeit import default_timer
def ccl_fr(binary, shape=(2048,2048), size_upper=100000, size_lower=10, device=0, cells_out=False):
	end = default_timer()
	labels = measure.label(binary, background=0, connectivity=1)
	labels.resize((shape[0]*shape[1]))

	print("ccl", default_timer()-end)
	end = default_timer()

	INDEX, PIVOT, NUM, _ = labels2matrix(labels, shape=shape, device=0)
	select = (NUM > size_lower)*(NUM < size_upper)
	selected = fr(select, d2=False, device=0)
	new_pivot = PIVOT[selected]

	print("map", default_timer()-end)
	end = default_timer()

	if(cells_out==True):
		LIST = []
		if type(selected) is not list:

			for i in range(selected.shape[0]):
				_start = PIVOT[selected[i]]
				_end = PIVOT[selected[i]+1]
				LIST.append(INDEX[_start:_end,:].tolist())

		print("list", default_timer()-end)

		return LIST

	else:
		return INDEX, new_pivot




# from CV.ImageIO import imreadTif, imwrite, gray2rgb, center_cell, drawMarker, symmetry_mapping
# from GPU.local_threshold import threshold_loader
# from timeit import default_timer
# from GRAPH.connect import sc_single
# from skimage import measure
# import copy, numpy
# # from ms import bfs_mesh


# imgPath = "../normal.tif"
# binary = threshold_loader(imgPath, stride=32, grid_size=32, color_channel=4095)

# img = imreadTif(imgPath)

# binary = img > 2059

# binary = numpy.ones((2048, 2048))

# for i in range(10):
# labels = measure.label(binary, background=0, connectivity=1)
# labels.resize((2048*2048))

# for i in range(10):
# end = default_timer()
# _labels = copy.deepcopy(labels)
# INDEX, PIVOT, NUM, _ = labels2matrix(labels, device=0)
# print(default_timer()-end)


# select = NUM > 500

# selected = fr(select, d2=False, device=0)

# print(selected)


# img = numpy.zeros((2048,2048))
# for i in range(selected.shape[0]):
# 	_index = selected[i]
# 	for index in range(PIVOT[_index], PIVOT[_index+1]):
# 		# print(index)
# 		img[INDEX[index,0], INDEX[index,1]] = 255

# imwrite(img, "./result.jpeg")


# for i in range(10):
# 	end = default_timer()
# 	# _labels = copy.deepcopy(labels)
# 	INDEX, PIVOT = ccl_fr(binary, shape=(2048,2048), size_upper=100000, size_lower=10, device=0)
# 	print(default_timer()-end)