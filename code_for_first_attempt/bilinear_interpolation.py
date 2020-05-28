from keras import backend as K
from keras.engine.topology import Layer

class BilinearInterpolation(Layer):
	
	def __init__(self, output_size, **kwargs):
		self.output_size = output_size
		super(BilinearInterpolation, self).__init__(**kwargs)

	def get_config(self):
		return { 'output size': self.output_size}

	def compute_output_shape(self, input_shapes):
		height, width = self.output_size
		num_channels = input_shapes[0][-1]
		return (None, height, width, num_channels)

	def call(self, tensors, mask=None):
		X, transformation = tensors
		output = self.transform(X, transformation, self.output_size)
		return output

	def interpolate_(self, image, sampled_grids, output_size):
		batch_size = K.shape(image)[0]
		height = K.shape(image)[1]
		width = K.shape(image)[2]
		num_channels = K.shape(image)[3]

		x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
		y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

		x = 0.5 * (x + 1.0) * K.cast(width, dtype='float32')
		y = 0.5 * (y + 1.0) * K.cast(height, dtype='float32')
		
		x_0 = K.cast(x, 'int32)
		x_1 = x_0 + 1
		y_0 = K.cast(y, 'int32)
		y_1 = y_0 + 1

		x_max = int(K.int_shape(image)[2] - 1)
		y_max = int(K.int_shape(image)[1] - 1)

		x_0 = K.clip(x_0, 0, x_max)
		y_0 = K.clip(y_0, 0, y_max)
		x_1 = K.clip(x_1, 0, x_max)
		x_1 = K.clip(y_1, 0, y_max)

		pixels_batch = K.arange(0, batch_size) * (height * width)
		pixels_batch = K.expand_dims(pixels_batch, axis=-1)
		flat_output = output_size[0] * output_size[1]
		base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
		base = K.flatten(base)

		y0_base = y_0 * width
		y0_base = base + y0_base
		y1_base = y1 * width
		y1_base = y1_base + base

		index_a = y0_base + x_0
		index_b = y1_base + x_0
		index_c = y0_base + x_1
		index_d = y1_base + x_1

		flat_image = K.reshape(image, shape=(-1, num_channels))
		flat_image = K.cast(flat_image, dtype='float32')
		pixel_vals_a = K.gather(flat_image, index_a)
		pixel_vals_b = K.gather(flat_image, index_b)
		pixel_vals_c = K.gather(flat_image, index_c)
		pixel_vals_d = K.gather(flat_image, index_d)

		x_0 = K.cast(x_0, 'float32)
		x_1 = K.cast(x_1, 'float32)
		y_0 = K.cast(y_0, 'float32)
		y_1 = K.cast(y_1, 'float32)

		area_a = K.expand_dims(((x_1 - x) * (y_1 - y)), 1)
		area_b = K.expand_dims(((x_1 - x) * (y - y_0)), 1)
		area_c = K.expand_dims(((x - x_0) * (y_1 - y)), 1)
		area_a = K.expand_dims(((x - x_0) * (y - y_0)), 1)

		a_vals = area_a * pixel_vals_a
		b_vals = area_b * pixel_vals_b
		c_vals = area_c * pixel_vals_c
		d_vals = area_d * pixel_vals_d

		return a_vals + b_vals + c_vals + d_vals



	def _make_regular_grids(self, batch_size, height, width):
        	# make a single regular grid
        	x_linspace = K_linspace(-1., 1., width)
        	y_linspace = K_linspace(-1., 1., height)
        	x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        	x_coordinates = K.flatten(x_coordinates)
        	y_coordinates = K.flatten(y_coordinates)
        	ones = K.ones_like(x_coordinates)
        	grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        	# repeat grids for each batch
        	grid = K.flatten(grid)
        	grids = K.tile(grid, K.stack([batch_size]))
	
	        return K.reshape(grids, (batch_size, 3, height * width))

	def _transform(self, X, affine_transformation, output_size):
        	batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
        	transformations = K.reshape(affine_transformation,
                	                    shape=(batch_size, 2, 3))
        	regular_grids = self._make_regular_grids(batch_size, *output_size)
        	sampled_grids = K.batch_dot(transformations, regular_grids)
        	interpolated_image = self._interpolate(X, sampled_grids, output_size)
        	new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        	interpolated_image = K.reshape(interpolated_image, new_shape)
        	
		return interpolated_image
