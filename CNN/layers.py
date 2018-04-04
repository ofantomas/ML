import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, stride)

    cols = x[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=0,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    x = np.zeros((N, C, H, W), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, stride)
    cols_reshaped = cols.reshape(
        C * field_height * field_width, -1, N).transpose(2, 0, 1)
    np.add.at(x, (slice(None), k, i, j), cols_reshaped)
    return x


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """

        raise NotImplementedError("Not implemented in interface")

    def forward(self, input):
        """
        Takes input data of shape [batch, ...], returns output data [batch, ...]
        """

        raise NotImplementedError("Not implemented in interface")

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input. 
        Updates layer parameters and returns gradient for next layer
        Let x be layer weights, output – output of the layer on the given input and grad_output – 
        gradient of layer with respect to output

        To compute loss gradients w.r.t parameters, you need to apply chain rule (backprop):
        (d loss / d x)  = (d loss / d output) * (d output / d x)
        Luckily, you already receive (d loss / d output) as grad_output, so you only need to multiply 
        it by (d output / d x)
        If your layer has parameters (e.g. dense layer), you need to update them here using d loss / d x

        returns (d loss / d input) = (d loss / d output) * (d output / d input)
        """

        raise NotImplementedError("Not implemented in interface")


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        This layer does not have any parameters.
        """

    def forward(self, input):
        """
        Perform ReLU transformation
        input shape: [batch, input_units]
        output shape: [batch, input_units]
        """
        return np.where(input < 0, 0, input)

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """
        return grad_output * np.where(input < 0, 0, 1)


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = Wx + b

        W: matrix of shape [num_inputs, num_outputs]
        b: vector of shape [num_outputs]
        """
        self.input_units = input_units
        self.output_units = output_units

        self.learning_rate = learning_rate

        # initialize weights with small random numbers from normal distribution

        self.weights = np.random.normal(loc=0, scale=(
            2 / (input_units + output_units)), size=(input_units, output_units))
        self.biases = np.random.normal(loc=0, scale=(
            2 / (input_units + output_units)), size=(output_units))

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return input.dot(self.weights) + self.biases[np.newaxis, :]

    def backward(self, input, grad_output):
        """
        input shape: [batch, input_units]
        grad_output: [batch, output_units]

        Returns: grad_input, gradient of output w.r.t input
        """
        res = grad_output.dot(self.weights.T)
        self.weights -= self.learning_rate * input.T.dot(grad_output)
        self.biases -= self.learning_rate * grad_output.sum(axis=0)
        return res


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.1):
        """
        A convolutional layer with out_channels kernels of kernel_size.

        in_channels — number of input channels
        out_channels — number of convolutional filters
        kernel_size — tuple of two numbers: k_1 and k_2

        Initialize required weights.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(scale=(2 / (in_channels + out_channels)), size=(
            in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.weights = np.rot90(self.weights, k=2, axes=(2, 3))

    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        h_out = input.shape[2] - self.kernel_size[0] + 1
        w_out = input.shape[3] - self.kernel_size[1] + 1

        print(self.weights.shape, self.out_channels)
        new_input = im2col_indices(
            input, self.kernel_size[0], self.kernel_size[1])
        new_weights = self.weights.reshape(self.out_channels, -1)

        res = new_weights.dot(new_input).reshape(
            self.out_channels, h_out, w_out, input.shape[0])

        return res.transpose(3, 0, 1, 2)

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights
        """
        grad_output = grad_output.transpose(
            1, 2, 3, 0).reshape(self.out_channels, -1)
        res = self.weights.reshape(self.out_channels, -1).T.dot(grad_output)

        new_input = im2col_indices(
            input, self.kernel_size[0], self.kernel_size[1])
        self.weights -= self.learning_rate * grad_output.dot(
            new_input.T).reshape(self.weights.shape)

        return col2im_indices(res, input.shape, self.kernel_size[0], self.kernel_size[1])


class Maxpool2d(Layer):
    def __init__(self, kernel_size):
        """
        A maxpooling layer with kernel of kernel_size.
        This layer donwsamples [kernel_size, kernel_size] to
        1 number which represents maximum.

        Stride description is identical to the convolution
        layer. But default value we use is kernel_size to
        reduce dim by kernel_size times.

        This layer does not have any learnable parameters.
        """

        self.stride = kernel_size
        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Perform maxpooling transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        n, c, h, w = input.shape
        h_out = int((h - self.kernel_size) / self.stride + 1)
        w_out = int((w - self.kernel_size) / self.stride + 1)

        new_input = input.reshape(n * c, 1, h, w)
        new_input = im2col_indices(
            new_input, self.kernel_size, self.kernel_size, self.stride)

        self.col_shape = new_input.shape
        self.ind = np.argmax(new_input, axis=0)

        max_elem = np.amax(new_input, axis=0)
        return max_elem.reshape(h_out, w_out, n, c).transpose(2, 3, 0, 1)

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Maxpool2d input
        """
        n, c, h, w = input.shape
        res = np.zeros(self.col_shape)

        grad_output_ind = grad_output.transpose(2, 3, 0, 1).ravel()

        res[self.ind, range(grad_output_ind.shape[0])] = grad_output_ind
        res = col2im_indices(
            res, (n * c, 1, h, w), self.kernel_size, self.kernel_size, stride=self.stride)

        return res.reshape(input.shape)


class Flatten(Layer):
    def __init__(self):
        """
        This layer does not have any parameters
        """

    def forward(self, input):
        """
        input shape: [batch_size, channels, feature_nums_h, feature_nums_w]
        output shape: [batch_size, channels * feature_nums_h * feature_nums_w]
        """
        return input.reshape(-1, np.prod(input.shape[1::]))

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Flatten input
        """

        return grad_output.reshape(input.shape)


def softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    output is a number
    """
    probs = np.exp(logits - np.amax(logits, axis=1)[:, np.newaxis]) \
        / np.exp(logits - np.amax(logits, axis=1)[:, np.newaxis]).sum(axis=1)[:, np.newaxis]
    return -((y_true[:, np.newaxis] == np.arange(logits.shape[1])) * np.log(probs)).sum(axis=1)


def grad_softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy gradient from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    """
    probs = np.exp(logits - np.amax(logits, axis=1)[:, np.newaxis]) \
        / np.exp(logits - np.amax(logits, axis=1)[:, np.newaxis]).sum(axis=1)[:, np.newaxis]
    return -((y_true[:, np.newaxis] == np.arange(logits.shape[1])).astype(int) - probs) / (logits.shape[0])
