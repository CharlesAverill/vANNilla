from vANNilla.utils.matrix import (
    from_iterator,
    list_prod,
    multiply_accumulate,
    scalar_dot,
    zeros,
)


class Tensor:
    def __init__(self, tensor_values=None, shape=None, precision=5):
        """
        :param tensor_values: An integer or list of integers
        :param shape: If tensor_values are not provided,
                      create an empty tensor with this shape
        :param precision: Number of digits to round to when
                          checking equality
        """
        type_tval = type(tensor_values)
        if shape is None and tensor_values is None:
            self.tensor_values = []
            type_tval = list
        elif tensor_values is None:
            self.tensor_values = zeros(shape)
        elif type_tval == int or type_tval == float:
            self.tensor_values = round(tensor_values, precision)
        elif type_tval == Tensor:
            self.tensor_values = tensor_values.copy_tensor_values()
        elif type_tval == list:
            self.tensor_values = tensor_values
        else:
            raise TypeError(f"Cannot create tensor from type {type_tval}")
        self.precision = precision
        self.__shape = None
        self.__flattened = None
        self.__transposed = None

    def multidim_enumerate(self, submatrix=None, dim=None):
        """
        :param submatrix: Recursive parameter
        :param dim: Recursive parameter
        :return: Yield every scalar and its position in the tensor
        """
        if dim is None:
            dim = []
        if submatrix is None:
            submatrix = self.tensor_values
        try:
            for index, m_sub in enumerate(submatrix):
                yield from self.multidim_enumerate(m_sub, dim + [index])
        except TypeError:
            yield dim, submatrix

    def first_value(self):
        """Return the first value of a tensor or the value of a scalar"""
        try:
            return self.flattened[0].tensor_values
        except TypeError:
            return self.tensor_values

    @property
    def flattened(self):
        if not self.__flattened:
            self.__flattened = self.__flattened_helper()
        return self.__flattened

    def __flattened_helper(self):
        """Return a flattened (1-dimensional) copy of the Tensor"""
        ttype = self.tensor_type
        if ttype == "scalar" or ttype == "empty":
            out = Tensor()
            out.tensor_values = [self.tensor_values]
            return out
        return Tensor([item[1] for item in list(self.multidim_enumerate())])

    @property
    def transposed(self):
        if not self.__transposed:
            self.__transposed = self.__transposed_helper()
        return self.__transposed

    def __transposed_helper(self):
        if self.tensor_type == "vector":
            return Tensor([[item] for item in self])
        return Tensor(list(map(list, zip(*self.tensor_values))))

    @property
    def dims(self):
        return len(self.shape)

    @property
    def shape(self):
        if not self.__shape:
            self.__shape = self.__shape_helper()
        return self.__shape

    def __shape_helper(self):
        # Tuple of shape of vector m, accounts for raggedness
        if (
            type(self.tensor_values) == int
            or type(self.tensor_values) == float
        ):
            return ()
        if len(self.tensor_values) == 0:
            return -1
        m_shape = []
        m = self.tensor_values.copy()
        is_ragged = False
        while True:
            try:
                iter(m)
                last_len = len(m[0])
                for row in m:
                    iter(row)
                    if len(row) != last_len:
                        is_ragged = True
                        break
                    last_len = len(row)
                m_shape.append(len(m))
                m = m[0]
                if is_ragged:
                    break
            except TypeError:
                if type(m) == list:
                    m_shape.append(len(m))
                break
        return tuple(m_shape)

    @property
    def size(self):
        # Returns scalar size of matrix m
        return list_prod(self.shape)

    @property
    def tensor_type(self):
        if self.shape == -1:
            return "empty"
        len_shape = self.dims
        if len_shape == 0:
            return "scalar"
        if len_shape == 1:
            return "vector"
        if len_shape == 2:
            return "matrix"
        return f"{len_shape}-tensor"

    def reshape(self, new_shape):
        if list_prod(new_shape) != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size "
                f"{self.size} into shape {new_shape}"
            )
        return Tensor(
            from_iterator((flat for flat in self.flattened), new_shape)
        )

    def apply_all(self, function, *args):
        flat = self.flattened
        flat = Tensor([function(f, args) for f in flat])
        return flat.reshape(self.shape)

    def tensor_indeces(self):
        """
        :return: List of tuples of all indeces in the Tensor
        """
        return [tuple(i[0]) for i in list(self.multidim_enumerate())]

    def last_n_dimensions(self, n):
        """
        :param n: number of dimensions to pull from the end of the Tensor
        :return: a Tensor of the n-dimensional subtensors within the last
                 n dimensions of the Tensor
        """
        indeces = [pair[0][:-n] for pair in self.multidim_enumerate()]
        # Remove copies
        out = []
        for idx in indeces:
            try_append = self[tuple(idx)]
            if try_append not in out:
                out.append(try_append)
        return out

    def dot(self, n):
        n_type = type(n)
        if n_type not in (int, float, Tensor):
            raise TypeError(
                f"Cannot take inner product of Tensor and {n_type}"
            )
        if n_type != Tensor:
            n = Tensor(n)

        nshape = n.shape
        sshape = self.shape

        if len(nshape) == len(sshape) == 0:
            # Two Scalars
            return Tensor(self.tensor_values * n.tensor_values)

        elif len(nshape) == 0:
            # N is a Scalar, self is a Tensor
            def scalar_mul(*args):
                return args[0] * args[1][0].tensor_values

            return self.apply_all(scalar_mul, n)

        elif len(sshape) == 0:
            # N is a Scalar, self is a Tensor
            def scalar_mul(*args):
                return args[0] * args[1][0].tensor_values

            return n.apply_all(scalar_mul, self)

        elif len(nshape) == len(sshape) == 1:
            # Two 1D Vectors
            if nshape != sshape:
                raise ArithmeticError(
                    "Dimension mismatch: shapes "
                    f"{sshape} and {nshape} are not aligned"
                )
            return Tensor(scalar_dot(self.tensor_values, n.tensor_values))

        elif len(nshape) == len(sshape) == 2:
            # 2D x 2D Matrix multiplication
            if sshape[-1] != nshape[0]:
                raise ArithmeticError(
                    "Dimension mismatch: shapes "
                    f"{sshape} and {nshape} are not aligned"
                )

            out = zeros((sshape[-1], nshape[0]))
            for i in range(sshape[-1]):
                for j in range(nshape[0]):
                    total = 0
                    for k in range(sshape[-1]):
                        total += (
                            self.tensor_values[i][k] * n.tensor_values[k][j]
                        )
                    out[i][j] = total
            return out

        elif len(nshape) == 1:
            # ND Tensor and 1D Vector
            if sshape[-1] != nshape[0]:
                raise ArithmeticError(
                    "Dimension mismatch: shapes "
                    f"{sshape} and {nshape} are not aligned"
                )

            out = Tensor(shape=(sshape[:-1] + nshape[1:]))

            for axis, row in zip(
                out.tensor_indeces(), self.last_n_dimensions(1)
            ):
                out[axis] = scalar_dot(row, n)

            return out

        else:
            # ND Tensor x MD Tensor
            if sshape[-1] != nshape[-2]:
                raise ArithmeticError(
                    "Dimension mismatch: shapes "
                    f"{sshape} and {nshape} are not aligned"
                )

            out = Tensor(shape=(sshape[:-1] + nshape[:-2] + nshape[-1:]))
            for axis in zip(out.tensor_indeces()):
                axis = axis[0]
                second_to_last_axis_n = [i[axis[-1]] for i in n[axis[-2]]]
                out[axis] = scalar_dot(self[axis[:-2]], second_to_last_axis_n)

            return out

    def mean(self, axis=None):
        """
        :param axis: Axis along which to take the mean. Leave as none to take
                     mean of all values in the Tensor
        :return: Scalar mean if axis is None, Tensor mean along axis otherwise
        """
        if axis is not None:

            m_shape = self.shape
            axis = axis % len(m_shape)
            out = zeros(m_shape[:axis] + m_shape[axis + 1 :])
            for item in self.multidim_enumerate():
                index, val = item
                multiply_accumulate(
                    out,
                    index[:axis] + index[axis + 1 :],
                    1.0,
                    val / m_shape[axis],
                )
            return Tensor(out)
        flat = self.flattened
        return sum(flat.tensor_values) / len(flat)

    def sum(self, axis=None):
        """
        :param axis: Axis along which to take the sum. Leave as none to take
                     sum of all values in the Tensor
        :return: Scalar sum if axis is None, Tensor sum along axis otherwise
        """
        if axis is not None:

            m_shape = self.shape
            axis = axis % len(m_shape)
            out = zeros(m_shape[:axis] + m_shape[axis + 1 :])
            for item in self.multidim_enumerate():
                index, val = item
                multiply_accumulate(
                    out, index[:axis] + index[axis + 1 :], 1.0, val
                )
            return Tensor(out)
        flat = self.flattened
        return sum(flat.tensor_values)

    def copy_tensor_values(self):
        """Return a copy of the tensor's values"""
        if isinstance(self.tensor_values, list):
            return self.tensor_values.copy()
        else:
            return self.tensor_values

    def __copy__(self):
        return Tensor(self.copy_tensor_values())

    def __contains__(self, item):
        return item in self.tensor_values

    def __getitem__(self, key):
        if type(key) == tuple:
            temp_tensor = self
            for slice_index in key:
                temp_tensor = temp_tensor[slice_index]
            return temp_tensor
        return Tensor(self.tensor_values[key])

    def __setitem__(self, key, value):
        old = self[key]
        if old.shape != Tensor(value).shape:
            raise RuntimeWarning(
                "This Tensor is now ragged due to an item "
                "assignment with non-identical shape"
            )
        if type(key) == tuple:
            temp_list = self.tensor_values
            for slice_index in key[:-1]:
                temp_list = temp_list[slice_index]
            temp_list[key[-1]] = Tensor(value).tensor_values
        else:
            self.tensor_values[key] = Tensor(value).tensor_values

    def __len__(self):
        return len(self.tensor_values)

    def __repr__(self):
        ttype = self.tensor_type
        if ttype == "empty":
            return "empty-tensor()"
        if ttype == "scalar":
            return f"scalar({self.tensor_values})"
        out = f"{ttype}(["
        offset_len = len(out)
        out += str(self.tensor_values[0]) + ",\n"
        for val in self.tensor_values[1:]:
            out += (" " * offset_len) + str(val) + ",\n"
        out = out[:-2] + "])"
        return out

    def __str__(self):
        return str(self.tensor_values)

    def __neg__(self):
        ttype = self.tensor_type
        if ttype == "scalar":
            return Tensor(-self.tensor_values)
        if ttype == "empty":
            return self
        return Tensor([-flat for flat in self.flattened]).reshape(self.shape)

    def __add__(self, other):
        otype = type(other)
        sshape = self.shape
        if otype not in (int, float, Tensor):
            raise TypeError(f"Tensor cannot add with type {otype}")
        if otype in (int, float) or (
            otype == Tensor and len(other.shape) in (0, 1)
        ):
            return Tensor([flat + other for flat in self.flattened]).reshape(
                sshape
            )
        else:
            oshape = other.shape
            if sshape != oshape:
                raise ValueError(
                    "When adding tensors, shapes must be identical. "
                    f"Provided shapes are {sshape}, {oshape}"
                )
            return Tensor(
                [
                    self_flat + other_flat
                    for self_flat, other_flat in zip(
                        self.flattened, other.flattened
                    )
                ]
            ).reshape(sshape)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + Tensor(other)

    def __mul__(self, other):
        otype = type(other)
        if otype not in (int, float, Tensor):
            raise TypeError(f"Tensor cannot multiply with type {otype}")
        return self.dot(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iter__(self):
        if isinstance(self.tensor_values, list):
            return iter(self.tensor_values)
        else:
            return iter([self.tensor_values])

    def __eq__(self, other):
        if type(other) != Tensor:
            raise TypeError(f"Cannot compare Tensor with type {type(other)}")
        return [
            round(i, self.precision) for i in self.flattened.tensor_values
        ] == [round(j, self.precision) for j in other.flattened.tensor_values]

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if type(other) != Tensor:
            raise TypeError(f"Cannot compare Tensor with type {type(other)}")
        return max(self.flattened) < min(other.flattened)

    def __gt__(self, other):
        if type(other) != Tensor:
            raise TypeError(f"Cannot compare Tensor with type {type(other)}")
        return min(self.flattened) > max(other.flattened)

    def __le__(self, other):
        if type(other) != Tensor:
            raise TypeError(f"Cannot compare Tensor with type {type(other)}")
        return max(self.flattened) <= min(other.flattened)

    def __ge__(self, other):
        if type(other) != Tensor:
            raise TypeError(f"Cannot compare Tensor with type {type(other)}")
        return min(self.flattened) >= max(other.flattened)
