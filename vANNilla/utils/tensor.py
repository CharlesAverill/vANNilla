from vANNilla.utils.matrix import (
    from_iterator,
    list_prod,
    multidim_enumerate,
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
        if shape is None and tensor_values is None:
            self.tensor_values = []
        elif tensor_values is None:
            self.tensor_values = zeros(shape)
        elif type(tensor_values) == int or type(tensor_values) == float:
            self.tensor_values = tensor_values
        elif type(tensor_values) == Tensor:
            self.tensor_values = tensor_values.copy_tensor_values()
        elif type(tensor_values) == list:
            self.tensor_values = [
                Tensor(t).tensor_values for t in tensor_values
            ]
        else:
            raise TypeError(
                f"Cannot create tensor from type {type(tensor_values)}"
            )
        self.precision = precision

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
        """Return a flattened (1-dimensional) copy of the Tensor"""
        ttype = self.tensor_type
        if ttype == "scalar" or ttype == "empty":
            out = Tensor()
            out.tensor_values = [self.tensor_values]
            return out
        return Tensor([item[1] for item in list(self.multidim_enumerate())])

    @property
    def transposed(self):
        if self.tensor_type == "vector":
            return Tensor([[item] for item in self])
        return Tensor(list(map(list, zip(*self.tensor_values))))

    @property
    def dims(self):
        return len(self.shape)

    @property
    def shape(self):
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

    def dot(self, n):
        if type(n) == int or type(n) == float:
            n = Tensor(n)
        ttype = self.tensor_type
        nttype = n.tensor_type
        if type(n) != Tensor:
            raise TypeError(f"Tensors cannot interact with type {type(n)}")
        elif ttype == "empty" or nttype == "empty":
            raise ValueError("Cannot operate on empty tensors")
        elif nttype == ttype == "scalar":
            return Tensor(n.tensor_values * self.tensor_values)
        elif nttype == "scalar":
            flattened_scaled = (
                flat * n.tensor_values for flat in self.flattened
            )
            return Tensor(from_iterator(flattened_scaled, self.shape))
        elif ttype == "scalar":
            flattened_scaled = (
                flat * self.tensor_values for flat in n.flattened
            )
            return Tensor(from_iterator(flattened_scaled, n.shape))
        elif self.shape[-1] != n.shape[0]:
            raise ValueError(
                f"Dimension mismatch: shapes "
                f"{self.shape} and {n.shape} are not aligned"
            )
        elif self.dims != n.dims:
            if self.dims < n.dims:
                tensor_a = self
                tensor_b = n
            else:
                tensor_a = n
                tensor_b = self
            out = []

            for column in tensor_b.tensor_values:
                to_append = []
                for a_value, b_value in zip(tensor_a.tensor_values, column):
                    to_append.append(a_value * b_value)
                out.append(sum(to_append))

            return Tensor(out)
        elif ttype == nttype == "vector":
            return scalar_dot(self.tensor_values, n.tensor_values)
        out = zeros((self.shape[-1], n.shape[0]))
        for i in range(self.shape[-1]):
            for j in range(n.shape[0]):
                total = 0
                for k in range(self.shape[-1]):
                    total += self.tensor_values[i][k] * n.tensor_values[k][j]
                out[i][j] = total

        return Tensor(out)

    def mean(self, axis=None):
        """
        :param axis: Axis along which to take the mean. Leave as none to take
                     mean of all values in the Tensor
        :return: Scalar mean if axis is None, Tensor mean along axis otherwise
        """
        if not axis:
            m_shape = self.shape
            axis = axis % len(m_shape)
            out = zeros(m_shape[:axis] + m_shape[axis + 1 :])
            for index, val in multidim_enumerate(self.tensor_values):
                out = multiply_accumulate(
                    out,
                    index[:axis] + index[axis + 1 :],
                    1,
                    val / m_shape[axis],
                )
            return Tensor(out)
        flat = self.flattened
        return sum(flat.tensor_values) / len(flat)

    def copy_tensor_values(self):
        """Return a copy of the tensor's values"""
        try:
            return self.tensor_values.copy()
        except AttributeError:
            return self.tensor_values

    def __getitem__(self, item):
        return Tensor(self.tensor_values[item])

    def __len__(self):
        return len(self.tensor_values)

    def __str__(self):
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

    def __neg__(self):
        ttype = self.tensor_type
        if ttype == "scalar":
            return Tensor(-self.tensor_values)
        if ttype == "empty":
            return self
        return Tensor([-flat for flat in self.flattened]).reshape(self.shape)

    def __add__(self, other):
        otype = type(other)
        if otype != int and otype != float and otype != Tensor:
            raise TypeError(f"Tensor cannot add with type {otype}")
        sshape = self.shape
        if otype == int or otype == float:
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

    def __sub__(self, other):
        otype = type(other)
        if otype != int and otype != float and otype != Tensor:
            raise TypeError(f"Tensor cannot subtract with type {otype}")
        return self + -other

    def __mul__(self, other):
        otype = type(other)
        if otype != int and otype != float and otype != Tensor:
            raise TypeError(f"Tensor cannot multiply with type {otype}")
        return self.dot(other)

    def __iter__(self):
        return iter(self.tensor_values)

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
