import onnx
import unittest
from onnx import helper
from onnx import TensorProto
from mutation.match_shape import broadcast_constrain, get_dim


def make_tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


class MatchShapeT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = onnx.helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 32, 14, 14])

    def test_broadcast_constrain(self):
        b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [14, 14])
        assert broadcast_constrain(get_dim(self.a), get_dim(b))
        c = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 32, 1, 1])
        assert broadcast_constrain(get_dim(self.a), get_dim(c))
        d = make_tensor('D', [4, 32, 1, 14])
        assert not broadcast_constrain(get_dim(self.a), get_dim(d))
        e = make_tensor('E', [])
        assert broadcast_constrain(get_dim(self.a), get_dim(e))

    def test_no_type(self):
        f = helper.make_tensor_value_info('F', TensorProto.FLOAT, None)
        assert broadcast_constrain(get_dim(self.a), get_dim(f))
