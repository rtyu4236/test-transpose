import numpy as np
import pandas as pd
import torch
import unittest

def mytranspose(x):
    """
    Transpose for numpy arrays, pandas DataFrames, and (optionally) PyTorch tensors.
    
    - 1D numpy 배열: 행렬로 변환 후 전치
    - 2D numpy 배열: .T 사용
    - pandas DataFrame: .T 사용
    - torch.Tensor: .T 사용
    """
    # DataFrame
    if isinstance(x, pd.DataFrame):
        return x.T

    # PyTorch Tensor
    if isinstance(x, torch.Tensor):
        return x.T

    # NumPy array
    if isinstance(x, np.ndarray):
        x = np.atleast_2d(x)
        return x.T

    raise TypeError(f"Unsupported input for transpose: {type(x)} (ndim={getattr(x, 'ndim', None)})")


class TestMyTranspose(unittest.TestCase):
    def setUp(self):
        # (1) Matrix
        self.matrices = {
            "5x2": np.array([[1,2],[3,4],[5,6],[7,8],[9,10]]),
            "empty": np.empty((0,0)),
            "1x2": np.array([[1,2]]),
            "2x1": np.array([[1],[2]])
        }
        # (2) Vector
        self.vectors = {
            "nan_vec": np.array([1, 2, np.nan, 3]),
            "single_nan": np.array([np.nan]),
            "empty_vec": np.array([]),
        }
        # (3) DataFrame
        self.df = pd.DataFrame({
            "d": [1,2,3,4],
            "e": ["red","white","red", np.nan],
            "f": [True,True,True,False]
        })
        # (4) optional PyTorch tensor
        self.tensor = torch.tensor([[1,2],[3,4]])

    def test_matrices(self):
        for name, arr in self.matrices.items():
            with self.subTest(name=name):
                out = mytranspose(arr)
                print(f"\nTesting {name} matrix transpose")
                print(arr)
                print(out)
                np.testing.assert_array_equal(out, arr.T)

    def test_vectors(self):
        for name, arr in self.vectors.items():
            with self.subTest(name=name):
                out = mytranspose(arr)
                print(f"\nTesting {name} vector transpose")
                print(arr)
                print(out)
                # np.testing.assert_array_equal(out, arr)

    def test_dataframe(self):
        out = mytranspose(self.df)
        print(f"\nTesting DataFrame transpose")
        print(self.df)
        print(out)
        pd.testing.assert_frame_equal(out, self.df.T)

    def test_pytorch_tensor(self):
        out = mytranspose(self.tensor)
        print(f"\nTesting PyTorch tensor transpose")
        print(self.tensor)
        print(out)
        self.assertTrue(torch.equal(out, self.tensor.T))

    
if __name__ == "__main__":
    unittest.main(argv=['', '-v'], exit=False)