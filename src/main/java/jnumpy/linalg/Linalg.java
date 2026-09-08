package jnumpy.linalg;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.util.Util;

public final class Linalg {

    private Linalg() {}

    private static void checkSquare(NDArray a) {
        if (a.ndim() != 2 || a.shape(0) != a.shape(1))
            throw new IllegalArgumentException("Expected square matrix, got shape [" + a.shape(0) + ", " + a.shape(1) + "]");
    }

    private static void check2D(NDArray a) {
        if (a.ndim() != 2)
            throw new IllegalArgumentException("Expected 2D array, got " + a.ndim() + "D");
    }

    public static NDArray dot(NDArray a, NDArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) {
            if (a.size() != b.size()) throw new IllegalArgumentException("Incompatible sizes for dot product");
            double sum = 0;
            for (long i = 0; i < a.size(); i++) sum += Util.readElement(a, (int) i) * Util.readElement(b, (int) i);
            return NDArray.create(new double[]{ sum });
        }
        if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b);
        if (a.ndim() == 2 && b.ndim() == 1) {
            if (a.shape(1) != b.size()) throw new IllegalArgumentException("Incompatible shapes for dot");
            int m = a.shape(0), n = a.shape(1);
            NDArray result = NDArray.create(new int[]{ m }, DType.FLOAT64);
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += Util.readElement(a, i, j) * Util.readElement(b, j);
                result.setDouble(sum, i);
            }
            return result;
        }
        if (a.ndim() == 1 && b.ndim() == 2) {
            if (a.size() != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for dot");
            int n = b.shape(1);
            NDArray result = NDArray.create(new int[]{ n }, DType.FLOAT64);
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int i = 0; i < a.size(); i++) sum += Util.readElement(a, i) * Util.readElement(b, i, j);
                result.setDouble(sum, j);
            }
            return result;
        }
        throw new IllegalArgumentException("Unsupported dimensions for dot: " + a.ndim() + "D and " + b.ndim() + "D");
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        check2D(a);
        check2D(b);
        int m = a.shape(0), k = a.shape(1), n = b.shape(1);
        if (k != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for matmul");
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int t = 0; t < k; t++) sum += Util.readElement(a, i, t) * Util.readElement(b, t, j);
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray inner(NDArray a, NDArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) return dot(a, b);
        if (a.ndim() == 2 && b.ndim() == 2) {
            if (a.shape(1) != b.shape(1))
                throw new IllegalArgumentException("inner requires matching last dimensions: " + a.shape(1) + " vs " + b.shape(1));
            int m = a.shape(0), n = b.shape(0), k = a.shape(1);
            NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int t = 0; t < k; t++) sum += Util.readElement(a, i, t) * Util.readElement(b, j, t);
                    result.setDouble(sum, i, j);
                }
            }
            return result;
        }
        throw new IllegalArgumentException("inner requires 1D or 2D arrays, got " + a.ndim() + "D and " + b.ndim() + "D");
    }

    public static NDArray outer(NDArray a, NDArray b) {
        if (a.ndim() != 1 || b.ndim() != 1)
            throw new IllegalArgumentException("outer requires 1D arrays, got " + a.ndim() + "D and " + b.ndim() + "D");
        int m = (int) a.size(), n = (int) b.size();
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result.setDouble(Util.readElement(a, i) * Util.readElement(b, j), i, j);
        return result;
    }

    public static NDArray cross(NDArray a, NDArray b) {
        if (a.ndim() != 1 || b.ndim() != 1)
            throw new IllegalArgumentException("cross requires 1D arrays");
        if (a.size() != 3 || b.size() != 3)
            throw new IllegalArgumentException("cross requires size-3 vectors");
        NDArray result = NDArray.create(new int[]{ 3 }, DType.FLOAT64);
        double a0 = Util.readElement(a, 0), a1 = Util.readElement(a, 1), a2 = Util.readElement(a, 2);
        double b0 = Util.readElement(b, 0), b1 = Util.readElement(b, 1), b2 = Util.readElement(b, 2);
        result.setDouble(a1 * b2 - a2 * b1, 0);
        result.setDouble(a2 * b0 - a0 * b2, 1);
        result.setDouble(a0 * b1 - a1 * b0, 2);
        return result;
    }

    public static double norm(NDArray a) {
        double sum = 0;
        for (long i = 0; i < a.size(); i++) {
            double v = Util.readElement(a, (int) i);
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    public static NDArray inv(NDArray a) {
        checkSquare(a);
        int n = a.shape(0);
        double[][] A = toMatrix(a);
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            double maxVal = Math.abs(aug[col][col]);
            for (int row = col + 1; row < n; row++) {
                double v = Math.abs(aug[row][col]);
                if (v > maxVal) { maxVal = v; maxRow = row; }
            }
            if (maxVal < 1e-15)
                throw new IllegalArgumentException("Matrix is singular (zero pivot at column " + col + ")");
            double[] temp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = temp;
            double pivot = aug[col][col];
            for (int j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (int j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv[i][j] = aug[i][n + j];
        return fromMatrix(inv);
    }

    public static double det(NDArray a) {
        checkSquare(a);
        int n = a.shape(0);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) M[i][j] = Util.readElement(a, i, j);
        double det = 1;
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
            if (maxRow != col) {
                double[] temp = M[col]; M[col] = M[maxRow]; M[maxRow] = temp;
                det = -det;
            }
            double pivot = M[col][col];
            if (Math.abs(pivot) < 1e-15) return 0;
            det *= pivot;
            for (int row = col + 1; row < n; row++) {
                double factor = M[row][col] / pivot;
                for (int j = col; j < n; j++) M[row][j] -= factor * M[col][j];
            }
        }
        return det;
    }

    private static double[][] toMatrix(NDArray a) {
        check2D(a);
        int m = a.shape(0), n = a.shape(1);
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = Util.readElement(a, i, j);
        return result;
    }

    private static NDArray fromMatrix(double[][] m) {
        int rows = m.length, cols = m[0].length;
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.FLOAT64);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Util.writeElement(result, m[i][j], i, j);
        return result;
    }
}
