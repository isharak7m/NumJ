package jnumpy.linalg;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Linalg {

    private Linalg() {}

    public static NDArray dot(NDArray a, NDArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) {
            if (a.size() != b.size()) throw new IllegalArgumentException("Incompatible sizes for dot product");
            double sum = 0;
            for (long i = 0; i < a.size(); i++) sum += a.getDouble(new int[]{ (int) i }) * b.getDouble(new int[]{ (int) i });
            return NDArray.create(new double[]{ sum });
        }
        if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b);
        if (a.ndim() == 2 && b.ndim() == 1) {
            int m = a.shape(0);
            int n = a.shape(1);
            NDArray result = NDArray.create(new int[]{ m }, DType.FLOAT64);
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += a.getDouble(i, j) * b.getDouble(j);
                result.setDouble(sum, i);
            }
            return result;
        }
        throw new IllegalArgumentException("Unsupported dimensions for dot: " + a.ndim() + " and " + b.ndim());
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        if (a.ndim() != 2 || b.ndim() != 2)
            throw new IllegalArgumentException("matmul currently supports 2D arrays only");
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(1);
        if (k != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for matmul");
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int t = 0; t < k; t++) sum += a.getDouble(i, t) * b.getDouble(t, j);
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray inner(NDArray a, NDArray b) {
        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int lastA = shapeA[shapeA.length - 1];
        int lastB = shapeB[shapeB.length - 1];
        int m = 1, n = 1;
        for (int i = 0; i < shapeA.length - 1; i++) m *= shapeA[i];
        for (int i = 0; i < shapeB.length - 1; i++) n *= shapeB[i];
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int t = 0; t < lastA; t++) {
                    double va = a.getDouble(indicesFromFlat(shapeA, i * lastA + t));
                    double vb = b.getDouble(indicesFromFlat(shapeB, j * lastB + t));
                    sum += va * vb;
                }
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray outer(NDArray a, NDArray b) {
        int m = (int) a.size();
        int n = (int) b.size();
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result.setDouble(a.getDouble(i) * b.getDouble(j), i, j);
        return result;
    }

    public static NDArray cross(NDArray a, NDArray b) {
        NDArray result = NDArray.create(new int[]{ 3 }, DType.FLOAT64);
        result.setDouble(a.getDouble(1) * b.getDouble(2) - a.getDouble(2) * b.getDouble(1), 0);
        result.setDouble(a.getDouble(2) * b.getDouble(0) - a.getDouble(0) * b.getDouble(2), 1);
        result.setDouble(a.getDouble(0) * b.getDouble(1) - a.getDouble(1) * b.getDouble(0), 2);
        return result;
    }

    public static NDArray kron(NDArray a, NDArray b) {
        if (a.ndim() != 2 || b.ndim() != 2)
            throw new IllegalArgumentException("kron currently supports 2D arrays only");
        int m = a.shape(0), n = a.shape(1);
        int p = b.shape(0), q = b.shape(1);
        NDArray result = NDArray.create(new int[]{ m * p, n * q }, DType.FLOAT64);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < p; k++)
                    for (int l = 0; l < q; l++)
                        result.setDouble(a.getDouble(i, j) * b.getDouble(k, l), i * p + k, j * q + l);
        return result;
    }

    public static double norm(NDArray a) {
        double sum = 0;
        int[] idx = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                idx[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            double v = a.getDouble(idx);
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    public static double trace(NDArray a) {
        if (a.ndim() != 2) throw new IllegalArgumentException("trace requires 2D array");
        int n = Math.min(a.shape(0), a.shape(1));
        double sum = 0;
        for (int i = 0; i < n; i++) sum += a.getDouble(i, i);
        return sum;
    }

    public static NDArray inv(NDArray a) {
        int n = a.shape(0);
        double[][] A = toMatrix(a);
        double[][] inv = invertMatrix(A);
        return fromMatrix(inv);
    }

    public static double det(NDArray a) {
        int n = a.shape(0);
        double[][] A = toMatrix(a);
        return determinant(A);
    }

    public static int rank(NDArray a) {
        double[][] A = toMatrix(a);
        double tol = Math.max(A.length, A[0].length) * 1e-15;
        double[][] svdA = svdDecompose(A, null, null);
        int r = 0;
        for (int i = 0; i < Math.min(A.length, A[0].length); i++) {
            if (svdA[i][i] > tol) r++;
        }
        return r;
    }

    public static NDArray qr(NDArray a) {
        int m = a.shape(0), n = a.shape(1);
        double[][] A = toMatrix(a);
        double[][] Q = new double[m][n];
        double[][] R = new double[n][n];
        qrDecompose(A, Q, R);
        NDArray qResult = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        NDArray rResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) qResult.setDouble(Q[i][j], i, j);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) rResult.setDouble(R[i][j], i, j);
        return qResult;
    }

    public static NDArray cholesky(NDArray a) {
        int n = a.shape(0);
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
                if (i == j) L[i][j] = Math.sqrt(a.getDouble(i, i) - sum);
                else L[i][j] = (a.getDouble(i, j) - sum) / L[j][j];
            }
        }
        NDArray result = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) result.setDouble(L[i][j], i, j);
        return result;
    }

    public static NDArray svd(NDArray a) {
        int m = a.shape(0), n = a.shape(1);
        double[][] A = toMatrix(a);
        double[][] U = new double[m][m];
        double[][] S = new double[m][n];
        double[][] Vt = new double[n][n];
        svdDecompose(A, U, S, Vt);
        NDArray uResult = NDArray.create(new int[]{ m, m }, DType.FLOAT64);
        NDArray sResult = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        NDArray vtResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) uResult.setDouble(U[i][j], i, j);
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) sResult.setDouble(S[i][j], i, j);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) vtResult.setDouble(Vt[i][j], i, j);
        return sResult;
    }

    public static NDArray eigen(NDArray a) {
        int n = a.shape(0);
        double[][] A = toMatrix(a);
        double[] eigenvalues = new double[n];
        double[][] eigenvectors = new double[n][n];
        powerIteration(A, eigenvalues, eigenvectors);
        NDArray valResult = NDArray.create(new double[]{ eigenvalues[0] });
        return valResult;
    }

    private static double[][] toMatrix(NDArray a) {
        int m = a.shape(0), n = a.shape(1);
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = a.getDouble(i, j);
        return result;
    }

    private static NDArray fromMatrix(double[][] m) {
        int rows = m.length, cols = m[0].length;
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.FLOAT64);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.setDouble(m[i][j], i, j);
        return result;
    }

    private static int[] indicesFromFlat(int[] shape, long flat) {
        int[] idx = new int[shape.length];
        long remaining = flat;
        for (int d = shape.length - 1; d >= 0; d--) {
            idx[d] = (int) (remaining % shape[d]);
            remaining /= shape[d];
        }
        return idx;
    }

    private static double[][] invertMatrix(double[][] A) {
        int n = A.length;
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
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
        return inv;
    }

    private static double determinant(double[][] A) {
        int n = A.length;
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(A[i], 0, M[i], 0, n);
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

    private static void qrDecompose(double[][] A, double[][] Q, double[][] R) {
        int m = A.length, n = A[0].length;
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) System.arraycopy(A[i], 0, a[i], 0, n);
        for (int k = 0; k < n; k++) {
            double norm = 0;
            for (int i = 0; i < m; i++) norm += a[i][k] * a[i][k];
            norm = Math.sqrt(norm);
            R[k][k] = norm;
            for (int i = 0; i < m; i++) Q[i][k] = a[i][k] / norm;
            for (int j = k + 1; j < n; j++) {
                double dot = 0;
                for (int i = 0; i < m; i++) dot += Q[i][k] * a[i][j];
                R[k][j] = dot;
                for (int i = 0; i < m; i++) a[i][j] -= dot * Q[i][k];
            }
        }
    }

    private static double[][] svdDecompose(double[][] A, double[][] U, double[][] Vt) {
        int m = A.length, n = A[0].length;
        int k = Math.min(m, n);
        double[][] S = new double[k][k];
        double[][] Uout = new double[m][k];
        double[][] Vout = new double[n][k];
        double[][] AtA = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int t = 0; t < m; t++)
                    AtA[i][j] += A[t][i] * A[t][j];
        double[][] eigenvectors = new double[n][n];
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            double[] vec = new double[n];
            vec[i] = 1;
            for (int iter = 0; iter < 100; iter++) {
                double[] newVec = new double[n];
                for (int j = 0; j < n; j++)
                    for (int t = 0; t < n; t++)
                        newVec[j] += AtA[j][t] * vec[t];
                double norm = 0;
                for (double v : newVec) norm += v * v;
                norm = Math.sqrt(norm);
                for (int j = 0; j < n; j++) newVec[j] /= norm;
                double diff = 0;
                for (int j = 0; j < n; j++) diff += Math.abs(newVec[j] - vec[j]);
                vec = newVec;
                if (diff < 1e-12) break;
            }
            double eig = 0;
            for (int j = 0; j < n; j++)
                for (int t = 0; t < n; t++)
                    eig += vec[j] * AtA[j][t] * vec[t];
            eigenvalues[i] = eig;
            for (int j = 0; j < n; j++) eigenvectors[j][i] = vec[j];
            for (int j = 0; j < n; j++)
                for (int t = 0; t < n; t++)
                    AtA[j][t] -= eig * vec[j] * vec[t];
        }
        for (int i = 0; i < k; i++) {
            double sigma = Math.sqrt(Math.max(0, eigenvalues[i]));
            S[i][i] = sigma;
            for (int j = 0; j < m; j++) Uout[j][i] = sigma > 1e-15 ? A[j][0] / sigma : 0;
            for (int j = 0; j < n; j++) Vout[j][i] = eigenvectors[j][i];
        }
        if (U != null) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    U[i][j] = j < k ? Uout[i][j] : (i == j ? 1 : 0);
        }
        if (Vt != null) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Vt[i][j] = j < k ? Vout[i][j] : (i == j ? 1 : 0);
        }
        return new double[k][k];
    }

    private static void powerIteration(double[][] A, double[] eigenvalues, double[][] eigenvectors) {
        int n = A.length;
        double[][] Ak = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(A[i], 0, Ak[i], 0, n);
        for (int eig = 0; eig < n; eig++) {
            double[] v = new double[n];
            v[eig] = 1;
            for (int iter = 0; iter < 200; iter++) {
                double[] Av = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Av[i] += Ak[i][j] * v[j];
                double norm = 0;
                for (double val : Av) norm += val * val;
                norm = Math.sqrt(norm);
                if (norm < 1e-15) break;
                for (int i = 0; i < n; i++) v[i] = Av[i] / norm;
            }
            double eigVal = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    eigVal += v[i] * Ak[i][j] * v[j];
            eigenvalues[eig] = eigVal;
            for (int i = 0; i < n; i++) eigenvectors[i][eig] = v[i];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Ak[i][j] -= eigVal * v[i] * v[j];
        }
    }

    private static void svdDecompose(double[][] A, double[][] U, double[][] S, double[][] Vt) {
        int m = A.length, n = A[0].length;
        S = svdDecompose(A, U, Vt);
    }
}
