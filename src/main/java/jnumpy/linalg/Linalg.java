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
            int[] ai = new int[1];
            int[] bi = new int[1];
            for (long i = 0; i < a.size(); i++) { ai[0] = (int) i; bi[0] = (int) i; sum += Util.readElement(a, ai) * Util.readElement(b, bi); }
            return NDArray.create(new double[]{ sum });
        }
        if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b);
        if (a.ndim() == 2 && b.ndim() == 1) {
            if (a.shape(1) != b.size()) throw new IllegalArgumentException("Incompatible shapes for dot");
            int m = a.shape(0);
            int n = a.shape(1);
            NDArray result = NDArray.create(new int[]{ m }, DType.FLOAT64);
            int[] ai = new int[2];
            int[] bi = new int[1];
            for (int i = 0; i < m; i++) {
                ai[0] = i;
                double sum = 0;
                for (int j = 0; j < n; j++) { ai[1] = j; bi[0] = j; sum += Util.readElement(a, ai) * Util.readElement(b, bi); }
                result.setDouble(sum, i);
            }
            return result;
        }
        if (a.ndim() == 1 && b.ndim() == 2) {
            if (a.size() != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for dot");
            int n = b.shape(1);
            NDArray result = NDArray.create(new int[]{ n }, DType.FLOAT64);
            int[] ai = new int[1];
            int[] bi = new int[2];
            for (int j = 0; j < n; j++) {
                bi[1] = j;
                double sum = 0;
                for (int i = 0; i < a.size(); i++) { ai[0] = i; bi[0] = i; sum += Util.readElement(a, ai) * Util.readElement(b, bi); }
                result.setDouble(sum, j);
            }
            return result;
        }
        throw new IllegalArgumentException("Unsupported dimensions for dot: " + a.ndim() + "D and " + b.ndim() + "D");
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        check2D(a);
        check2D(b);
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(1);
        if (k != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for matmul");
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        int[] ai = new int[2];
        int[] bi = new int[2];
        for (int i = 0; i < m; i++) {
            ai[0] = i;
            for (int j = 0; j < n; j++) {
                bi[1] = j;
                double sum = 0;
                for (int t = 0; t < k; t++) { ai[1] = t; bi[0] = t; sum += Util.readElement(a, ai) * Util.readElement(b, bi); }
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray inner(NDArray a, NDArray b) {
        if (a.ndim() == 0 || b.ndim() == 0)
            throw new IllegalArgumentException("inner requires at least 1D arrays");
        int lastA = a.shape(a.ndim() - 1);
        int lastB = b.shape(b.ndim() - 1);
        if (lastA != lastB)
            throw new IllegalArgumentException("inner requires matching last dimensions: " + lastA + " vs " + lastB);
        int m = 1, n = 1;
        for (int i = 0; i < a.ndim() - 1; i++) m *= a.shape(i);
        for (int i = 0; i < b.ndim() - 1; i++) n *= b.shape(i);
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int t = 0; t < lastA; t++) {
                    double va = Util.readElement(a, indicesFromFlat(a.shape(), (long) i * lastA + t));
                    double vb = Util.readElement(b, indicesFromFlat(b.shape(), (long) j * lastB + t));
                    sum += va * vb;
                }
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray outer(NDArray a, NDArray b) {
        if (a.ndim() != 1 || b.ndim() != 1)
            throw new IllegalArgumentException("outer requires 1D arrays, got " + a.ndim() + "D and " + b.ndim() + "D");
        int m = (int) a.size();
        int n = (int) b.size();
        NDArray result = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        int[] ai = new int[1];
        int[] bi = new int[1];
        for (int i = 0; i < m; i++) {
            ai[0] = i;
            for (int j = 0; j < n; j++) {
                bi[0] = j;
                result.setDouble(Util.readElement(a, ai) * Util.readElement(b, bi), i, j);
            }
        }
        return result;
    }

    public static NDArray cross(NDArray a, NDArray b) {
        if (a.ndim() != 1 || b.ndim() != 1)
            throw new IllegalArgumentException("cross requires 1D arrays");
        if (a.size() != 3 || b.size() != 3)
            throw new IllegalArgumentException("cross requires size-3 vectors");
        NDArray result = NDArray.create(new int[]{ 3 }, DType.FLOAT64);
        double a0 = Util.readElement(a, new int[]{ 0 }), a1 = Util.readElement(a, new int[]{ 1 }), a2 = Util.readElement(a, new int[]{ 2 });
        double b0 = Util.readElement(b, new int[]{ 0 }), b1 = Util.readElement(b, new int[]{ 1 }), b2 = Util.readElement(b, new int[]{ 2 });
        result.setDouble(a1 * b2 - a2 * b1, 0);
        result.setDouble(a2 * b0 - a0 * b2, 1);
        result.setDouble(a0 * b1 - a1 * b0, 2);
        return result;
    }

    public static NDArray kron(NDArray a, NDArray b) {
        check2D(a);
        check2D(b);
        int m = a.shape(0), n = a.shape(1);
        int p = b.shape(0), q = b.shape(1);
        NDArray result = NDArray.create(new int[]{ m * p, n * q }, DType.FLOAT64);
        int[] ai = new int[2];
        int[] bi = new int[2];
        for (int i = 0; i < m; i++) {
            ai[0] = i;
            for (int j = 0; j < n; j++) {
                ai[1] = j;
                for (int k = 0; k < p; k++) {
                    bi[0] = k;
                    for (int l = 0; l < q; l++) {
                        bi[1] = l;
                        result.setDouble(Util.readElement(a, ai) * Util.readElement(b, bi), i * p + k, j * q + l);
                    }
                }
            }
        }
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
            double v = Util.readElement(a, idx);
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    public static double trace(NDArray a) {
        check2D(a);
        int n = Math.min(a.shape(0), a.shape(1));
        double sum = 0;
        int[] idx = new int[2];
        for (int i = 0; i < n; i++) { idx[0] = i; idx[1] = i; sum += Util.readElement(a, idx); }
        return sum;
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

    public static NDArray[] lu(NDArray a) {
        checkSquare(a);
        int n = a.shape(0);
        double[][] M = toMatrix(a);
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];
        for (int i = 0; i < n; i++) L[i][i] = 1.0;
        for (int i = 0; i < n; i++) System.arraycopy(M[i], 0, U[i], 0, n);
        int[] piv = new int[n];
        for (int i = 0; i < n; i++) piv[i] = i;
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            double maxVal = Math.abs(U[col][col]);
            for (int row = col + 1; row < n; row++) {
                double v = Math.abs(U[row][col]);
                if (v > maxVal) { maxVal = v; maxRow = row; }
            }
            if (maxRow != col) {
                double[] temp = U[col]; U[col] = U[maxRow]; U[maxRow] = temp;
                for (int j = 0; j < col; j++) {
                    double t = L[col][j]; L[col][j] = L[maxRow][j]; L[maxRow][j] = t;
                }
                int t = piv[col]; piv[col] = piv[maxRow]; piv[maxRow] = t;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = U[row][col] / U[col][col];
                L[row][col] = factor;
                for (int j = col; j < n; j++) {
                    U[row][j] -= factor * U[col][j];
                }
            }
        }
        NDArray lResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        NDArray uResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Util.writeElement(lResult, L[i][j], i, j);
                Util.writeElement(uResult, U[i][j], i, j);
            }
        }
        return new NDArray[]{ lResult, uResult };
    }

    public static double det(NDArray a) {
        checkSquare(a);
        int n = a.shape(0);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) M[i][j] = Util.readElement(a, i, j);
        double det = 1;
        double tol = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) tol = Math.max(tol, Math.abs(M[i][j]));
        tol = tol * n * 1e-15;
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
            if (maxRow != col) {
                double[] temp = M[col]; M[col] = M[maxRow]; M[maxRow] = temp;
                det = -det;
            }
            double pivot = M[col][col];
            if (Math.abs(pivot) < tol) return 0;
            det *= pivot;
            for (int row = col + 1; row < n; row++) {
                double factor = M[row][col] / pivot;
                for (int j = col; j < n; j++) M[row][j] -= factor * M[col][j];
            }
        }
        return det;
    }

    public static int rank(NDArray a) {
        check2D(a);
        int m = a.shape(0), n = a.shape(1);
        double[][] A = toMatrix(a);
        double[][] U = new double[m][m];
        double[][] S = new double[m][n];
        double[][] Vt = new double[n][n];
        svdDecompose(A, U, S, Vt);
        double tol = Math.max(m, n) * S[0][0] * 1e-15;
        int r = 0;
        for (int i = 0; i < Math.min(m, n); i++) {
            if (S[i][i] > tol) r++;
        }
        return r;
    }

    public static NDArray[] qr(NDArray a) {
        check2D(a);
        int m = a.shape(0), n = a.shape(1);
        double[][] A = toMatrix(a);
        double[][] Q = new double[m][m];
        double[][] R = new double[m][n];
        qrDecompose(A, Q, R);
        NDArray qResult = NDArray.create(new int[]{ m, m }, DType.FLOAT64);
        NDArray rResult = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) qResult.setDouble(Q[i][j], i, j);
            for (int j = 0; j < n; j++) rResult.setDouble(R[i][j], i, j);
        }
        return new NDArray[]{ qResult, rResult };
    }

    public static NDArray cholesky(NDArray a) {
        checkSquare(a);
        int[] idx1 = new int[2];
        int[] idx2 = new int[2];
        for (int i = 0; i < a.shape(0); i++) {
            idx1[0] = i;
            for (int j = 0; j < a.shape(1); j++) {
                idx1[1] = j; idx2[0] = j; idx2[1] = i;
                if (Math.abs(Util.readElement(a, idx1) - Util.readElement(a, idx2)) > 1e-15)
                    throw new IllegalArgumentException("Matrix is not symmetric");
            }
        }
        int n = a.shape(0);
        double[][] L = new double[n][n];
        int[] idx = new int[2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
                if (i == j) {
                    idx[0] = i; idx[1] = i;
                    double val = Util.readElement(a, idx) - sum;
                    if (val <= 0)
                        throw new IllegalArgumentException("Matrix is not positive definite");
                    L[i][j] = Math.sqrt(val);
                } else {
                    idx[0] = i; idx[1] = j;
                    L[i][j] = (Util.readElement(a, idx) - sum) / L[j][j];
                }
            }
        }
        NDArray result = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) result.setDouble(L[i][j], i, j);
        return result;
    }

    public static NDArray[] svd(NDArray a) {
        check2D(a);
        int m = a.shape(0), n = a.shape(1);
        double[][] A = toMatrix(a);
        double[][] U = new double[m][m];
        double[][] S = new double[m][n];
        double[][] Vt = new double[n][n];
        svdDecompose(A, U, S, Vt);
        NDArray uResult = NDArray.create(new int[]{ m, m }, DType.FLOAT64);
        NDArray sResult = NDArray.create(new int[]{ m, n }, DType.FLOAT64);
        NDArray vtResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) uResult.setDouble(U[i][j], i, j);
            for (int j = 0; j < n; j++) sResult.setDouble(S[i][j], i, j);
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) vtResult.setDouble(Vt[i][j], i, j);
        return new NDArray[]{ uResult, sResult, vtResult };
    }

    public static NDArray[] eigen(NDArray a) {
        checkSquare(a);
        int n = a.shape(0);
        double[][] A = toMatrix(a);
        double[] eigenvalues = new double[n];
        double[][] eigenvectors = new double[n][n];
        powerIteration(A, eigenvalues, eigenvectors);
        NDArray valResult = NDArray.create(new int[]{ n }, DType.FLOAT64);
        NDArray vecResult = NDArray.create(new int[]{ n, n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) {
            valResult.setDouble(eigenvalues[i], i);
            for (int j = 0; j < n; j++) vecResult.setDouble(eigenvectors[j][i], j, i);
        }
        return new NDArray[]{ valResult, vecResult };
    }

    private static double[][] toMatrix(NDArray a) {
        check2D(a);
        int m = a.shape(0), n = a.shape(1);
        double[][] result = new double[m][n];
        int[] idx = new int[2];
        for (int i = 0; i < m; i++) {
            idx[0] = i;
            for (int j = 0; j < n; j++) {
                idx[1] = j;
                result[i][j] = Util.readElement(a, idx);
            }
        }
        return result;
    }

    private static NDArray fromMatrix(double[][] m) {
        int rows = m.length, cols = m[0].length;
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.FLOAT64);
        int[] idx = new int[2];
        for (int i = 0; i < rows; i++) {
            idx[0] = i;
            for (int j = 0; j < cols; j++) {
                idx[1] = j;
                Util.writeElement(result, m[i][j], idx);
            }
        }
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

    private static void qrDecompose(double[][] A, double[][] Q, double[][] R) {
        int m = A.length, n = A[0].length;
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) System.arraycopy(A[i], 0, a[i], 0, n);
        for (int k = 0; k < n; k++) {
            double norm = 0;
            for (int i = 0; i < m; i++) norm += a[i][k] * a[i][k];
            norm = Math.sqrt(norm);
            if (norm < 1e-15) {
                R[k][k] = 0;
                for (int i = 0; i < m; i++) Q[i][k] = 0;
                Q[k][k] = 1;
                continue;
            }
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

    private static void svdDecompose(double[][] A, double[][] U, double[][] S, double[][] Vt) {
        int m = A.length, n = A[0].length;
        int k = Math.min(m, n);

        double[][] AtA = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int t = 0; t < m; t++)
                    AtA[i][j] += A[t][i] * A[t][j];

        double[] eigenvalues = new double[n];
        double[][] eigenvectors = new double[n][n];
        double[][] Ak = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(AtA[i], 0, Ak[i], 0, n);

        for (int eig = 0; eig < n; eig++) {
            double[] vec = new double[n];
            vec[eig] = 1;
            for (int iter = 0; iter < 100; iter++) {
                double[] newVec = new double[n];
                for (int j = 0; j < n; j++)
                    for (int t = 0; t < n; t++)
                        newVec[j] += Ak[j][t] * vec[t];
                double norm = 0;
                for (double v : newVec) norm += v * v;
                norm = Math.sqrt(norm);
                if (norm < 1e-15) break;
                for (int j = 0; j < n; j++) newVec[j] /= norm;
                double diff = 0;
                for (int j = 0; j < n; j++) diff += Math.abs(newVec[j] - vec[j]);
                vec = newVec;
                if (diff < 1e-12) break;
            }
            double eigVal = 0;
            for (int j = 0; j < n; j++)
                for (int t = 0; t < n; t++)
                    eigVal += vec[j] * AtA[j][t] * vec[t];
            eigenvalues[eig] = eigVal;
            for (int j = 0; j < n; j++) eigenvectors[j][eig] = vec[j];
            for (int j = 0; j < n; j++)
                for (int t = 0; t < n; t++)
                    Ak[j][t] -= eigVal * vec[j] * vec[t];
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (eigenvalues[i] < eigenvalues[j]) {
                    double tmp = eigenvalues[i]; eigenvalues[i] = eigenvalues[j]; eigenvalues[j] = tmp;
                    for (int t = 0; t < n; t++) {
                        double tv = eigenvectors[t][i]; eigenvectors[t][i] = eigenvectors[t][j]; eigenvectors[t][j] = tv;
                    }
                }
            }
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                U[i][j] = i == j ? 1.0 : 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Vt[i][j] = i == j ? 1.0 : 0.0;

        for (int i = 0; i < k; i++) {
            double sigma = Math.sqrt(Math.max(0, eigenvalues[i]));
            S[i][i] = sigma;
            for (int j = 0; j < n; j++) Vt[j][i] = eigenvectors[j][i];
            if (sigma > 1e-15) {
                double[] uCol = new double[m];
                for (int j = 0; j < m; j++) uCol[j] = 0;
                for (int t = 0; t < n; t++)
                    for (int j = 0; j < m; j++)
                        uCol[j] += A[j][t] * eigenvectors[t][i];
                for (int j = 0; j < m; j++) U[j][i] = uCol[j] / sigma;
            }
        }

        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                if (S[i][i] < S[j][j]) {
                    double tmp = S[i][i]; S[i][i] = S[j][j]; S[j][j] = tmp;
                    for (int t = 0; t < m; t++) { double tv = U[t][i]; U[t][i] = U[t][j]; U[t][j] = tv; }
                    for (int t = 0; t < n; t++) { double tv = Vt[t][i]; Vt[t][i] = Vt[t][j]; Vt[t][j] = tv; }
                }
            }
        }

        for (int i = k; i < m; i++) U[i][i] = 1.0;
        for (int i = k; i < n; i++) Vt[i][i] = 1.0;
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
}
