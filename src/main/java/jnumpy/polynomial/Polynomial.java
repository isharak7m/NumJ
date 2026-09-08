package jnumpy.polynomial;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.util.Util;

public final class Polynomial {

    private Polynomial() {}

    public static NDArray polyfit(NDArray x, NDArray y, int deg) {
        int n = (int) x.size();
        int m = deg + 1;
        double[][] A = new double[n][m];
        for (int i = 0; i < n; i++) {
            double xi = Util.readElement(x, i);
            double pow = 1;
            for (int j = 0; j < m; j++) {
                A[i][j] = pow;
                pow *= xi;
            }
        }
        double[][] AtA = new double[m][m];
        double[] AtY = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < n; k++) AtA[i][j] += A[k][i] * A[k][j];
            }
            for (int k = 0; k < n; k++) AtY[i] += A[k][i] * Util.readElement(y, k);
        }
        double[] coeffs = solveLinear(AtA, AtY);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, m);
        for (int i = 0; i < m; i++) buf.setDouble(i, coeffs[i]);
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray polyval(NDArray p, NDArray x) {
        int m = (int) p.size();
        NDArray result = NDArray.create(x.shape(), DType.FLOAT64);
        int[] indices = new int[x.ndim()];
        for (long i = 0; i < x.size(); i++) {
            long remaining = i;
            for (int d = x.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % x.shape()[d]);
                remaining /= x.shape()[d];
            }
            double xi = Util.readElement(x, indices);
            double val = 0;
            for (int j = m - 1; j >= 0; j--) val = val * xi + Util.readElement(p, j);
            Util.writeElement(result, val, indices);
        }
        return result;
    }

    public static NDArray roots(NDArray p) {
        int n = (int) p.size() - 1;
        if (n == 0) return NDArray.create(new double[0]);
        if (n == 1) return NDArray.create(new double[]{ -Util.readElement(p, 0) / Util.readElement(p, 1) });
        double[][] companion = new double[n][n];
        double a0 = Util.readElement(p, 0);
        for (int i = 0; i < n; i++) {
            if (i < n - 1) companion[i + 1][i] = 1;
            companion[i][n - 1] = -Util.readElement(p, i) / a0;
        }
        double[] eigenvalues = qrEigenvalues(companion, n);
        return new NDArray(MemoryBuffer.wrap(eigenvalues), DType.FLOAT64);
    }

    private static double[] qrEigenvalues(double[][] a, int n) {
        double[][] h = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], 0, h[i], 0, n);
        hessenbergReduce(h, n);
        double[] result = new double[n];
        int m = n;
        int maxIter = 200 * n;
        int iter = 0;
        while (m > 0 && iter < maxIter) {
            iter++;
            double eps = 1e-10 * (Math.abs(h[m-1][m-1]) + 1e-30);
            if (m == 1 || Math.abs(h[m-1][m-2]) <= eps) {
                result[m-1] = h[m-1][m-1];
                m--;
                iter = 0;
                continue;
            }
            if (m == 2 || Math.abs(h[m-2][m-3]) <= eps) {
                double tr = h[m-2][m-2] + h[m-1][m-1];
                double det = h[m-2][m-2] * h[m-1][m-1] - h[m-2][m-1] * h[m-1][m-2];
                double disc = tr * tr - 4 * det;
                if (disc >= 0) {
                    double sq = Math.sqrt(disc);
                    result[m-2] = (tr + sq) / 2;
                    result[m-1] = (tr - sq) / 2;
                } else {
                    result[m-2] = tr / 2;
                    result[m-1] = tr / 2;
                }
                m -= 2;
                iter = 0;
                continue;
            }
            double shift = h[m-1][m-1];
            double subDiag = h[m-1][m-2];
            for (int i = 0; i < m; i++) h[i][i] -= shift;
            qrDecomize(h, m);
            for (int i = 0; i < m; i++) h[i][i] += shift;
        }
        for (int i = 0; i < m; i++) result[i] = h[i][i];
        return result;
    }

    private static void hessenbergReduce(double[][] a, int n) {
        for (int k = 0; k < n - 2; k++) {
            double maxVal = 0;
            for (int i = k + 1; i < n; i++) maxVal = Math.max(maxVal, Math.abs(a[i][k]));
            if (maxVal < 1e-15) continue;
            double[] u = new double[n];
            for (int i = k + 1; i < n; i++) u[i] = a[i][k] / (a[k+1][k] >= 0 ? maxVal : -maxVal);
            double sigma = 0;
            for (int i = k + 1; i < n; i++) sigma += u[i] * u[i];
            sigma = Math.sqrt(sigma);
            if (u[k+1] >= 0) sigma = -sigma;
            u[k+1] -= sigma;
            double normU = 0;
            for (int i = k + 1; i < n; i++) normU += u[i] * u[i];
            if (normU < 1e-30) continue;
            for (int j = k; j < n; j++) {
                double dot = 0;
                for (int i = k + 1; i < n; i++) dot += u[i] * a[i][j];
                for (int i = k + 1; i < n; i++) a[i][j] -= 2 * u[i] * dot / normU;
            }
            for (int i = 0; i < n; i++) {
                double dot = 0;
                for (int j = k + 1; j < n; j++) dot += a[i][j] * u[j];
                for (int j = k + 1; j < n; j++) a[i][j] -= 2 * dot * u[j] / normU;
            }
        }
    }

    private static void qrDecomize(double[][] a, int n) {
        for (int k = 0; k < n - 1; k++) {
            double x = a[k][k];
            double y = a[k+1][k];
            double r = Math.hypot(x, y);
            if (r < 1e-15) continue;
            double c = x / r;
            double s = y / r;
            for (int j = k; j < n; j++) {
                double t1 = a[k][j];
                double t2 = a[k+1][j];
                a[k][j] = c * t1 + s * t2;
                a[k+1][j] = -s * t1 + c * t2;
            }
            for (int i = 0; i < Math.min(k + 3, n); i++) {
                double t1 = a[i][k];
                double t2 = a[i][k+1];
                a[i][k] = c * t1 + s * t2;
                a[i][k+1] = -s * t1 + c * t2;
            }
        }
    }

    public static NDArray polyadd(NDArray a, NDArray b) {
        int n = Math.max((int) a.size(), (int) b.size());
        double[] result = new double[n];
        for (int i = 0; i < a.size(); i++) result[i] += Util.readElement(a, i);
        for (int i = 0; i < b.size(); i++) result[i] += Util.readElement(b, i);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray polymul(NDArray a, NDArray b) {
        int na = (int) a.size(), nb = (int) b.size();
        double[] result = new double[na + nb - 1];
        for (int i = 0; i < na; i++)
            for (int j = 0; j < nb; j++)
                result[i + j] += Util.readElement(a, i) * Util.readElement(b, j);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray polydiv(NDArray a, NDArray b) {
        int na = (int) a.size(), nb = (int) b.size();
        if (na < nb) return NDArray.create(new double[0]);
        double[] dividend = new double[na];
        for (int i = 0; i < na; i++) dividend[i] = Util.readElement(a, i);
        double[] divisor = new double[nb];
        for (int i = 0; i < nb; i++) divisor[i] = Util.readElement(b, i);
        int nq = na - nb + 1;
        double[] quotient = new double[nq];
        for (int i = 0; i < nq; i++) {
            if (Math.abs(dividend[na - 1 - i]) < 1e-15) continue;
            quotient[nq - 1 - i] = dividend[na - 1 - i] / divisor[nb - 1];
            for (int j = 0; j < nb; j++)
                dividend[na - 1 - i - j] -= quotient[nq - 1 - i] * divisor[nb - 1 - j];
        }
        return new NDArray(MemoryBuffer.wrap(quotient), DType.FLOAT64);
    }

    private static double[] solveLinear(double[][] A, double[] b) {
        int n = b.length;
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
            double[] temp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = temp;
            double pivot = aug[col][col];
            for (int j = col; j <= n; j++) aug[col][j] /= pivot;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (int j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = aug[i][n];
        return x;
    }
}
