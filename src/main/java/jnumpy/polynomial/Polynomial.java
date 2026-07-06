package jnumpy.polynomial;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Polynomial {

    private Polynomial() {}

    public static NDArray polyfit(NDArray x, NDArray y, int deg) {
        int n = (int) x.size();
        int m = deg + 1;
        double[][] A = new double[n][m];
        for (int i = 0; i < n; i++) {
            double xi = x.getDouble(i);
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
            for (int k = 0; k < n; k++) AtY[i] += A[k][i] * y.getDouble(k);
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
            double xi = x.getDouble(indices);
            double val = 0;
            for (int j = m - 1; j >= 0; j--) val = val * xi + p.getDouble(j);
            result.setDouble(val, indices);
        }
        return result;
    }

    public static NDArray roots(NDArray p) {
        int n = (int) p.size() - 1;
        if (n == 0) return NDArray.create(new double[0]);
        if (n == 1) return NDArray.create(new double[]{ -p.getDouble(0) / p.getDouble(1) });
        double[][] companion = new double[n][n];
        double a0 = p.getDouble(0);
        for (int i = 0; i < n; i++) {
            if (i < n - 1) companion[i + 1][i] = 1;
            companion[i][n - 1] = -p.getDouble(i) / a0;
        }
        java.util.ArrayList<Double> eigVals = new java.util.ArrayList<>();
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) re[i] = companion[i][i];
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = re[i];
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray polyadd(NDArray a, NDArray b) {
        int n = Math.max((int) a.size(), (int) b.size());
        double[] result = new double[n];
        for (int i = 0; i < a.size(); i++) result[i] += a.getDouble(i);
        for (int i = 0; i < b.size(); i++) result[i] += b.getDouble(i);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray polymul(NDArray a, NDArray b) {
        int na = (int) a.size(), nb = (int) b.size();
        double[] result = new double[na + nb - 1];
        for (int i = 0; i < na; i++)
            for (int j = 0; j < nb; j++)
                result[i + j] += a.getDouble(i) * b.getDouble(j);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray polydiv(NDArray a, NDArray b) {
        int na = (int) a.size(), nb = (int) b.size();
        if (na < nb) return NDArray.create(new double[0]);
        double[] dividend = new double[na];
        for (int i = 0; i < na; i++) dividend[i] = a.getDouble(i);
        double[] divisor = new double[nb];
        for (int i = 0; i < nb; i++) divisor[i] = b.getDouble(i);
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
