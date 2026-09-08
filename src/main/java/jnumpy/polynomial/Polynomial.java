/*
 * Copyright 2026 JNumj Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jnumpy.polynomial;

import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.ndarray.NDArray;
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
            for (int j = 0; j < nb; j++) result[i + j] += Util.readElement(a, i) * Util.readElement(b, j);
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
            for (int j = 0; j < nb; j++) dividend[na - 1 - i - j] -= quotient[nq - 1 - i] * divisor[nb - 1 - j];
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
            double[] temp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = temp;
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
