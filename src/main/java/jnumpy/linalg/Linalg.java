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

package jnumpy.linalg;

import jnumpy.dtype.DType;
import jnumpy.ndarray.NDArray;
import jnumpy.util.Util;

public final class Linalg {

    private Linalg() {}

    public static NDArray dot(NDArray a, NDArray b) {
        if (a.ndim() == 1 && b.ndim() == 1) {
            if (a.size() != b.size()) throw new IllegalArgumentException("Incompatible sizes for dot product");
            double sum = 0;
            for (long i = 0; i < a.size(); i++) sum += Util.readElement(a, (int) i) * Util.readElement(b, (int) i);
            return NDArray.create(new double[] {sum});
        }
        if (a.ndim() == 2 && b.ndim() == 2) return matmul(a, b);
        if (a.ndim() == 1 && b.ndim() == 2) return dot(b, a.T());
        if (a.ndim() == 2 && b.ndim() == 1) {
            if (a.shape(1) != b.size()) throw new IllegalArgumentException("Incompatible shapes for dot");
            int m = a.shape(0), n = a.shape(1);
            NDArray result = NDArray.create(new int[] {m}, DType.FLOAT64);
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += Util.readElement(a, i, j) * Util.readElement(b, j);
                result.setDouble(sum, i);
            }
            return result;
        }
        throw new IllegalArgumentException("Unsupported dimensions for dot: " + a.ndim() + "D and " + b.ndim() + "D");
    }

    public static NDArray matmul(NDArray a, NDArray b) {
        int m = a.shape(0), k = a.shape(1), n = b.shape(1);
        if (k != b.shape(0)) throw new IllegalArgumentException("Incompatible shapes for matmul");
        NDArray result = NDArray.create(new int[] {m, n}, DType.FLOAT64);
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
        if (a.shape(a.ndim() - 1) != b.shape(b.ndim() - 1))
            throw new IllegalArgumentException("inner requires matching last dimensions: " + a.shape(a.ndim() - 1)
                    + " vs " + b.shape(b.ndim() - 1));
        int m = 1, n = 1, k = a.shape(a.ndim() - 1);
        for (int i = 0; i < a.ndim() - 1; i++) m *= a.shape(i);
        for (int i = 0; i < b.ndim() - 1; i++) n *= b.shape(i);
        NDArray result = NDArray.create(new int[] {m, n}, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int t = 0; t < k; t++) {
                    int ai = i * k + t;
                    int bj = j * k + t;
                    sum += Util.readElement(a, (int) (ai / a.shape(a.ndim() - 1)), ai % a.shape(a.ndim() - 1))
                            * Util.readElement(b, (int) (bj / b.shape(b.ndim() - 1)), bj % b.shape(b.ndim() - 1));
                }
                result.setDouble(sum, i, j);
            }
        }
        return result;
    }

    public static NDArray outer(NDArray a, NDArray b) {
        if (a.ndim() != 1 || b.ndim() != 1)
            throw new IllegalArgumentException("outer requires 1D arrays, got " + a.ndim() + "D and " + b.ndim() + "D");
        int m = (int) a.size(), n = (int) b.size();
        NDArray result = NDArray.create(new int[] {m, n}, DType.FLOAT64);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) result.setDouble(Util.readElement(a, i) * Util.readElement(b, j), i, j);
        return result;
    }

    public static NDArray cross(NDArray a, NDArray b) {
        if (a.size() != 3 || b.size() != 3) throw new IllegalArgumentException("cross requires size-3 vectors");
        NDArray result = NDArray.create(new int[] {3}, DType.FLOAT64);
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
        int n = a.shape(0);
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = Util.readElement(a, i, j);
            aug[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            double maxVal = Math.abs(aug[col][col]);
            for (int row = col + 1; row < n; row++) {
                double v = Math.abs(aug[row][col]);
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = row;
                }
            }
            if (maxVal < 1e-15)
                throw new IllegalArgumentException("Matrix is singular (zero pivot at column " + col + ")");
            double[] temp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = temp;
            double pivot = aug[col][col];
            for (int j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (int j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        NDArray result = NDArray.create(new int[] {n, n}, DType.FLOAT64);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Util.writeElement(result, aug[i][n + j], i, j);
        return result;
    }

    public static double det(NDArray a) {
        int n = a.shape(0);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) M[i][j] = Util.readElement(a, i, j);
        double det = 1;
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            for (int row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
            if (maxRow != col) {
                double[] temp = M[col];
                M[col] = M[maxRow];
                M[maxRow] = temp;
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
}
