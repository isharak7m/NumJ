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

package jnumpy.creation;

import java.util.function.IntToDoubleFunction;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.ndarray.NDArray;

public final class Creation {

    private Creation() {}

    public static NDArray array(Object data) {
        if (data instanceof NDArray nd) return nd;
        if (data instanceof double[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.FLOAT64);
        }
        if (data instanceof float[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.FLOAT32);
        }
        if (data instanceof int[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.INT32);
        }
        if (data instanceof long[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.INT64);
        }
        if (data instanceof byte[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.INT8);
        }
        if (data instanceof short[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.INT16);
        }
        if (data instanceof boolean[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.BOOL);
        }
        if (data instanceof char[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.CHAR);
        }
        if (data instanceof String[] arr) {
            MemoryBuffer buf = MemoryBuffer.wrap(arr);
            return new NDArray(buf, DType.STRING);
        }
        if (data instanceof int[][] arr2d) {
            int rows = arr2d.length;
            int cols = arr2d[0].length;
            MemoryBuffer buf = MemoryBuffer.allocate(DType.INT32, (long) rows * cols);
            for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) buf.setInt((long) i * cols + j, arr2d[i][j]);
            return new NDArray(buf, new int[] {rows, cols}, DType.INT32, 'C');
        }
        if (data instanceof double[][] arr2d) {
            int rows = arr2d.length;
            int cols = arr2d[0].length;
            MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, (long) rows * cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++) buf.setDouble((long) i * cols + j, arr2d[i][j]);
            return new NDArray(buf, new int[] {rows, cols}, DType.FLOAT64, 'C');
        }
        throw new IllegalArgumentException("Unsupported array type: " + data.getClass());
    }

    public static NDArray zeros(int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray zeros(DType dtype, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(dtype, size);
        return new NDArray(buf, shape, dtype, 'C');
    }

    public static NDArray ones(int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) buf.setDouble(i, 1.0);
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray ones(DType dtype, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(dtype, size);
        for (long i = 0; i < size; i++) buf.setDouble(i, 1.0);
        return new NDArray(buf, shape, dtype, 'C');
    }

    public static NDArray empty(int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray full(int[] shape, double fillValue, DType dtype) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(dtype, size);
        for (long i = 0; i < size; i++) buf.setDouble(i, fillValue);
        return new NDArray(buf, shape, dtype, 'C');
    }

    public static NDArray eye(int n) {
        return eye(n, n);
    }

    public static NDArray eye(int rows, int cols) {
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, (long) rows * cols);
        for (int i = 0; i < Math.min(rows, cols); i++) buf.setDouble((long) i * cols + i, 1.0);
        return new NDArray(buf, new int[] {rows, cols}, DType.FLOAT64, 'C');
    }

    public static NDArray identity(int n) {
        return eye(n, n);
    }

    public static NDArray diag(double[] diagonal) {
        int n = diagonal.length;
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, (long) n * n);
        for (int i = 0; i < n; i++) buf.setDouble((long) i * n + i, diagonal[i]);
        return new NDArray(buf, new int[] {n, n}, DType.FLOAT64, 'C');
    }

    public static NDArray diag(NDArray v) {
        if (v.ndim() == 1) {
            int n = (int) v.size();
            return diag(jnumpy.util.Util.toDoubleArray(v));
        }
        return jnumpy.manipulation.Manipulation.diagFlat(v);
    }

    public static NDArray diagFlat(NDArray a) {
        if (a.ndim() != 2) throw new IllegalArgumentException("diagflat requires 2D array");
        int n = Math.min(a.shape(0), a.shape(1));
        double[] diag = new double[n];
        for (int i = 0; i < n; i++) diag[i] = a.getDouble(i, i);
        return new NDArray(MemoryBuffer.wrap(diag), DType.FLOAT64);
    }

    public static NDArray arange(double start, double stop, double step) {
        long size = (long) Math.ceil((stop - start) / step);
        if (size < 0) size = 0;
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) buf.setDouble(i, start + i * step);
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray arange(double stop) {
        return arange(0, stop, 1);
    }

    public static NDArray arange(double start, double stop) {
        return arange(start, stop, 1);
    }

    public static NDArray linspace(double start, double stop, int num) {
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, num);
        if (num == 1) {
            buf.setDouble(0, start);
        } else {
            double step = (stop - start) / (num - 1);
            for (int i = 0; i < num; i++) buf.setDouble(i, start + i * step);
        }
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray logspace(double start, double stop, int num) {
        NDArray linear = linspace(start, stop, num);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, num);
        for (long i = 0; i < num; i++)
            buf.setDouble(i, Math.pow(10, linear.buffer().getDouble(i)));
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray geomspace(double start, double stop, int num) {
        double logStart = Math.log(start);
        double logStop = Math.log(stop);
        return linspace(logStart, logStop, num).exp();
    }

    public static NDArray[] meshgrid(NDArray x, NDArray y) {
        int nx = (int) x.size();
        int ny = (int) y.size();
        MemoryBuffer xBuf = MemoryBuffer.allocate(DType.FLOAT64, (long) nx * ny);
        MemoryBuffer yBuf = MemoryBuffer.allocate(DType.FLOAT64, (long) nx * ny);
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                xBuf.setDouble((long) i * nx + j, x.getDouble(j));
                yBuf.setDouble((long) i * nx + j, y.getDouble(i));
            }
        }
        NDArray xResult = new NDArray(xBuf, new int[] {ny, nx}, DType.FLOAT64, 'C');
        NDArray yResult = new NDArray(yBuf, new int[] {ny, nx}, DType.FLOAT64, 'C');
        return new NDArray[] {xResult, yResult};
    }

    public static NDArray fromFunction(IntToDoubleFunction f, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) {
            buf.setDouble(i, f.applyAsDouble((int) i));
        }
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray copy(NDArray a) {
        return a.copy();
    }

    public static NDArray fromBuffer(byte[] data, DType dtype) {
        int elemSize = dtype.byteSize();
        if (elemSize <= 0) throw new IllegalArgumentException("fromBuffer does not support variable-size dtypes");
        int numElements = data.length / elemSize;
        MemoryBuffer buf = MemoryBuffer.allocate(dtype, numElements);
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.nativeOrder());
        for (int i = 0; i < numElements; i++) {
            switch (dtype) {
                case DType.Int8Type ignored -> buf.setByte(i, bb.get());
                case DType.Int16Type ignored -> buf.setShort(i, bb.getShort());
                case DType.Int32Type ignored -> buf.setInt(i, bb.getInt());
                case DType.Int64Type ignored -> buf.setLong(i, bb.getLong());
                case DType.Float32Type ignored -> buf.setFloat(i, bb.getFloat());
                case DType.Float64Type ignored -> buf.setDouble(i, bb.getDouble());
                case DType.BoolType ignored -> buf.setBool(i, bb.get() != 0);
                case DType.UInt8Type ignored -> buf.setByte(i, bb.get());
                case DType.UInt16Type ignored -> buf.setShort(i, bb.getShort());
                case DType.UInt32Type ignored -> buf.setInt(i, bb.getInt());
                case DType.UInt64Type ignored -> buf.setLong(i, bb.getLong());
                default -> buf.setDouble(i, bb.getDouble());
            }
        }
        return new NDArray(buf, dtype);
    }
}
