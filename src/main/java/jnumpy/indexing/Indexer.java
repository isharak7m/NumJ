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

package jnumpy.indexing;

import java.util.ArrayList;
import java.util.List;
import jnumpy.broadcast.Broadcast;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.ndarray.NDArray;

public final class Indexer {

    private Indexer() {}

    private static double readElement(NDArray a, int[] idx) {
        MemoryBuffer buf = a.buffer();
        long offset = a.flatIndex(idx);
        return switch (a.dtype()) {
            case DType.BoolType ignored -> buf.getBool(offset) ? 1.0 : 0.0;
            case DType.Int8Type ignored -> (double) buf.getByte(offset);
            case DType.Int16Type ignored -> (double) buf.getShort(offset);
            case DType.Int32Type ignored -> (double) buf.getInt(offset);
            case DType.Int64Type ignored -> (double) buf.getLong(offset);
            case DType.UInt8Type ignored -> (double) (buf.getByte(offset) & 0xFF);
            case DType.UInt16Type ignored -> (double) (buf.getShort(offset) & 0xFFFF);
            case DType.UInt32Type ignored -> (double) (buf.getInt(offset) & 0xFFFFFFFFL);
            case DType.UInt64Type ignored -> (double) buf.getLong(offset);
            case DType.Float16Type ignored -> (double) buf.getShort(offset);
            case DType.Float32Type ignored -> (double) buf.getFloat(offset);
            case DType.Float64Type ignored -> buf.getDouble(offset);
            default -> buf.getDouble(offset);
        };
    }

    private static void writeElement(NDArray a, double val, int[] idx) {
        MemoryBuffer buf = a.buffer();
        long offset = a.flatIndex(idx);
        switch (a.dtype()) {
            case DType.BoolType ignored -> buf.setBool(offset, val != 0.0);
            case DType.Int8Type ignored -> buf.setByte(offset, (byte) ((int) val));
            case DType.Int16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.Int32Type ignored -> buf.setInt(offset, (int) val);
            case DType.Int64Type ignored -> buf.setLong(offset, (long) val);
            case DType.UInt8Type ignored -> buf.setByte(offset, (byte) ((int) val));
            case DType.UInt16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.UInt32Type ignored -> buf.setInt(offset, (int) val);
            case DType.UInt64Type ignored -> buf.setLong(offset, (long) val);
            case DType.Float16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.Float32Type ignored -> buf.setFloat(offset, (float) val);
            case DType.Float64Type ignored -> buf.setDouble(offset, val);
            default -> buf.setDouble(offset, val);
        }
    }

    private static double readBuffer(MemoryBuffer buf, DType dtype, long offset) {
        return switch (dtype) {
            case DType.BoolType ignored -> buf.getBool(offset) ? 1.0 : 0.0;
            case DType.Int8Type ignored -> (double) buf.getByte(offset);
            case DType.Int16Type ignored -> (double) buf.getShort(offset);
            case DType.Int32Type ignored -> (double) buf.getInt(offset);
            case DType.Int64Type ignored -> (double) buf.getLong(offset);
            case DType.UInt8Type ignored -> (double) (buf.getByte(offset) & 0xFF);
            case DType.UInt16Type ignored -> (double) (buf.getShort(offset) & 0xFFFF);
            case DType.UInt32Type ignored -> (double) (buf.getInt(offset) & 0xFFFFFFFFL);
            case DType.UInt64Type ignored -> (double) buf.getLong(offset);
            case DType.Float16Type ignored -> (double) buf.getShort(offset);
            case DType.Float32Type ignored -> (double) buf.getFloat(offset);
            case DType.Float64Type ignored -> buf.getDouble(offset);
            default -> buf.getDouble(offset);
        };
    }

    private static void writeBuffer(MemoryBuffer buf, DType dtype, long offset, double val) {
        switch (dtype) {
            case DType.BoolType ignored -> buf.setBool(offset, val != 0.0);
            case DType.Int8Type ignored -> buf.setByte(offset, (byte) ((int) val));
            case DType.Int16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.Int32Type ignored -> buf.setInt(offset, (int) val);
            case DType.Int64Type ignored -> buf.setLong(offset, (long) val);
            case DType.UInt8Type ignored -> buf.setByte(offset, (byte) ((int) val));
            case DType.UInt16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.UInt32Type ignored -> buf.setInt(offset, (int) val);
            case DType.UInt64Type ignored -> buf.setLong(offset, (long) val);
            case DType.Float16Type ignored -> buf.setShort(offset, (short) ((int) val));
            case DType.Float32Type ignored -> buf.setFloat(offset, (float) val);
            case DType.Float64Type ignored -> buf.setDouble(offset, val);
            default -> buf.setDouble(offset, val);
        }
    }

    public static NDArray get(NDArray array, int... indices) {
        if (indices.length > array.ndim()) {
            throw new IllegalArgumentException("Too many indices for array");
        }
        int[] shape = array.shape();
        long[] strides = array.strides();
        long offset = array.offset();
        int dimsConsumed = 0;
        List<Integer> resultShape = new ArrayList<>();
        List<Long> resultStrides = new ArrayList<>();
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx < 0) idx += shape[i];
            if (idx < 0 || idx >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " out of bounds for axis " + i + " with size " + shape[i]);
            }
            offset += (long) idx * strides[i];
            dimsConsumed++;
        }
        for (int i = dimsConsumed; i < array.ndim(); i++) {
            resultShape.add(shape[i]);
            resultStrides.add(strides[i]);
        }
        if (resultShape.isEmpty()) {
            resultShape.add(1);
            resultStrides.add(0L);
        }
        return new NDArray(
                resultShape.stream().mapToInt(Integer::intValue).toArray(),
                resultStrides.stream().mapToLong(Long::longValue).toArray(),
                array.dtype(),
                offset,
                array.buffer(),
                true);
    }

    public static NDArray set(NDArray array, NDArray value, int... indices) {
        NDArray view = get(array, indices);
        int[] vShape = value.shape();
        int[] targetShape = view.shape();
        int[] bShape = Broadcast.broadcastShape(vShape, targetShape);
        NDArray broadcastValue = Broadcast.broadcastTo(value, bShape);
        NDArray broadcastTarget = Broadcast.broadcastTo(view, bShape);
        Broadcast.BroadcastIterator it = new Broadcast.BroadcastIterator(broadcastValue, broadcastTarget, bShape);
        it.forEach((flat, aOff, bOff) -> {
            double val = readBuffer(value.buffer(), value.dtype(), aOff);
            writeBuffer(array.buffer(), array.dtype(), bOff, val);
        });
        return array;
    }

    public static NDArray slice(NDArray array, int dim, int start, int end, int step) {
        if (step == 0) throw new IllegalArgumentException("Step cannot be 0");
        int size = array.shape(dim);
        if (start < 0) start += size;
        if (end < 0) end += size;
        if (start < 0) start = 0;
        if (end > size) end = size;
        if (start >= end) {
            int[] newShape = array.shape().clone();
            newShape[dim] = 0;
            return new NDArray(MemoryBuffer.allocate(array.dtype(), 0), newShape, array.dtype(), 'C');
        }
        int newSize = (end - start + step - 1) / step;
        int[] newShape = array.shape().clone();
        newShape[dim] = newSize;
        long[] newStrides = array.strides().clone();
        newStrides[dim] *= step;
        long newOffset = array.offset() + (long) start * array.stride(dim);
        return new NDArray(newShape, newStrides, array.dtype(), newOffset, array.buffer(), true);
    }

    public static NDArray sliceArray(NDArray array, int dim, int[] indices) {
        int[] newShape = array.shape().clone();
        newShape[dim] = indices.length;
        long[] newStrides = array.strides().clone();
        NDArray result = NDArray.create(newShape, array.dtype());
        int[] srcIdx = new int[array.ndim()];
        int[] dstIdx = new int[array.ndim()];
        for (int i = 0; i < array.ndim(); i++) {
            if (i != dim) dstIdx[i] = 0;
        }
        for (int j = 0; j < indices.length; j++) {
            dstIdx[dim] = j;
            srcIdx[dim] = indices[j] < 0 ? indices[j] + array.shape(dim) : indices[j];
            for (int i = 0; i < array.ndim(); i++) {
                if (i == dim) continue;
                for (int k = 0; k < array.shape(i); k++) {
                    srcIdx[i] = k;
                    dstIdx[i] = k;
                    double val = readElement(array, srcIdx);
                    writeElement(result, val, dstIdx);
                }
            }
        }
        return result;
    }

    public static NDArray booleanIndex(NDArray array, NDArray mask) {
        int count = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.getBoolean(new int[] {(int) i})) count++;
        }
        int[] resultShape;
        if (array.ndim() == 1) {
            resultShape = new int[] {count};
        } else {
            resultShape = new int[array.ndim()];
            resultShape[0] = count;
            for (int i = 1; i < array.ndim(); i++) resultShape[i] = array.shape(i);
        }
        NDArray result = NDArray.create(resultShape, array.dtype());
        int[] srcIdx = new int[array.ndim()];
        int[] dstIdx = new int[array.ndim()];
        int dst = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.getBoolean(new int[] {(int) i})) {
                array.indices(i, srcIdx);
                dstIdx[0] = dst++;
                for (int d = 1; d < array.ndim(); d++) dstIdx[d] = srcIdx[d];
                double val = readElement(array, srcIdx);
                writeElement(result, val, dstIdx);
            }
        }
        return result;
    }

    public static class ShapeHelper {
        private ShapeHelper() {}

        public static int[] broadcastShape(int[]... shapes) {
            int maxDim = 0;
            for (int[] s : shapes) maxDim = Math.max(maxDim, s.length);
            int[] result = new int[maxDim];
            for (int i = 0; i < maxDim; i++) result[i] = 1;
            for (int[] shape : shapes) {
                int diff = maxDim - shape.length;
                for (int i = 0; i < shape.length; i++) {
                    int dim = shape[shape.length - 1 - i];
                    int resIdx = maxDim - 1 - i;
                    if (dim != 1 && result[resIdx] != 1 && result[resIdx] != dim) {
                        throw new IllegalArgumentException("Cannot broadcast shapes");
                    }
                    result[resIdx] = Math.max(result[resIdx], dim);
                }
            }
            return result;
        }
    }
}
