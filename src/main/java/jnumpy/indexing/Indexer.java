package jnumpy.indexing;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public final class Indexer {

    private Indexer() {}

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
                array.dtype(), offset, array.buffer(), true);
    }

    public static NDArray set(NDArray array, NDArray value, int... indices) {
        NDArray view = get(array, indices);
        int[] vShape = value.shape();
        int[] targetShape = view.shape();
        int[] bShape = Broadcast.broadcastShape(vShape, targetShape);
        NDArray broadcastValue = jnumpy.broadcast.Broadcast.broadcastTo(value, bShape);
        NDArray broadcastTarget = jnumpy.broadcast.Broadcast.broadcastTo(view, bShape);
        jnumpy.broadcast.Broadcast.BroadcastIterator it =
                new jnumpy.broadcast.Broadcast.BroadcastIterator(broadcastValue, broadcastTarget, bShape);
        it.forEach((flat, aOff, bOff) -> {
            array.buffer().setDouble(bOff, value.buffer().getDouble(aOff));
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
                    double val = array.getDouble(srcIdx);
                    result.setDouble(val, dstIdx);
                }
            }
        }
        return result;
    }

    public static NDArray booleanIndex(NDArray array, NDArray mask) {
        int count = 0;
        for (long i = 0; i < mask.size(); i++) {
            if (mask.getBoolean(new int[]{ (int) i })) count++;
        }
        int[] resultShape;
        if (array.ndim() == 1) {
            resultShape = new int[]{ count };
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
            if (mask.getBoolean(new int[]{ (int) i })) {
                array.indices(i, srcIdx);
                dstIdx[0] = dst++;
                for (int d = 1; d < array.ndim(); d++) dstIdx[d] = srcIdx[d];
                result.setDouble(array.getDouble(srcIdx), dstIdx);
            }
        }
        return result;
    }

    public static class Broadcast {
        private Broadcast() {}
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
