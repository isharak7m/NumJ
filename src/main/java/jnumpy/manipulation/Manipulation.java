package jnumpy.manipulation;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Manipulation {

    private Manipulation() {}

    public static NDArray concatenate(NDArray[] arrays, int axis) {
        if (arrays.length == 0) throw new IllegalArgumentException("No arrays to concatenate");
        if (arrays.length == 1) return arrays[0];
        int[] shape = arrays[0].shape();
        int totalSize = 0;
        for (NDArray arr : arrays) {
            if (arr.ndim() != shape.length) throw new IllegalArgumentException("All arrays must have same ndim");
            for (int i = 0; i < shape.length; i++) {
                if (i != axis && arr.shape(i) != shape[i])
                    throw new IllegalArgumentException("Shapes must match except on concat axis");
            }
            totalSize += arr.shape(axis);
        }
        int[] newShape = shape.clone();
        newShape[axis] = totalSize;
        NDArray result = NDArray.create(newShape, arrays[0].dtype());
        int offset = 0;
        int[] indices = new int[newShape.length];
        for (NDArray arr : arrays) {
            int dimSize = arr.shape(axis);
            for (int i = 0; i < dimSize; i++) {
                for (int j = 0; j < arr.size() / dimSize; j++) {
                    long remaining = j;
                    for (int d = newShape.length - 1; d >= 0; d--) {
                        if (d == axis) continue;
                        indices[d] = (int) (remaining % shape[d]);
                        remaining /= shape[d];
                    }
                    indices[axis] = offset + i;
                    int[] srcIdx = indices.clone();
                    srcIdx[axis] = i;
                    result.setDouble(arr.getDouble(srcIdx), indices);
                }
            }
            offset += dimSize;
        }
        return result;
    }

    public static NDArray stack(NDArray[] arrays, int axis) {
        int ndim = arrays[0].ndim();
        int[] newShape = new int[ndim + 1];
        int pos = axis < 0 ? axis + ndim + 1 : axis;
        for (int i = 0, j = 0; i < newShape.length; i++) {
            if (i == pos) newShape[i] = arrays.length;
            else newShape[i] = arrays[0].shape(j++);
        }
        NDArray result = NDArray.create(newShape, arrays[0].dtype());
        int[] indices = new int[newShape.length];
        for (int k = 0; k < arrays.length; k++) {
            for (long i = 0; i < arrays[k].size(); i++) {
                long remaining = i;
                for (int d = ndim - 1; d >= 0; d--) {
                    int targetDim = d >= pos ? d + 1 : d;
                    indices[targetDim] = (int) (remaining % arrays[k].shape(d));
                    remaining /= arrays[k].shape(d);
                }
                indices[pos] = k;
                result.setDouble(arrays[k].getDouble(indicesAtDim(arrays[k], i)), indices);
            }
        }
        return result;
    }

    public static NDArray hstack(NDArray[] arrays) {
        return concatenate(arrays, 1);
    }

    public static NDArray vstack(NDArray[] arrays) {
        return concatenate(arrays, 0);
    }

    public static NDArray columnStack(NDArray[] arrays) {
        NDArray[] reshaped = new NDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            reshaped[i] = arrays[i].reshape(arrays[i].shape(0), 1);
        }
        return concatenate(reshaped, 1);
    }

    public static NDArray split(NDArray a, int sections, int axis) {
        int dimSize = a.shape(axis);
        int sectionSize = dimSize / sections;
        int remainder = dimSize % sections;
        NDArray[] results = new NDArray[sections];
        int start = 0;
        for (int i = 0; i < sections; i++) {
            int size = sectionSize + (i < remainder ? 1 : 0);
            int[] newShape = a.shape().clone();
            newShape[axis] = size;
            results[i] = jnumpy.indexing.Indexer.slice(a, axis, start, start + size, 1);
            start += size;
        }
        return results[0];
    }

    public static NDArray diagFlat(NDArray a) {
        int n = Math.min(a.shape(0), a.shape(1));
        double[] diag = new double[n];
        for (int i = 0; i < n; i++) diag[i] = a.getDouble(i, i);
        return new NDArray(MemoryBuffer.wrap(diag), DType.FLOAT64);
    }

    private static int[] indicesAtDim(NDArray a, long flat) {
        int[] idx = new int[a.ndim()];
        long remaining = flat;
        for (int d = a.ndim() - 1; d >= 0; d--) {
            idx[d] = (int) (remaining % a.shape()[d]);
            remaining /= a.shape()[d];
        }
        return idx;
    }
}
