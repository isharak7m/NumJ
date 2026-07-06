package jnumpy.sort;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.util.Arrays;
import java.util.Comparator;

public final class Sort {

    private Sort() {}

    public static NDArray sort(NDArray a, int... axis) {
        if (axis.length == 0) {
            long n = a.size();
            double[] data = new double[(int) n];
            int[] idx = new int[a.ndim()];
            for (long i = 0; i < n; i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    idx[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                data[(int) i] = a.getDouble(idx);
            }
            Arrays.sort(data);
            NDArray result = NDArray.create(a.shape(), DType.FLOAT64);
            for (long i = 0; i < n; i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    idx[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                result.setDouble(data[(int) i], idx);
            }
            return result;
        }
        int ax = axis[0];
        int[] shape = a.shape();
        int outerSize = 1;
        for (int i = 0; i < ax; i++) outerSize *= shape[i];
        int innerSize = 1;
        for (int i = ax + 1; i < shape.length; i++) innerSize *= shape[i];
        int dimSize = shape[ax];
        NDArray result = a.copy();
        int[] indices = new int[a.ndim()];
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                double[] slice = new double[dimSize];
                int outerRemaining = outer;
                for (int i = ax - 1; i >= 0; i--) {
                    indices[i] = outerRemaining % shape[i];
                    outerRemaining /= shape[i];
                }
                int innerRemaining = inner;
                for (int i = shape.length - 1; i > ax; i--) {
                    indices[i] = innerRemaining % shape[i];
                    innerRemaining /= shape[i];
                }
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    slice[d] = result.getDouble(indices);
                }
                Arrays.sort(slice);
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    result.setDouble(slice[d], indices);
                }
            }
        }
        return result;
    }

    public static NDArray argsort(NDArray a, int axis) {
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] shape = a.shape();
        int outerSize = 1;
        for (int i = 0; i < ax; i++) outerSize *= shape[i];
        int innerSize = 1;
        for (int i = ax + 1; i < shape.length; i++) innerSize *= shape[i];
        int dimSize = shape[ax];
        int[] resultShape = shape.clone();
        NDArray result = NDArray.create(resultShape, DType.INT64);
        int[] indices = new int[a.ndim()];
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                double[] slice = new double[dimSize];
                int outerRemaining = outer;
                for (int i = ax - 1; i >= 0; i--) {
                    indices[i] = outerRemaining % shape[i];
                    outerRemaining /= shape[i];
                }
                int innerRemaining = inner;
                for (int i = shape.length - 1; i > ax; i--) {
                    indices[i] = innerRemaining % shape[i];
                    innerRemaining /= shape[i];
                }
                Integer[] order = new Integer[dimSize];
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    slice[d] = a.getDouble(indices);
                    order[d] = d;
                }
                Arrays.sort(order, Comparator.comparingDouble(i -> slice[i]));
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    result.setLong(order[d], indices);
                }
            }
        }
        return result;
    }

    public static NDArray unique(NDArray a) {
        long n = a.size();
        double[] data = new double[(int) n];
        int[] idx = new int[a.ndim()];
        for (long i = 0; i < n; i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                idx[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            data[(int) i] = a.getDouble(idx);
        }
        Arrays.sort(data);
        int uniqueCount = 1;
        for (int i = 1; i < n; i++) if (data[i] != data[i - 1]) uniqueCount++;
        double[] uniqueData = new double[uniqueCount];
        uniqueData[0] = data[0];
        int j = 1;
        for (int i = 1; i < n; i++) {
            if (data[i] != data[i - 1]) uniqueData[j++] = data[i];
        }
        MemoryBuffer buf = MemoryBuffer.wrap(uniqueData);
        return new NDArray(buf, new int[]{ uniqueCount }, DType.FLOAT64, 'C');
    }

    public static NDArray searchsorted(NDArray a, double v) {
        long n = a.size();
        int lo = 0, hi = (int) n;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a.getDouble(mid) < v) lo = mid + 1;
            else hi = mid;
        }
        return NDArray.create(new double[]{ lo });
    }

    public static NDArray partition(NDArray a, int kth, int axis) {
        NDArray result = a.copy();
        int ax = axis < 0 ? axis + a.ndim() : axis;
        int[] shape = a.shape();
        int dimSize = shape[ax];
        int outerSize = 1;
        for (int i = 0; i < ax; i++) outerSize *= shape[i];
        int innerSize = 1;
        for (int i = ax + 1; i < shape.length; i++) innerSize *= shape[i];
        int[] indices = new int[a.ndim()];
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                double[] slice = new double[dimSize];
                int outerRemaining = outer;
                for (int i = ax - 1; i >= 0; i--) {
                    indices[i] = outerRemaining % shape[i];
                    outerRemaining /= shape[i];
                }
                int innerRemaining = inner;
                for (int i = shape.length - 1; i > ax; i--) {
                    indices[i] = innerRemaining % shape[i];
                    innerRemaining /= shape[i];
                }
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    slice[d] = result.getDouble(indices);
                }
                int k = kth < 0 ? kth + dimSize : kth;
                int nth = quickSelect(slice, k, 0, dimSize - 1);
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    result.setDouble(slice[d], indices);
                }
            }
        }
        return result;
    }

    private static int quickSelect(double[] arr, int k, int left, int right) {
        while (left < right) {
            int pivotIdx = partition(arr, left, right);
            if (k == pivotIdx) return pivotIdx;
            else if (k < pivotIdx) right = pivotIdx - 1;
            else left = pivotIdx + 1;
        }
        return left;
    }

    private static int partition(double[] arr, int left, int right) {
        double pivot = arr[right];
        int i = left;
        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                double tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
                i++;
            }
        }
        double tmp = arr[i]; arr[i] = arr[right]; arr[right] = tmp;
        return i;
    }
}
