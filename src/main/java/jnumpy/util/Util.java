package jnumpy.util;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Util {

    private Util() {}

    public static double[] toDoubleArray(NDArray a) {
        int n = (int) a.size();
        double[] result = new double[n];
        int[] indices = new int[a.ndim()];
        for (int i = 0; i < n; i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result[i] = a.getDouble(indices);
        }
        return result;
    }

    public static NDArray asContiguous(NDArray a) {
        if (a.isContiguous()) return a;
        return a.copy();
    }

    public static boolean allClose(NDArray a, NDArray b, double rtol, double atol) {
        if (!java.util.Arrays.equals(a.shape(), b.shape())) return false;
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            double va = a.getDouble(indices);
            double vb = b.getDouble(indices);
            if (Double.isNaN(va) && Double.isNaN(vb)) continue;
            if (Double.isInfinite(va) && Double.isInfinite(vb) && va == vb) continue;
            double diff = Math.abs(va - vb);
            double tolerance = atol + rtol * Math.max(Math.abs(va), Math.abs(vb));
            if (diff > tolerance) return false;
        }
        return true;
    }

    public static NDArray reshapeWithCopy(NDArray a, int... newShape) {
        return a.reshape(newShape).copy();
    }
}
