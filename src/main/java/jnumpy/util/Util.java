package jnumpy.util;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Util {

    private Util() {}

    public static double readBuffer(MemoryBuffer buf, DType dtype, long offset) {
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

    public static void writeBuffer(MemoryBuffer buf, DType dtype, long offset, double val) {
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

    public static double readElement(NDArray a, int... idx) {
        return readBuffer(a.buffer(), a.dtype(), a.flatIndex(idx));
    }

    public static void writeElement(NDArray a, double val, int... idx) {
        writeBuffer(a.buffer(), a.dtype(), a.flatIndex(idx), val);
    }

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
