package jnumpy.broadcast;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.util.Arrays;

public final class Broadcast {

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
                    throw new IllegalArgumentException(
                            "Cannot broadcast shapes: " + Arrays.deepToString(shapes));
                }
                result[resIdx] = Math.max(result[resIdx], dim);
            }
        }
        return result;
    }

    public static NDArray broadcastTo(NDArray array, int[] targetShape) {
        int[] arrayShape = array.shape();
        int targetNdim = targetShape.length;
        int sourceNdim = arrayShape.length;
        int diff = targetNdim - sourceNdim;
        int[] paddedShape = new int[targetNdim];
        long[] paddedStrides = new long[targetNdim];
        for (int i = 0; i < diff; i++) {
            paddedShape[i] = 1;
            paddedStrides[i] = 0;
        }
        long[] srcStrides = array.strides();
        for (int i = 0; i < sourceNdim; i++) {
            paddedShape[diff + i] = arrayShape[i];
            paddedStrides[diff + i] = srcStrides[i];
        }
        long[] newStrides = new long[targetNdim];
        for (int i = 0; i < targetNdim; i++) {
            if (paddedShape[i] == 1 && targetShape[i] > 1) {
                newStrides[i] = 0;
            } else if (paddedShape[i] != targetShape[i]) {
                throw new IllegalArgumentException(
                        "Cannot broadcast shape " + Arrays.toString(arrayShape) + " to " + Arrays.toString(targetShape));
            } else {
                newStrides[i] = paddedStrides[i];
            }
        }
        return new NDArray(targetShape, newStrides, array.dtype(), array.offset(), array.buffer(), true);
    }

    public static BroadcastIterator iterator(NDArray a, NDArray b) {
        int[] shape = broadcastShape(a.shape(), b.shape());
        return new BroadcastIterator(a, b, shape);
    }

    public static final class BroadcastIterator {
        private final NDArray a;
        private final NDArray b;
        private final int[] shape;
        private final long[] aStrides;
        private final long[] bStrides;
        private final int ndim;
        private final long size;
        private final long[] aPadStrides;
        private final long[] bPadStrides;
        private final int[] aPadShape;
        private final int[] bPadShape;

        public BroadcastIterator(NDArray a, NDArray b, int[] resultShape) {
            this.a = a;
            this.b = b;
            this.shape = resultShape;
            this.ndim = resultShape.length;
            long s = 1;
            for (int d : resultShape) s *= d;
            this.size = s;
            int[] aShape = a.shape();
            int[] bShape = b.shape();
            long[] aStr = a.strides();
            long[] bStr = b.strides();
            int aDiff = ndim - aShape.length;
            int bDiff = ndim - bShape.length;
            aPadShape = new int[ndim];
            bPadShape = new int[ndim];
            aPadStrides = new long[ndim];
            bPadStrides = new long[ndim];
            for (int i = 0; i < ndim; i++) {
                aPadShape[i] = (i < aDiff) ? 1 : aShape[i - aDiff];
                aPadStrides[i] = (i < aDiff) ? 0 : aStr[i - aDiff];
                bPadShape[i] = (i < bDiff) ? 1 : bShape[i - bDiff];
                bPadStrides[i] = (i < bDiff) ? 0 : bStr[i - bDiff];
            }
            this.aStrides = aPadStrides;
            this.bStrides = bPadStrides;
        }

        public int[] shape() { return shape.clone(); }
        public long size() { return size; }
        public int ndim() { return ndim; }

        public long aIndex(long flat) {
            long idx = a.offset();
            long remaining = flat;
            for (int i = ndim - 1; i >= 0; i--) {
                int coord = (int) (remaining % shape[i]);
                remaining /= shape[i];
                if (aPadShape[i] > 1) idx += (long) coord * aPadStrides[i];
            }
            return idx;
        }

        public long bIndex(long flat) {
            long idx = b.offset();
            long remaining = flat;
            for (int i = ndim - 1; i >= 0; i--) {
                int coord = (int) (remaining % shape[i]);
                remaining /= shape[i];
                if (bPadShape[i] > 1) idx += (long) coord * bPadStrides[i];
            }
            return idx;
        }

        public void forEach(BroadcastConsumer consumer) {
            int[] coords = new int[ndim];
            for (long flat = 0; flat < size; flat++) {
                long remaining = flat;
                for (int i = ndim - 1; i >= 0; i--) {
                    coords[i] = (int) (remaining % shape[i]);
                    remaining /= shape[i];
                }
                long aOff = a.offset();
                long bOff = b.offset();
                for (int i = 0; i < ndim; i++) {
                    if (aPadShape[i] > 1) aOff += (long) coords[i] * aPadStrides[i];
                    if (bPadShape[i] > 1) bOff += (long) coords[i] * bPadStrides[i];
                }
                consumer.accept(flat, aOff, bOff);
            }
        }
    }

    @FunctionalInterface
    public interface BroadcastConsumer {
        void accept(long flatIndex, long aOffset, long bOffset);
    }
}
