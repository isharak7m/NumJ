package jnumpy.ndarray;

import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.broadcast.Broadcast;
import jnumpy.indexing.Indexer;
import jnumpy.ufunc.UFunc;
import java.util.Arrays;
import java.util.Objects;

public final class NDArray {

    private final int[] shape;
    private final long[] strides;
    private final DType dtype;
    private final long offset;
    private final MemoryBuffer buffer;
    private final boolean isView;
    private final int ndim;
    private final long size;

    public NDArray(int[] shape, long[] strides, DType dtype, long offset, MemoryBuffer buffer, boolean isView) {
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.dtype = Objects.requireNonNull(dtype);
        this.offset = offset;
        this.buffer = Objects.requireNonNull(buffer);
        this.isView = isView;
        this.ndim = shape.length;
        long s = 1;
        for (int d : shape) s *= d;
        this.size = s;
    }

    public NDArray(MemoryBuffer buffer, DType dtype) {
        this(new int[]{ (int) buffer.size() }, computeStrides(new int[]{ (int) buffer.size() }, 'C'),
                dtype, 0, buffer, false);
    }

    public NDArray(MemoryBuffer buffer, int[] shape, DType dtype, char order) {
        this(shape, computeStrides(shape, order), dtype, 0, buffer, false);
    }

    public int[] shape() { return shape.clone(); }
    public long[] strides() { return strides.clone(); }
    public DType dtype() { return dtype; }
    public long offset() { return offset; }
    public MemoryBuffer buffer() { return buffer; }
    public boolean isView() { return isView; }
    public int ndim() { return ndim; }
    public long size() { return size; }

    public int shape(int axis) {
        if (axis < 0) axis += ndim;
        return shape[axis];
    }

    public long stride(int axis) {
        if (axis < 0) axis += ndim;
        return strides[axis];
    }

    public boolean isScalar() { return ndim == 0; }
    public boolean isVector() { return ndim == 1; }
    public boolean isMatrix() { return ndim == 2; }
    public boolean isEmpty() { return size == 0; }

    public boolean isContiguous() {
        return isContiguous('C') || isContiguous('F');
    }

    public boolean isContiguous(char order) {
        if (order == 'C') {
            long expectedStride = 1;
            for (int i = ndim - 1; i >= 0; i--) {
                if (strides[i] != expectedStride) return false;
                expectedStride *= shape[i];
            }
            return true;
        } else {
            long expectedStride = 1;
            for (int i = 0; i < ndim; i++) {
                if (strides[i] != expectedStride) return false;
                expectedStride *= shape[i];
            }
            return true;
        }
    }

    public long flatIndex(int... indices) {
        if (indices.length != ndim) {
            throw new IllegalArgumentException("Expected " + ndim + " indices, got " + indices.length);
        }
        long idx = offset;
        for (int i = 0; i < ndim; i++) {
            int ix = indices[i];
            if (ix < 0) ix += shape[i];
            if (ix < 0 || ix >= shape[i]) {
                throw new IndexOutOfBoundsException(
                        "Index " + indices[i] + " out of bounds for axis " + i + " with size " + shape[i]);
            }
            idx += ix * strides[i];
        }
        return idx;
    }

    public byte getByte(int... indices) { return buffer.getByte(flatIndex(indices)); }
    public short getShort(int... indices) { return buffer.getShort(flatIndex(indices)); }
    public int getInt(int... indices) { return buffer.getInt(flatIndex(indices)); }
    public long getLong(int... indices) { return buffer.getLong(flatIndex(indices)); }
    public float getFloat(int... indices) { return buffer.getFloat(flatIndex(indices)); }
    public double getDouble(int... indices) { return buffer.getDouble(flatIndex(indices)); }
    public boolean getBoolean(int... indices) { return buffer.getBool(flatIndex(indices)); }
    public char getChar(int... indices) { return buffer.getChar(flatIndex(indices)); }
    public String getString(int... indices) { return buffer.getString(flatIndex(indices)); }
    public Object getObject(int... indices) { return buffer.getObject(flatIndex(indices)); }

    public void setByte(byte value, int... indices) { buffer.setByte(flatIndex(indices), value); }
    public void setShort(short value, int... indices) { buffer.setShort(flatIndex(indices), value); }
    public void setInt(int value, int... indices) { buffer.setInt(flatIndex(indices), value); }
    public void setLong(long value, int... indices) { buffer.setLong(flatIndex(indices), value); }
    public void setFloat(float value, int... indices) { buffer.setFloat(flatIndex(indices), value); }
    public void setDouble(double value, int... indices) { buffer.setDouble(flatIndex(indices), value); }
    public void setBoolean(boolean value, int... indices) { buffer.setBool(flatIndex(indices), value); }
    public void setChar(char value, int... indices) { buffer.setChar(flatIndex(indices), value); }
    public void setString(String value, int... indices) { buffer.setString(flatIndex(indices), value); }
    public void setObject(Object value, int... indices) { buffer.setObject(flatIndex(indices), value); }

    public NDArray copy() {
        MemoryBuffer newBuf = MemoryBuffer.allocate(dtype, size);
        copyData(this, new NDArray(newBuf, shape, dtype, 'C'));
        return new NDArray(newBuf, shape, dtype, 'C');
    }

    public NDArray deepCopy() {
        return copy();
    }

    public NDArray view() {
        return new NDArray(shape, strides, dtype, offset, buffer, true);
    }

    public NDArray reshape(int... newShape) {
        long newSize = 1;
        int autoDim = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (autoDim != -1) throw new IllegalArgumentException("Only one dimension can be -1");
                autoDim = i;
            } else {
                newSize *= newShape[i];
            }
        }
        if (autoDim >= 0) {
            newShape = newShape.clone();
            newShape[autoDim] = (int) (size / newSize);
            newSize = size;
        }
        if (newSize != size) {
            throw new IllegalArgumentException("Cannot reshape array of size " + size + " into shape " + Arrays.toString(newShape));
        }
        if (isContiguous()) {
            return new NDArray(newShape, computeStrides(newShape, 'C'), dtype, offset, buffer, true);
        }
        NDArray contig = ravel();
        return new NDArray(contig.buffer, newShape, dtype, 'C');
    }

    public NDArray ravel() {
        if (isContiguous()) {
            return new NDArray(new int[]{ (int) size }, new long[]{ 1 }, dtype, offset, buffer, true);
        }
        NDArray flat = copy();
        return new NDArray(flat.buffer, new int[]{ (int) size }, dtype, 'C');
    }

    public NDArray flatten() {
        return copy().ravel();
    }

    public NDArray transpose(int... axes) {
        int n = ndim;
        if (axes.length == 0) {
            axes = new int[n];
            for (int i = 0; i < n; i++) axes[i] = n - 1 - i;
        }
        if (axes.length != n) {
            throw new IllegalArgumentException("Axes length " + axes.length + " does not match ndim " + n);
        }
        int[] newShape = new int[n];
        long[] newStrides = new long[n];
        for (int i = 0; i < n; i++) {
            newShape[i] = shape[axes[i]];
            newStrides[i] = strides[axes[i]];
        }
        return new NDArray(newShape, newStrides, dtype, offset, buffer, true);
    }

    public NDArray T() {
        if (ndim <= 1) return this;
        return transpose();
    }

    public NDArray swapAxes(int axis1, int axis2) {
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) axes[i] = i;
        axes[axis1] = axis2;
        axes[axis2] = axis1;
        return transpose(axes);
    }

    public NDArray moveAxis(int source, int destination) {
        int[] axes = new int[ndim];
        for (int i = 0; i < ndim; i++) axes[i] = i;
        if (source < destination) {
            for (int i = source; i < destination; i++) axes[i] = i + 1;
            axes[destination] = source;
        } else {
            for (int i = source; i > destination; i--) axes[i] = i - 1;
            axes[destination] = source;
        }
        return transpose(axes);
    }

    public NDArray squeeze(int... axes) {
        if (axes.length == 0) {
            int squeezed = 0;
            for (int s : shape) if (s == 1) squeezed++;
            int[] newShape = new int[ndim - squeezed];
            long[] newStrides = new long[ndim - squeezed];
            int j = 0;
            for (int i = 0; i < ndim; i++) {
                if (shape[i] != 1) {
                    newShape[j] = shape[i];
                    newStrides[j] = strides[i];
                    j++;
                }
            }
            return new NDArray(newShape, newStrides, dtype, offset, buffer, true);
        }
        int[] newShape = shape.clone();
        long[] newStrides = strides.clone();
        for (int axis : axes) {
            if (shape[axis] != 1) {
                throw new IllegalArgumentException("Cannot squeeze axis " + axis + " with size " + shape[axis]);
            }
        }
        int[] result = new int[ndim - axes.length];
        long[] resultStrides = new long[ndim - axes.length];
        int j = 0;
        for (int i = 0; i < ndim; i++) {
            boolean skip = false;
            for (int axis : axes) { if (i == axis) { skip = true; break; } }
            if (!skip) {
                result[j] = shape[i];
                resultStrides[j] = strides[i];
                j++;
            }
        }
        return new NDArray(result, resultStrides, dtype, offset, buffer, true);
    }

    public NDArray expandDims(int axis) {
        int newNdim = ndim + 1;
        int[] newShape = new int[newNdim];
        long[] newStrides = new long[newNdim];
        int pos = axis < 0 ? axis + newNdim : axis;
        for (int i = 0, j = 0; i < newNdim; i++) {
            if (i == pos) {
                newShape[i] = 1;
                newStrides[i] = 0;
            } else {
                newShape[i] = shape[j];
                newStrides[i] = strides[j];
                j++;
            }
        }
        return new NDArray(newShape, newStrides, dtype, offset, buffer, true);
    }

    public NDArray repeat(int repeats, int axis) {
        int[] newShape = shape.clone();
        newShape[axis] *= repeats;
        NDArray result = new NDArray(MemoryBuffer.allocate(dtype, sizeOfShape(newShape)), newShape, dtype, 'C');
        int[] srcIdx = new int[ndim];
        int[] dstIdx = new int[ndim];
        long[] srcPos = new long[ndim];
        for (long flat = 0; flat < result.size; flat++) {
            result.indices(flat, dstIdx);
            for (int r = 0; r < ndim; r++) srcIdx[r] = dstIdx[r] % shape[r];
            srcPos[0] = offset;
            for (int r = 0; r < ndim; r++) srcPos[0] += (long) srcIdx[r] * strides[r];
            switch (dtype) {
                case DType.Float64Type ignored -> result.buffer.setDouble(flat, buffer.getDouble(srcPos[0]));
                case DType.Float32Type ignored -> result.buffer.setFloat(flat, buffer.getFloat(srcPos[0]));
                case DType.Int32Type ignored -> result.buffer.setInt(flat, buffer.getInt(srcPos[0]));
                case DType.Int64Type ignored -> result.buffer.setLong(flat, buffer.getLong(srcPos[0]));
                case DType.BoolType ignored -> result.buffer.setBool(flat, buffer.getBool(srcPos[0]));
                default -> result.buffer.setDouble(flat, buffer.getDouble(srcPos[0]));
            }
        }
        return result;
    }

    public NDArray tile(int... reps) {
        int[] newShape = new int[ndim];
        for (int i = 0; i < ndim; i++) newShape[i] = shape[i] * reps[i];
        NDArray result = new NDArray(MemoryBuffer.allocate(dtype, sizeOfShape(newShape)), newShape, dtype, 'C');
        int[] srcIdx = new int[ndim];
        int[] dstIdx = new int[ndim];
        for (long flat = 0; flat < result.size; flat++) {
            result.indices(flat, dstIdx);
            for (int r = 0; r < ndim; r++) srcIdx[r] = dstIdx[r] % shape[r];
            result.buffer.setDouble(flat, getDouble(srcIdx));
        }
        return result;
    }

    public NDArray flip(int... axes) {
        if (axes.length == 0) {
            axes = new int[ndim];
            for (int i = 0; i < ndim; i++) axes[i] = i;
        }
        long[] newStrides = strides.clone();
        long[] newOffsets = new long[axes.length];
        long baseOffset = offset;
        for (int a = 0; a < axes.length; a++) {
            int axis = axes[a] < 0 ? axes[a] + ndim : axes[a];
            newStrides[axis] = -strides[axis];
            baseOffset += (shape[axis] - 1) * strides[axis];
        }
        return new NDArray(shape, newStrides, dtype, baseOffset, buffer, true);
    }

    public NDArray rot90(int k, int... axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("rot90 requires exactly 2 axes");
        }
        k = ((k % 4) + 4) % 4;
        NDArray result = this;
        for (int i = 0; i < k; i++) {
            result = result.flip(axes[0]).swapAxes(axes[0], axes[1]);
        }
        return result;
    }

    public NDArray broadcastTo(int[] targetShape) {
        return Broadcast.broadcastTo(this, targetShape);
    }

    public NDArray add(NDArray other) { return UFunc.add(this, other); }
    public NDArray sub(NDArray other) { return UFunc.sub(this, other); }
    public NDArray mul(NDArray other) { return UFunc.mul(this, other); }
    public NDArray div(NDArray other) { return UFunc.div(this, other); }
    public NDArray pow(NDArray other) { return UFunc.pow(this, other); }
    public NDArray neg() { return UFunc.neg(this); }
    public NDArray abs() { return UFunc.abs(this); }
    public NDArray sqrt() { return UFunc.sqrt(this); }
    public NDArray sin() { return UFunc.sin(this); }
    public NDArray cos() { return UFunc.cos(this); }
    public NDArray tan() { return UFunc.tan(this); }
    public NDArray exp() { return UFunc.exp(this); }
    public NDArray log() { return UFunc.log(this); }

    public NDArray sum(int... axis) { return jnumpy.statistics.Statistics.sum(this, axis); }
    public NDArray mean(int... axis) { return jnumpy.statistics.Statistics.mean(this, axis); }
    public NDArray var(int... axis) { return jnumpy.statistics.Statistics.var(this, axis); }
    public NDArray std(int... axis) { return jnumpy.statistics.Statistics.std(this, axis); }
    public NDArray prod(int... axis) { return jnumpy.statistics.Statistics.prod(this, axis); }
    public NDArray cumsum(int... axis) { return jnumpy.statistics.Statistics.cumsum(this, axis); }
    public NDArray cumprod(int... axis) { return jnumpy.statistics.Statistics.cumprod(this, axis); }
    public NDArray min(int... axis) { return jnumpy.statistics.Statistics.min(this, axis); }
    public NDArray max(int... axis) { return jnumpy.statistics.Statistics.max(this, axis); }
    public NDArray argmin(int... axis) { return jnumpy.statistics.Statistics.argmin(this, axis); }
    public NDArray argmax(int... axis) { return jnumpy.statistics.Statistics.argmax(this, axis); }

    public NDArray matmul(NDArray other) { return jnumpy.linalg.Linalg.matmul(this, other); }
    public NDArray dot(NDArray other) { return jnumpy.linalg.Linalg.dot(this, other); }
    public double norm() { return jnumpy.linalg.Linalg.norm(this); }
    public NDArray inverse() { return jnumpy.linalg.Linalg.inv(this); }
    public double determinant() { return jnumpy.linalg.Linalg.det(this); }

    public NDArray get(int... indices) {
        return Indexer.get(this, indices);
    }

    public NDArray set(NDArray value, int... indices) {
        return Indexer.set(this, value, indices);
    }

    public NDArray slice(int dim, int start, int end, int step) {
        return Indexer.slice(this, dim, start, end, step);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        toString(sb, 0, new int[ndim]);
        sb.append(", dtype=").append(dtype.name());
        return sb.toString();
    }

    private void toString(StringBuilder sb, int dim, int[] indices) {
        if (dim == ndim) {
            long idx = offset;
            for (int i = 0; i < ndim; i++) idx += (long) indices[i] * strides[i];
            sb.append(formatElement(idx));
            return;
        }
        sb.append("[");
        for (int i = 0; i < shape[dim]; i++) {
            if (i > 0) sb.append(dim == ndim - 1 ? ", " : ",\n");
            indices[dim] = i;
            toString(sb, dim + 1, indices);
        }
        sb.append("]");
    }

    private String formatElement(long idx) {
        return switch (dtype) {
            case DType.Float64Type ignored -> String.format("%.6f", buffer.getDouble(idx));
            case DType.Float32Type ignored -> String.format("%.6f", buffer.getFloat(idx));
            case DType.Int32Type ignored -> String.valueOf(buffer.getInt(idx));
            case DType.Int64Type ignored -> String.valueOf(buffer.getLong(idx));
            case DType.BoolType ignored -> String.valueOf(buffer.getBool(idx));
            default -> String.valueOf(buffer.getDouble(idx));
        };
    }

    public void indices(long flatIndex, int[] out) {
        long remaining = flatIndex;
        for (int i = ndim - 1; i >= 0; i--) {
            out[i] = (int) (remaining % shape[i]);
            remaining /= shape[i];
        }
    }

    private static void copyData(NDArray src, NDArray dst) {
        long size = src.size;
        int ndim = src.ndim;
        int[] shape = src.shape;
        int[] indices = new int[ndim];
        for (long i = 0; i < size; i++) {
            long remaining = i;
            for (int d = ndim - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            long srcIdx = src.offset;
            for (int d = 0; d < ndim; d++) srcIdx += (long) indices[d] * src.strides[d];
            switch (src.dtype) {
                case DType.Float64Type ignored -> dst.buffer.setDouble(i, src.buffer.getDouble(srcIdx));
                case DType.Float32Type ignored -> dst.buffer.setFloat(i, src.buffer.getFloat(srcIdx));
                case DType.Int32Type ignored -> dst.buffer.setInt(i, src.buffer.getInt(srcIdx));
                case DType.Int64Type ignored -> dst.buffer.setLong(i, src.buffer.getLong(srcIdx));
                case DType.BoolType ignored -> dst.buffer.setBool(i, src.buffer.getBool(srcIdx));
                default -> dst.buffer.setDouble(i, src.buffer.getDouble(srcIdx));
            }
        }
    }

    public static long[] computeStrides(int[] shape, char order) {
        int n = shape.length;
        long[] strides = new long[n];
        if (order == 'C') {
            strides[n - 1] = 1;
            for (int i = n - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
        } else {
            strides[0] = 1;
            for (int i = 1; i < n; i++) strides[i] = strides[i - 1] * shape[i - 1];
        }
        return strides;
    }

    public static NDArray create(int[] shape, DType dtype) {
        long size = sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(dtype, size);
        return new NDArray(buf, shape, dtype, 'C');
    }

    public static NDArray create(double[] data) {
        MemoryBuffer buf = MemoryBuffer.wrap(data);
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray create(int[] data) {
        MemoryBuffer buf = MemoryBuffer.wrap(data);
        return new NDArray(buf, DType.INT32);
    }

    public static NDArray create(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, (long) rows * cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                buf.setDouble((long) i * cols + j, data[i][j]);
        return new NDArray(buf, new int[]{ rows, cols }, DType.FLOAT64, 'C');
    }

    public static long sizeOfShape(int[] shape) {
        long s = 1;
        for (int d : shape) s *= d;
        return s;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NDArray other)) return false;
        if (size != other.size || ndim != other.ndim) return false;
        if (!Arrays.equals(shape, other.shape)) return false;
        NDArray flatThis = this.ravel();
        NDArray flatOther = other.ravel();
        for (long i = 0; i < size; i++) {
            if (Double.compare(flatThis.getDouble(new int[]{ (int) i }),
                    flatOther.getDouble(new int[]{ (int) i })) != 0) return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        NDArray flat = ravel();
        for (long i = 0; i < Math.min(size, 10); i++) {
            long v = Double.doubleToLongBits(flat.getDouble(new int[]{ (int) i }));
            result = 31 * result + (int) (v ^ (v >>> 32));
        }
        return result;
    }
}
