package jnumpy.memory;

import jnumpy.dtype.DType;
import java.nio.ByteBuffer;

public sealed interface MemoryBuffer permits HeapMemoryBuffer.ByteBuffer, HeapMemoryBuffer.ShortBuffer,
        HeapMemoryBuffer.IntBuffer, HeapMemoryBuffer.LongBuffer, HeapMemoryBuffer.FloatBuffer,
        HeapMemoryBuffer.DoubleBuffer, HeapMemoryBuffer.BoolBuffer, HeapMemoryBuffer.CharBuffer,
        HeapMemoryBuffer.ObjectBuffer, OffHeapMemoryBuffer {

    DType dtype();
    long size();
    boolean isHeap();
    boolean isOffHeap();
    boolean isContiguous();
    MemoryBuffer slice(long offset, long length);
    MemoryBuffer duplicate();
    MemoryBuffer copy(DType dtype);

    byte getByte(long index);
    short getShort(long index);
    int getInt(long index);
    long getLong(long index);
    float getFloat(long index);
    double getDouble(long index);
    boolean getBool(long index);
    char getChar(long index);
    String getString(long index);
    Object getObject(long index);

    void setByte(long index, byte value);
    void setShort(long index, short value);
    void setInt(long index, int value);
    void setLong(long index, long value);
    void setFloat(long index, float value);
    void setDouble(long index, double value);
    void setBool(long index, boolean value);
    void setChar(long index, char value);
    void setString(long index, String value);
    void setObject(long index, Object value);

    long elementSize();

    static MemoryBuffer allocate(DType dtype, long size) {
        return switch (dtype) {
            case DType.BoolType b -> new HeapMemoryBuffer.BoolBuffer(new boolean[(int) size]);
            case DType.Int8Type b -> new HeapMemoryBuffer.ByteBuffer(new byte[(int) size]);
            case DType.Int16Type b -> new HeapMemoryBuffer.ShortBuffer(new short[(int) size]);
            case DType.Int32Type b -> new HeapMemoryBuffer.IntBuffer(new int[(int) size]);
            case DType.Int64Type b -> new HeapMemoryBuffer.LongBuffer(new long[(int) size]);
            case DType.UInt8Type b -> new HeapMemoryBuffer.ByteBuffer(new byte[(int) size]);
            case DType.UInt16Type b -> new HeapMemoryBuffer.ShortBuffer(new short[(int) size]);
            case DType.UInt32Type b -> new HeapMemoryBuffer.IntBuffer(new int[(int) size]);
            case DType.UInt64Type b -> new HeapMemoryBuffer.LongBuffer(new long[(int) size]);
            case DType.Float16Type b -> new HeapMemoryBuffer.ShortBuffer(new short[(int) size]);
            case DType.Float32Type b -> new HeapMemoryBuffer.FloatBuffer(new float[(int) size]);
            case DType.Float64Type b -> new HeapMemoryBuffer.DoubleBuffer(new double[(int) size]);
            case DType.Complex64Type b -> new HeapMemoryBuffer.FloatBuffer(new float[(int) size * 2]);
            case DType.Complex128Type b -> new HeapMemoryBuffer.DoubleBuffer(new double[(int) size * 2]);
            case DType.CharType b -> new HeapMemoryBuffer.CharBuffer(new char[(int) size]);
            case DType.StringType b -> new HeapMemoryBuffer.ObjectBuffer(new String[(int) size]);
            case DType.ObjectType b -> new HeapMemoryBuffer.ObjectBuffer(new Object[(int) size]);
            default -> throw new IllegalArgumentException("Unknown dtype: " + dtype);
        };
    }

    static MemoryBuffer wrap(Object array) {
        return switch (array) {
            case boolean[] a -> new HeapMemoryBuffer.BoolBuffer(a);
            case byte[] a -> new HeapMemoryBuffer.ByteBuffer(a);
            case short[] a -> new HeapMemoryBuffer.ShortBuffer(a);
            case int[] a -> new HeapMemoryBuffer.IntBuffer(a);
            case long[] a -> new HeapMemoryBuffer.LongBuffer(a);
            case float[] a -> new HeapMemoryBuffer.FloatBuffer(a);
            case double[] a -> new HeapMemoryBuffer.DoubleBuffer(a);
            case char[] a -> new HeapMemoryBuffer.CharBuffer(a);
            case String[] a -> new HeapMemoryBuffer.ObjectBuffer(a);
            case Object[] a -> new HeapMemoryBuffer.ObjectBuffer(a);
            default -> throw new IllegalArgumentException("Cannot wrap: " + array.getClass());
        };
    }

    static MemoryBuffer wrapOffHeap(ByteBuffer buffer, DType dtype) {
        return new OffHeapMemoryBuffer(buffer, dtype);
    }
}
