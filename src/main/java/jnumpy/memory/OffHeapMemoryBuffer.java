package jnumpy.memory;

import jnumpy.dtype.DType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

final class OffHeapMemoryBuffer implements MemoryBuffer {

    private final ByteBuffer buffer;
    private final DType dtype;
    private final long size;

    OffHeapMemoryBuffer(ByteBuffer buffer, DType dtype) {
        this.buffer = buffer.duplicate().order(ByteOrder.nativeOrder());
        this.dtype = dtype;
        this.size = buffer.remaining() / dtype.byteSize();
    }

    @Override
    public DType dtype() { return dtype; }

    @Override
    public long size() { return size; }

    @Override
    public boolean isHeap() { return false; }

    @Override
    public boolean isOffHeap() { return true; }

    @Override
    public boolean isContiguous() { return true; }

    @Override
    public MemoryBuffer slice(long offset, long length) {
        int pos = (int) (offset * elementSize());
        int limit = (int) (pos + length * elementSize());
        ByteBuffer slice = buffer.duplicate().order(ByteOrder.nativeOrder());
        slice.position(pos);
        slice.limit(limit);
        return new OffHeapMemoryBuffer(slice, dtype);
    }

    @Override
    public MemoryBuffer duplicate() {
        ByteBuffer dup = ByteBuffer.allocateDirect(buffer.capacity()).order(ByteOrder.nativeOrder());
        dup.put(buffer.duplicate().order(ByteOrder.nativeOrder()));
        dup.flip();
        return new OffHeapMemoryBuffer(dup, dtype);
    }

    @Override
    public MemoryBuffer copy(DType dtype) {
        if (dtype == this.dtype) return duplicate();
        MemoryBuffer result = MemoryBuffer.allocate(dtype, size);
        for (long i = 0; i < size; i++) {
            long byteOffset = i * elementSize();
            switch (dtype) {
                case DType.BoolType d -> result.setBool(i, buffer.get((int) byteOffset) != 0);
                case DType.Int8Type d -> result.setByte(i, buffer.get((int) byteOffset));
                case DType.Int16Type d -> result.setShort(i, buffer.getShort((int) byteOffset));
                case DType.Int32Type d -> result.setInt(i, buffer.getInt((int) byteOffset));
                case DType.Int64Type d -> result.setLong(i, buffer.getLong((int) byteOffset));
                case DType.Float32Type d -> result.setFloat(i, buffer.getFloat((int) byteOffset));
                case DType.Float64Type d -> result.setDouble(i, buffer.getDouble((int) byteOffset));
                default -> result.setObject(i, null);
            }
        }
        return result;
    }

    @Override
    public long elementSize() { return dtype.byteSize(); }

    @Override
    public byte getByte(long index) { return buffer.get((int) (index * elementSize())); }
    @Override
    public short getShort(long index) { return buffer.getShort((int) (index * elementSize())); }
    @Override
    public int getInt(long index) { return buffer.getInt((int) (index * elementSize())); }
    @Override
    public long getLong(long index) { return buffer.getLong((int) (index * elementSize())); }
    @Override
    public float getFloat(long index) { return buffer.getFloat((int) (index * elementSize())); }
    @Override
    public double getDouble(long index) { return buffer.getDouble((int) (index * elementSize())); }
    @Override
    public boolean getBool(long index) { return buffer.get((int) (index * elementSize())) != 0; }
    @Override
    public char getChar(long index) { return buffer.getChar((int) (index * elementSize())); }
    @Override
    public String getString(long index) { return String.valueOf(getObject(index)); }
    @Override
    public Object getObject(long index) { return null; }

    @Override
    public void setByte(long index, byte value) { buffer.put((int) (index * elementSize()), value); }
    @Override
    public void setShort(long index, short value) { buffer.putShort((int) (index * elementSize()), value); }
    @Override
    public void setInt(long index, int value) { buffer.putInt((int) (index * elementSize()), value); }
    @Override
    public void setLong(long index, long value) { buffer.putLong((int) (index * elementSize()), value); }
    @Override
    public void setFloat(long index, float value) { buffer.putFloat((int) (index * elementSize()), value); }
    @Override
    public void setDouble(long index, double value) { buffer.putDouble((int) (index * elementSize()), value); }
    @Override
    public void setBool(long index, boolean value) { buffer.put((int) (index * elementSize()), (byte) (value ? 1 : 0)); }
    @Override
    public void setChar(long index, char value) { buffer.putChar((int) (index * elementSize()), value); }
    @Override
    public void setString(long index, String value) { }
    @Override
    public void setObject(long index, Object value) { }
}
