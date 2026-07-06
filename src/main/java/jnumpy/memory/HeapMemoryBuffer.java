package jnumpy.memory;

import jnumpy.dtype.DType;

final class HeapMemoryBuffer {

    record ByteBuffer(byte[] data) implements MemoryBuffer {
        ByteBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.INT8; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new ByteBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new ByteBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.INT8) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setByte(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return data[(int) index]; }
        @Override public short getShort(long index) { return data[(int) index]; }
        @Override public int getInt(long index) { return data[(int) index]; }
        @Override public long getLong(long index) { return data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = (byte) value; }
        @Override public void setInt(long index, int value) { data[(int) index] = (byte) value; }
        @Override public void setLong(long index, long value) { data[(int) index] = (byte) value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = (byte) value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (byte) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = (byte) (value ? 1 : 0); }
        @Override public void setChar(long index, char value) { data[(int) index] = (byte) value; }
        @Override public void setString(long index, String value) { data[(int) index] = Byte.parseByte(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Byte b ? b : 0; }
        @Override public long elementSize() { return 1; }
    }

    record ShortBuffer(short[] data) implements MemoryBuffer {
        ShortBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.INT16; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new ShortBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new ShortBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.INT16) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setShort(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return data[(int) index]; }
        @Override public int getInt(long index) { return data[(int) index]; }
        @Override public long getLong(long index) { return data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = (short) value; }
        @Override public void setLong(long index, long value) { data[(int) index] = (short) value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = (short) value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (short) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = (short) (value ? 1 : 0); }
        @Override public void setChar(long index, char value) { data[(int) index] = (short) value; }
        @Override public void setString(long index, String value) { data[(int) index] = Short.parseShort(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Short s ? s : 0; }
        @Override public long elementSize() { return 2; }
    }

    record IntBuffer(int[] data) implements MemoryBuffer {
        IntBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.INT32; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new IntBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new IntBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.INT32) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setInt(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return (short) data[(int) index]; }
        @Override public int getInt(long index) { return data[(int) index]; }
        @Override public long getLong(long index) { return data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = value; }
        @Override public void setLong(long index, long value) { data[(int) index] = (int) value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = (int) value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (int) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value ? 1 : 0; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = Integer.parseInt(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Integer i ? i : 0; }
        @Override public long elementSize() { return 4; }
    }

    record LongBuffer(long[] data) implements MemoryBuffer {
        LongBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.INT64; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new LongBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new LongBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.INT64) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setLong(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return (short) data[(int) index]; }
        @Override public int getInt(long index) { return (int) data[(int) index]; }
        @Override public long getLong(long index) { return data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = value; }
        @Override public void setLong(long index, long value) { data[(int) index] = value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = (long) value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (long) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value ? 1L : 0L; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = Long.parseLong(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Long l ? l : 0L; }
        @Override public long elementSize() { return 8; }
    }

    record FloatBuffer(float[] data) implements MemoryBuffer {
        FloatBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.FLOAT32; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new FloatBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new FloatBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.FLOAT32) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setFloat(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return (short) data[(int) index]; }
        @Override public int getInt(long index) { return (int) data[(int) index]; }
        @Override public long getLong(long index) { return (long) data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = value; }
        @Override public void setLong(long index, long value) { data[(int) index] = value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (float) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value ? 1f : 0f; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = Float.parseFloat(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Float f ? f : 0f; }
        @Override public long elementSize() { return 4; }
    }

    record DoubleBuffer(double[] data) implements MemoryBuffer {
        DoubleBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.FLOAT64; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new DoubleBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new DoubleBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.FLOAT64) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setDouble(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return (short) data[(int) index]; }
        @Override public int getInt(long index) { return (int) data[(int) index]; }
        @Override public long getLong(long index) { return (long) data[(int) index]; }
        @Override public float getFloat(long index) { return (float) data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return (char) data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = value; }
        @Override public void setLong(long index, long value) { data[(int) index] = value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value ? 1.0 : 0.0; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = Double.parseDouble(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Double d ? d : 0.0; }
        @Override public long elementSize() { return 8; }
    }

    record BoolBuffer(boolean[] data) implements MemoryBuffer {
        BoolBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.BOOL; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new BoolBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new BoolBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.BOOL) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setBool(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) (data[(int) index] ? 1 : 0); }
        @Override public short getShort(long index) { return (short) (data[(int) index] ? 1 : 0); }
        @Override public int getInt(long index) { return data[(int) index] ? 1 : 0; }
        @Override public long getLong(long index) { return data[(int) index] ? 1L : 0L; }
        @Override public float getFloat(long index) { return data[(int) index] ? 1f : 0f; }
        @Override public double getDouble(long index) { return data[(int) index] ? 1.0 : 0.0; }
        @Override public boolean getBool(long index) { return data[(int) index]; }
        @Override public char getChar(long index) { return data[(int) index] ? (char) 1 : (char) 0; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value != 0; }
        @Override public void setShort(long index, short value) { data[(int) index] = value != 0; }
        @Override public void setInt(long index, int value) { data[(int) index] = value != 0; }
        @Override public void setLong(long index, long value) { data[(int) index] = value != 0; }
        @Override public void setFloat(long index, float value) { data[(int) index] = value != 0; }
        @Override public void setDouble(long index, double value) { data[(int) index] = value != 0; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value; }
        @Override public void setChar(long index, char value) { data[(int) index] = value != 0; }
        @Override public void setString(long index, String value) { data[(int) index] = Boolean.parseBoolean(value); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Boolean b && b; }
        @Override public long elementSize() { return 1; }
    }

    record CharBuffer(char[] data) implements MemoryBuffer {
        CharBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.CHAR; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new CharBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new CharBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.CHAR) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setChar(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return (byte) data[(int) index]; }
        @Override public short getShort(long index) { return (short) data[(int) index]; }
        @Override public int getInt(long index) { return data[(int) index]; }
        @Override public long getLong(long index) { return data[(int) index]; }
        @Override public float getFloat(long index) { return data[(int) index]; }
        @Override public double getDouble(long index) { return data[(int) index]; }
        @Override public boolean getBool(long index) { return data[(int) index] != 0; }
        @Override public char getChar(long index) { return data[(int) index]; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = (char) value; }
        @Override public void setShort(long index, short value) { data[(int) index] = (char) value; }
        @Override public void setInt(long index, int value) { data[(int) index] = (char) value; }
        @Override public void setLong(long index, long value) { data[(int) index] = (char) value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = (char) value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = (char) value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value ? (char) 1 : (char) 0; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = value.isEmpty() ? '\u0000' : value.charAt(0); }
        @Override public void setObject(long index, Object value) { data[(int) index] = value instanceof Character c ? c : '\u0000'; }
        @Override public long elementSize() { return 2; }
    }

    record ObjectBuffer(Object[] data) implements MemoryBuffer {
        ObjectBuffer { if (data == null) throw new IllegalArgumentException("data cannot be null"); }
        @Override public DType dtype() { return DType.OBJECT; }
        @Override public long size() { return data.length; }
        @Override public boolean isHeap() { return true; }
        @Override public boolean isOffHeap() { return false; }
        @Override public boolean isContiguous() { return true; }
        @Override public MemoryBuffer slice(long offset, long length) {
            return new ObjectBuffer(java.util.Arrays.copyOfRange(data, (int) offset, (int) (offset + length)));
        }
        @Override public MemoryBuffer duplicate() { return new ObjectBuffer(data.clone()); }
        @Override public MemoryBuffer copy(DType dtype) {
            if (dtype == DType.OBJECT) return duplicate();
            MemoryBuffer result = MemoryBuffer.allocate(dtype, size());
            for (int i = 0; i < data.length; i++) result.setObject(i, data[i]);
            return result;
        }
        @Override public byte getByte(long index) { return data[(int) index] instanceof Number n ? n.byteValue() : 0; }
        @Override public short getShort(long index) { return data[(int) index] instanceof Number n ? n.shortValue() : 0; }
        @Override public int getInt(long index) { return data[(int) index] instanceof Number n ? n.intValue() : 0; }
        @Override public long getLong(long index) { return data[(int) index] instanceof Number n ? n.longValue() : 0L; }
        @Override public float getFloat(long index) { return data[(int) index] instanceof Number n ? n.floatValue() : 0f; }
        @Override public double getDouble(long index) { return data[(int) index] instanceof Number n ? n.doubleValue() : 0.0; }
        @Override public boolean getBool(long index) { return Boolean.TRUE.equals(data[(int) index]); }
        @Override public char getChar(long index) { return data[(int) index] instanceof Character c ? c : '\u0000'; }
        @Override public String getString(long index) { return String.valueOf(data[(int) index]); }
        @Override public Object getObject(long index) { return data[(int) index]; }
        @Override public void setByte(long index, byte value) { data[(int) index] = value; }
        @Override public void setShort(long index, short value) { data[(int) index] = value; }
        @Override public void setInt(long index, int value) { data[(int) index] = value; }
        @Override public void setLong(long index, long value) { data[(int) index] = value; }
        @Override public void setFloat(long index, float value) { data[(int) index] = value; }
        @Override public void setDouble(long index, double value) { data[(int) index] = value; }
        @Override public void setBool(long index, boolean value) { data[(int) index] = value; }
        @Override public void setChar(long index, char value) { data[(int) index] = value; }
        @Override public void setString(long index, String value) { data[(int) index] = value; }
        @Override public void setObject(long index, Object value) { data[(int) index] = value; }
        @Override public long elementSize() { return -1; }
    }
}
