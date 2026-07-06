package jnumpy.dtype;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public sealed interface DType permits DType.BoolType, DType.Int8Type, DType.Int16Type, DType.Int32Type, DType.Int64Type,
        DType.UInt8Type, DType.UInt16Type, DType.UInt32Type, DType.UInt64Type, DType.Float16Type, DType.Float32Type,
        DType.Float64Type, DType.Complex64Type, DType.Complex128Type, DType.CharType, DType.StringType,
        DType.ObjectType {

    enum Kind {
        BOOL,
        SIGNED_INT,
        UNSIGNED_INT,
        FLOAT,
        COMPLEX,
        CHAR,
        STRING,
        OBJECT
    }

    String name();
    Kind kind();
    int byteSize();
    boolean isInteger();
    boolean isFloat();
    boolean isComplex();
    boolean isSigned();
    boolean isUnsigned();
    boolean isNumeric();
    boolean isBuiltin();
    Class<?> javaType();
    Object defaultValue();
    DType promotedWith(DType other);

    record BoolType() implements DType {
        @Override public String name() { return "bool"; }
        @Override public Kind kind() { return Kind.BOOL; }
        @Override public int byteSize() { return 1; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return false; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return boolean.class; }
        @Override public Object defaultValue() { return false; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Int8Type() implements DType {
        @Override public String name() { return "int8"; }
        @Override public Kind kind() { return Kind.SIGNED_INT; }
        @Override public int byteSize() { return 1; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return byte.class; }
        @Override public Object defaultValue() { return (byte)0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Int16Type() implements DType {
        @Override public String name() { return "int16"; }
        @Override public Kind kind() { return Kind.SIGNED_INT; }
        @Override public int byteSize() { return 2; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return short.class; }
        @Override public Object defaultValue() { return (short)0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Int32Type() implements DType {
        @Override public String name() { return "int32"; }
        @Override public Kind kind() { return Kind.SIGNED_INT; }
        @Override public int byteSize() { return 4; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return int.class; }
        @Override public Object defaultValue() { return 0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Int64Type() implements DType {
        @Override public String name() { return "int64"; }
        @Override public Kind kind() { return Kind.SIGNED_INT; }
        @Override public int byteSize() { return 8; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return long.class; }
        @Override public Object defaultValue() { return 0L; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record UInt8Type() implements DType {
        @Override public String name() { return "uint8"; }
        @Override public Kind kind() { return Kind.UNSIGNED_INT; }
        @Override public int byteSize() { return 1; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return true; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return byte.class; }
        @Override public Object defaultValue() { return (byte)0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record UInt16Type() implements DType {
        @Override public String name() { return "uint16"; }
        @Override public Kind kind() { return Kind.UNSIGNED_INT; }
        @Override public int byteSize() { return 2; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return true; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return short.class; }
        @Override public Object defaultValue() { return (short)0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record UInt32Type() implements DType {
        @Override public String name() { return "uint32"; }
        @Override public Kind kind() { return Kind.UNSIGNED_INT; }
        @Override public int byteSize() { return 4; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return true; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return int.class; }
        @Override public Object defaultValue() { return 0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record UInt64Type() implements DType {
        @Override public String name() { return "uint64"; }
        @Override public Kind kind() { return Kind.UNSIGNED_INT; }
        @Override public int byteSize() { return 8; }
        @Override public boolean isInteger() { return true; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return true; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return long.class; }
        @Override public Object defaultValue() { return 0L; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Float16Type() implements DType {
        @Override public String name() { return "float16"; }
        @Override public Kind kind() { return Kind.FLOAT; }
        @Override public int byteSize() { return 2; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return true; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return short.class; }
        @Override public Object defaultValue() { return (short)0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Float32Type() implements DType {
        @Override public String name() { return "float32"; }
        @Override public Kind kind() { return Kind.FLOAT; }
        @Override public int byteSize() { return 4; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return true; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return float.class; }
        @Override public Object defaultValue() { return 0.0f; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Float64Type() implements DType {
        @Override public String name() { return "float64"; }
        @Override public Kind kind() { return Kind.FLOAT; }
        @Override public int byteSize() { return 8; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return true; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return double.class; }
        @Override public Object defaultValue() { return 0.0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Complex64Type() implements DType {
        @Override public String name() { return "complex64"; }
        @Override public Kind kind() { return Kind.COMPLEX; }
        @Override public int byteSize() { return 8; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return true; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return float.class; }
        @Override public Object defaultValue() { return 0.0f; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record Complex128Type() implements DType {
        @Override public String name() { return "complex128"; }
        @Override public Kind kind() { return Kind.COMPLEX; }
        @Override public int byteSize() { return 16; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return true; }
        @Override public boolean isSigned() { return true; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return true; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return double.class; }
        @Override public Object defaultValue() { return 0.0; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record CharType() implements DType {
        @Override public String name() { return "char"; }
        @Override public Kind kind() { return Kind.CHAR; }
        @Override public int byteSize() { return 2; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return false; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return char.class; }
        @Override public Object defaultValue() { return '\u0000'; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record StringType() implements DType {
        @Override public String name() { return "string"; }
        @Override public Kind kind() { return Kind.STRING; }
        @Override public int byteSize() { return -1; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return false; }
        @Override public boolean isBuiltin() { return true; }
        @Override public Class<?> javaType() { return String.class; }
        @Override public Object defaultValue() { return ""; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    record ObjectType() implements DType {
        @Override public String name() { return "object"; }
        @Override public Kind kind() { return Kind.OBJECT; }
        @Override public int byteSize() { return -1; }
        @Override public boolean isInteger() { return false; }
        @Override public boolean isFloat() { return false; }
        @Override public boolean isComplex() { return false; }
        @Override public boolean isSigned() { return false; }
        @Override public boolean isUnsigned() { return false; }
        @Override public boolean isNumeric() { return false; }
        @Override public boolean isBuiltin() { return false; }
        @Override public Class<?> javaType() { return Object.class; }
        @Override public Object defaultValue() { return null; }
        @Override public DType promotedWith(DType other) { return DType.promotionTable.get(this, other); }
    }

    DType BOOL = new BoolType();
    DType INT8 = new Int8Type();
    DType INT16 = new Int16Type();
    DType INT32 = new Int32Type();
    DType INT64 = new Int64Type();
    DType UINT8 = new UInt8Type();
    DType UINT16 = new UInt16Type();
    DType UINT32 = new UInt32Type();
    DType UINT64 = new UInt64Type();
    DType FLOAT16 = new Float16Type();
    DType FLOAT32 = new Float32Type();
    DType FLOAT64 = new Float64Type();
    DType COMPLEX64 = new Complex64Type();
    DType COMPLEX128 = new Complex128Type();
    DType CHAR = new CharType();
    DType STRING = new StringType();
    DType OBJECT = new ObjectType();

    static DType fromClass(Class<?> clazz) {
        if (clazz == boolean.class || clazz == Boolean.class) return BOOL;
        if (clazz == byte.class || clazz == Byte.class) return INT8;
        if (clazz == short.class || clazz == Short.class) return INT16;
        if (clazz == int.class || clazz == Integer.class) return INT32;
        if (clazz == long.class || clazz == Long.class) return INT64;
        if (clazz == float.class || clazz == Float.class) return FLOAT32;
        if (clazz == double.class || clazz == Double.class) return FLOAT64;
        if (clazz == char.class || clazz == Character.class) return CHAR;
        if (clazz == String.class) return STRING;
        return OBJECT;
    }

    static DType resolve(Object value) {
        if (value == null) return OBJECT;
        if (value instanceof Boolean) return BOOL;
        if (value instanceof Byte) return INT8;
        if (value instanceof Short) return INT16;
        if (value instanceof Integer) return INT32;
        if (value instanceof Long) return INT64;
        if (value instanceof Float) return FLOAT32;
        if (value instanceof Double) return FLOAT64;
        if (value instanceof Character) return CHAR;
        if (value instanceof String) return STRING;
        return OBJECT;
    }

    PromotionTable promotionTable = new PromotionTable();

    final class PromotionTable {
        private final Map<Long, DType> table = new HashMap<>();

        PromotionTable() {
            addRule(BOOL, BOOL, BOOL);
            addRule(BOOL, INT8, INT8);
            addRule(BOOL, INT16, INT16);
            addRule(BOOL, INT32, INT32);
            addRule(BOOL, INT64, INT64);
            addRule(BOOL, UINT8, UINT8);
            addRule(BOOL, UINT16, UINT16);
            addRule(BOOL, UINT32, UINT32);
            addRule(BOOL, UINT64, UINT64);
            addRule(BOOL, FLOAT16, FLOAT16);
            addRule(BOOL, FLOAT32, FLOAT32);
            addRule(BOOL, FLOAT64, FLOAT64);
            addRule(BOOL, COMPLEX64, COMPLEX64);
            addRule(BOOL, COMPLEX128, COMPLEX128);

            addRule(INT8, INT8, INT8);
            addRule(INT8, INT16, INT16);
            addRule(INT8, INT32, INT32);
            addRule(INT8, INT64, INT64);
            addRule(INT8, UINT8, INT16);
            addRule(INT8, UINT16, INT32);
            addRule(INT8, UINT32, INT64);
            addRule(INT8, UINT64, FLOAT64);
            addRule(INT8, FLOAT16, FLOAT32);
            addRule(INT8, FLOAT32, FLOAT32);
            addRule(INT8, FLOAT64, FLOAT64);
            addRule(INT8, COMPLEX64, COMPLEX64);
            addRule(INT8, COMPLEX128, COMPLEX128);

            addRule(INT16, INT16, INT16);
            addRule(INT16, INT32, INT32);
            addRule(INT16, INT64, INT64);
            addRule(INT16, UINT8, INT16);
            addRule(INT16, UINT16, INT32);
            addRule(INT16, UINT32, INT64);
            addRule(INT16, UINT64, FLOAT64);
            addRule(INT16, FLOAT16, FLOAT16);
            addRule(INT16, FLOAT32, FLOAT32);
            addRule(INT16, FLOAT64, FLOAT64);
            addRule(INT16, COMPLEX64, COMPLEX64);
            addRule(INT16, COMPLEX128, COMPLEX128);

            addRule(INT32, INT32, INT32);
            addRule(INT32, INT64, INT64);
            addRule(INT32, UINT8, INT32);
            addRule(INT32, UINT16, INT32);
            addRule(INT32, UINT32, INT64);
            addRule(INT32, UINT64, FLOAT64);
            addRule(INT32, FLOAT16, FLOAT32);
            addRule(INT32, FLOAT32, FLOAT32);
            addRule(INT32, FLOAT64, FLOAT64);
            addRule(INT32, COMPLEX64, COMPLEX64);
            addRule(INT32, COMPLEX128, COMPLEX128);

            addRule(INT64, INT64, INT64);
            addRule(INT64, UINT8, INT64);
            addRule(INT64, UINT16, INT64);
            addRule(INT64, UINT32, INT64);
            addRule(INT64, UINT64, FLOAT64);
            addRule(INT64, FLOAT16, FLOAT32);
            addRule(INT64, FLOAT32, FLOAT64);
            addRule(INT64, FLOAT64, FLOAT64);
            addRule(INT64, COMPLEX64, COMPLEX128);
            addRule(INT64, COMPLEX128, COMPLEX128);

            addRule(UINT8, UINT8, UINT8);
            addRule(UINT8, UINT16, UINT16);
            addRule(UINT8, UINT32, UINT32);
            addRule(UINT8, UINT64, UINT64);
            addRule(UINT8, FLOAT16, FLOAT16);
            addRule(UINT8, FLOAT32, FLOAT32);
            addRule(UINT8, FLOAT64, FLOAT64);
            addRule(UINT8, COMPLEX64, COMPLEX64);
            addRule(UINT8, COMPLEX128, COMPLEX128);

            addRule(UINT16, UINT16, UINT16);
            addRule(UINT16, UINT32, UINT32);
            addRule(UINT16, UINT64, UINT64);
            addRule(UINT16, FLOAT16, FLOAT16);
            addRule(UINT16, FLOAT32, FLOAT32);
            addRule(UINT16, FLOAT64, FLOAT64);
            addRule(UINT16, COMPLEX64, COMPLEX64);
            addRule(UINT16, COMPLEX128, COMPLEX128);

            addRule(UINT32, UINT32, UINT32);
            addRule(UINT32, UINT64, UINT64);
            addRule(UINT32, FLOAT16, FLOAT32);
            addRule(UINT32, FLOAT32, FLOAT64);
            addRule(UINT32, FLOAT64, FLOAT64);
            addRule(UINT32, COMPLEX64, COMPLEX64);
            addRule(UINT32, COMPLEX128, COMPLEX128);

            addRule(UINT64, UINT64, UINT64);
            addRule(UINT64, FLOAT16, FLOAT32);
            addRule(UINT64, FLOAT32, FLOAT64);
            addRule(UINT64, FLOAT64, FLOAT64);
            addRule(UINT64, COMPLEX64, COMPLEX128);
            addRule(UINT64, COMPLEX128, COMPLEX128);

            addRule(FLOAT16, FLOAT16, FLOAT16);
            addRule(FLOAT16, FLOAT32, FLOAT32);
            addRule(FLOAT16, FLOAT64, FLOAT64);
            addRule(FLOAT16, COMPLEX64, COMPLEX64);
            addRule(FLOAT16, COMPLEX128, COMPLEX128);

            addRule(FLOAT32, FLOAT32, FLOAT32);
            addRule(FLOAT32, FLOAT64, FLOAT64);
            addRule(FLOAT32, COMPLEX64, COMPLEX64);
            addRule(FLOAT32, COMPLEX128, COMPLEX128);

            addRule(FLOAT64, FLOAT64, FLOAT64);
            addRule(FLOAT64, COMPLEX64, COMPLEX128);
            addRule(FLOAT64, COMPLEX128, COMPLEX128);

            addRule(COMPLEX64, COMPLEX64, COMPLEX64);
            addRule(COMPLEX64, COMPLEX128, COMPLEX128);
            addRule(COMPLEX128, COMPLEX128, COMPLEX128);
        }

        private void addRule(DType a, DType b, DType result) {
            table.put(key(a, b), result);
        }

        DType get(DType a, DType b) {
            DType result = table.get(key(a, b));
            if (result == null) {
                if (a == STRING || b == STRING) return STRING;
                if (a == OBJECT || b == OBJECT) return OBJECT;
                return FLOAT64;
            }
            return result;
        }

        private static long key(DType a, DType b) {
            int ha = Objects.hashCode(a.name());
            int hb = Objects.hashCode(b.name());
            long h = ((long) ha << 32) | (hb & 0xFFFFFFFFL);
            long h2 = ((long) hb << 32) | (ha & 0xFFFFFFFFL);
            return Math.min(h, h2);
        }
    }
}
