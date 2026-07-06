package jnumpy.dtype;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class DTypeTest {

    @Test
    void testConstants() {
        assertEquals("bool", DType.BOOL.name());
        assertEquals("int32", DType.INT32.name());
        assertEquals("float64", DType.FLOAT64.name());
    }

    @Test
    void testKind() {
        assertEquals(DType.Kind.SIGNED_INT, DType.INT32.kind());
        assertEquals(DType.Kind.FLOAT, DType.FLOAT64.kind());
        assertEquals(DType.Kind.BOOL, DType.BOOL.kind());
    }

    @Test
    void testByteSize() {
        assertEquals(1, DType.INT8.byteSize());
        assertEquals(2, DType.INT16.byteSize());
        assertEquals(4, DType.INT32.byteSize());
        assertEquals(8, DType.INT64.byteSize());
        assertEquals(4, DType.FLOAT32.byteSize());
        assertEquals(8, DType.FLOAT64.byteSize());
    }

    @Test
    void testPromotion() {
        assertEquals(DType.FLOAT64, DType.INT32.promotedWith(DType.FLOAT64));
        assertEquals(DType.FLOAT64, DType.INT64.promotedWith(DType.FLOAT64));
        assertEquals(DType.FLOAT32, DType.INT8.promotedWith(DType.FLOAT16));
        assertEquals(DType.COMPLEX128, DType.FLOAT64.promotedWith(DType.COMPLEX64));
    }

    @Test
    void testIsNumeric() {
        assertTrue(DType.INT32.isNumeric());
        assertTrue(DType.FLOAT64.isNumeric());
        assertFalse(DType.STRING.isNumeric());
        assertFalse(DType.BOOL.isNumeric());
    }

    @Test
    void testFromClass() {
        assertEquals(DType.INT32, DType.fromClass(int.class));
        assertEquals(DType.FLOAT64, DType.fromClass(double.class));
        assertEquals(DType.BOOL, DType.fromClass(boolean.class));
    }

    @Test
    void testResolve() {
        assertEquals(DType.INT32, DType.resolve(42));
        assertEquals(DType.FLOAT64, DType.resolve(3.14));
        assertEquals(DType.STRING, DType.resolve("hello"));
    }

    @Test
    void testPromotionTable() {
        assertEquals(DType.INT64, DType.INT32.promotedWith(DType.UINT32));
        assertEquals(DType.FLOAT64, DType.INT64.promotedWith(DType.UINT64));
    }
}
