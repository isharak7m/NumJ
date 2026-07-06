package jnumpy.creation;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class CreationTest {

    @Test
    void testArrayFrom1D() {
        NDArray a = Creation.array(new double[]{1.0, 2.0, 3.0});
        assertEquals(3, a.size());
        assertEquals(1.0, a.getDouble(0), 1e-15);
    }

    @Test
    void testArrayFrom2D() {
        NDArray a = Creation.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertArrayEquals(new int[]{2, 2}, a.shape());
        assertEquals(4.0, a.getDouble(1, 1), 1e-15);
    }

    @Test
    void testZeros() {
        NDArray a = Creation.zeros(2, 3);
        assertEquals(0.0, a.getDouble(0, 0), 1e-15);
        assertEquals(0.0, a.getDouble(1, 2), 1e-15);
    }

    @Test
    void testOnes() {
        NDArray a = Creation.ones(4);
        assertEquals(1.0, a.getDouble(0), 1e-15);
        assertEquals(1.0, a.getDouble(3), 1e-15);
    }

    @Test
    void testEye() {
        NDArray a = Creation.eye(3);
        assertEquals(1.0, a.getDouble(0, 0), 1e-15);
        assertEquals(1.0, a.getDouble(1, 1), 1e-15);
        assertEquals(0.0, a.getDouble(0, 1), 1e-15);
    }

    @Test
    void testArange() {
        NDArray a = Creation.arange(0, 10, 2.5);
        assertEquals(4, a.size());
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(7.5, a.getDouble(3), 1e-15);
    }

    @Test
    void testLinspace() {
        NDArray a = Creation.linspace(0, 1, 11);
        assertEquals(11, a.size());
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(0.5, a.getDouble(5), 1e-15);
        assertEquals(1.0, a.getDouble(10), 1e-15);
    }

    @Test
    void testFull() {
        NDArray a = Creation.full(new int[]{2, 2}, 3.14, DType.FLOAT64);
        assertEquals(3.14, a.getDouble(0, 0), 1e-15);
        assertEquals(3.14, a.getDouble(1, 1), 1e-15);
    }

    @Test
    void testFromFunction() {
        NDArray a = Creation.fromFunction(i -> i * 2.0, 5);
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(8.0, a.getDouble(4), 1e-15);
    }

    @Test
    void testGeomspace() {
        NDArray a = Creation.geomspace(1, 1000, 4);
        assertEquals(4, a.size());
        assertEquals(1.0, a.getDouble(0), 1e-12);
        assertEquals(1000.0, a.getDouble(3), 1e-12);
    }

    @Test
    void testLogspace() {
        NDArray a = Creation.logspace(0, 3, 4);
        assertEquals(4, a.size());
        assertEquals(1.0, a.getDouble(0), 1e-12);
        assertEquals(1000.0, a.getDouble(3), 1e-12);
    }
}
