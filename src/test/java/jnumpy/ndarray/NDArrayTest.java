package jnumpy.ndarray;

import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class NDArrayTest {

    @Test
    void testCreate1D() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0});
        assertArrayEquals(new int[]{3}, a.shape());
        assertEquals(1.0, a.getDouble(0), 1e-15);
    }

    @Test
    void testCreate2D() {
        NDArray a = NDArray.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertArrayEquals(new int[]{2, 2}, a.shape());
        assertEquals(1.0, a.getDouble(0, 0), 1e-15);
        assertEquals(4.0, a.getDouble(1, 1), 1e-15);
    }

    @Test
    void testCopy() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0});
        NDArray b = a.copy();
        b.setDouble(99.0, 0);
        assertEquals(1.0, a.getDouble(0), 1e-15);
    }

    @Test
    void testView() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0});
        NDArray v = a.view();
        v.setDouble(99.0, 0);
        assertEquals(99.0, a.getDouble(0), 1e-15);
    }

    @Test
    void testReshape() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray r = a.reshape(2, 2);
        assertArrayEquals(new int[]{2, 2}, r.shape());
    }

    @Test
    void testTranspose() {
        NDArray a = NDArray.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray t = a.transpose();
        assertEquals(1.0, t.getDouble(0, 0), 1e-15);
        assertEquals(3.0, t.getDouble(0, 1), 1e-15);
    }

    @Test
    void testIsContiguous() {
        NDArray a = NDArray.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        assertTrue(a.isContiguous());
    }

    @Test
    void testNegativeIndexing() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0});
        assertEquals(3.0, a.getDouble(-1), 1e-15);
    }

    @Test
    void testSqueeze() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0, 3.0}).reshape(1, 3, 1);
        NDArray s = a.squeeze();
        assertArrayEquals(new int[]{3}, s.shape());
    }

    @Test
    void testExpandDims() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0});
        NDArray e = a.expandDims(0);
        assertArrayEquals(new int[]{1, 2}, e.shape());
    }

    @Test
    void testRavel() {
        NDArray a = NDArray.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray r = a.ravel();
        assertArrayEquals(new int[]{4}, r.shape());
    }

    @Test
    void testSwapAxes() {
        NDArray a = NDArray.create(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray s = a.swapAxes(0, 1);
        assertEquals(3.0, s.getDouble(0, 1), 1e-15);
    }

    @Test
    void testRepeat() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0});
        NDArray r = a.repeat(3, 0);
        assertEquals(6, r.size());
    }

    @Test
    void testEquality() {
        NDArray a = NDArray.create(new double[]{1.0, 2.0});
        NDArray b = NDArray.create(new double[]{1.0, 2.0});
        assertEquals(a, b);
    }
}
