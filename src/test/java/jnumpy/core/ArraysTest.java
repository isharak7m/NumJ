package jnumpy.core;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class ArraysTest {

    @Test
    void testArrayFromDoubleArray() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        assertEquals(3, a.size());
        assertEquals(1.0, a.getDouble(0), 1e-15);
        assertEquals(2.0, a.getDouble(1), 1e-15);
        assertEquals(3.0, a.getDouble(2), 1e-15);
    }

    @Test
    void testArrayFromIntArray() {
        NDArray a = Arrays.array(new int[]{1, 2, 3});
        assertEquals(DType.INT32, a.dtype());
        assertEquals(1, a.getInt(0));
    }

    @Test
    void testZeros() {
        NDArray a = Arrays.zeros(3, 4);
        assertArrayEquals(new int[]{3, 4}, a.shape());
        assertEquals(0.0, a.getDouble(0, 0), 1e-15);
        assertEquals(12, a.size());
    }

    @Test
    void testOnes() {
        NDArray a = Arrays.ones(5);
        assertEquals(1.0, a.getDouble(0), 1e-15);
        assertEquals(1.0, a.getDouble(4), 1e-15);
    }

    @Test
    void testEye() {
        NDArray a = Arrays.eye(3);
        assertEquals(1.0, a.getDouble(0, 0), 1e-15);
        assertEquals(1.0, a.getDouble(1, 1), 1e-15);
        assertEquals(1.0, a.getDouble(2, 2), 1e-15);
        assertEquals(0.0, a.getDouble(0, 1), 1e-15);
    }

    @Test
    void testArange() {
        NDArray a = Arrays.arange(5);
        assertEquals(5, a.size());
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(4.0, a.getDouble(4), 1e-15);
    }

    @Test
    void testArangeStartStop() {
        NDArray a = Arrays.arange(2, 7);
        assertEquals(5, a.size());
        assertEquals(2.0, a.getDouble(0), 1e-15);
        assertEquals(6.0, a.getDouble(4), 1e-15);
    }

    @Test
    void testArangeStep() {
        NDArray a = Arrays.arange(0, 10, 3);
        assertEquals(4, a.size());
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(9.0, a.getDouble(3), 1e-15);
    }

    @Test
    void testLinspace() {
        NDArray a = Arrays.linspace(0, 1, 5);
        assertEquals(5, a.size());
        assertEquals(0.0, a.getDouble(0), 1e-15);
        assertEquals(0.25, a.getDouble(1), 1e-15);
        assertEquals(0.5, a.getDouble(2), 1e-15);
        assertEquals(0.75, a.getDouble(3), 1e-15);
        assertEquals(1.0, a.getDouble(4), 1e-15);
    }

    @Test
    void testFull() {
        NDArray a = Arrays.full(new int[]{2, 3}, 7.0);
        assertEquals(7.0, a.getDouble(0, 0), 1e-15);
        assertEquals(7.0, a.getDouble(1, 2), 1e-15);
    }

    @Test
    void testIdentity() {
        NDArray a = Arrays.identity(4);
        assertEquals(1.0, a.getDouble(0, 0), 1e-15);
        assertEquals(1.0, a.getDouble(3, 3), 1e-15);
        assertEquals(0.0, a.getDouble(0, 1), 1e-15);
    }

    @Test
    void testAdd() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray b = Arrays.array(new double[]{4.0, 5.0, 6.0});
        NDArray c = Arrays.add(a, b);
        assertEquals(5.0, c.getDouble(0), 1e-15);
        assertEquals(7.0, c.getDouble(1), 1e-15);
        assertEquals(9.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testSubtract() {
        NDArray a = Arrays.array(new double[]{5.0, 7.0, 9.0});
        NDArray b = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray c = Arrays.subtract(a, b);
        assertEquals(4.0, c.getDouble(0), 1e-15);
        assertEquals(5.0, c.getDouble(1), 1e-15);
        assertEquals(6.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testMultiply() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray b = Arrays.array(new double[]{4.0, 5.0, 6.0});
        NDArray c = Arrays.multiply(a, b);
        assertEquals(4.0, c.getDouble(0), 1e-15);
        assertEquals(10.0, c.getDouble(1), 1e-15);
        assertEquals(18.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testDivide() {
        NDArray a = Arrays.array(new double[]{1.0, 4.0, 9.0});
        NDArray b = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray c = Arrays.divide(a, b);
        assertEquals(1.0, c.getDouble(0), 1e-15);
        assertEquals(2.0, c.getDouble(1), 1e-15);
        assertEquals(3.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testSum() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray s = Arrays.sum(a);
        assertEquals(10.0, s.getDouble(0), 1e-15);
    }

    @Test
    void testSumAxis() {
        NDArray a = Arrays.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray s = Arrays.sum(a, 0);
        assertEquals(4.0, s.getDouble(0), 1e-15);
        assertEquals(6.0, s.getDouble(1), 1e-15);
    }

    @Test
    void testMean() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray m = Arrays.mean(a);
        assertEquals(2.5, m.getDouble(0), 1e-15);
    }

    @Test
    void testStd() {
        NDArray a = Arrays.array(new double[]{1.0, 1.0, 1.0, 1.0});
        NDArray s = Arrays.std(a);
        assertEquals(0.0, s.getDouble(0), 1e-15);
    }

    @Test
    void testMinMax() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 4.0, 1.0, 5.0});
        assertEquals(1.0, Arrays.min(a).getDouble(0), 1e-15);
        assertEquals(5.0, Arrays.max(a).getDouble(0), 1e-15);
    }

    @Test
    void testReshape() {
        NDArray a = Arrays.arange(6);
        NDArray b = Arrays.reshape(a, 2, 3);
        assertArrayEquals(new int[]{2, 3}, b.shape());
        assertEquals(0.0, b.getDouble(0, 0), 1e-15);
        assertEquals(5.0, b.getDouble(1, 2), 1e-15);
    }

    @Test
    void testTranspose() {
        NDArray a = Arrays.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray b = Arrays.transpose(a);
        assertEquals(1.0, b.getDouble(0, 0), 1e-15);
        assertEquals(3.0, b.getDouble(0, 1), 1e-15);
        assertEquals(2.0, b.getDouble(1, 0), 1e-15);
        assertEquals(4.0, b.getDouble(1, 1), 1e-15);
    }

    @Test
    void testDot() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray b = Arrays.array(new double[]{4.0, 5.0, 6.0});
        NDArray c = Arrays.dot(a, b);
        assertEquals(32.0, c.getDouble(0), 1e-15);
    }

    @Test
    void testMatmul() {
        NDArray a = Arrays.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray b = Arrays.array(new double[][]{{5.0, 6.0}, {7.0, 8.0}});
        NDArray c = Arrays.matmul(a, b);
        assertEquals(19.0, c.getDouble(0, 0), 1e-15);
        assertEquals(22.0, c.getDouble(0, 1), 1e-15);
        assertEquals(43.0, c.getDouble(1, 0), 1e-15);
        assertEquals(50.0, c.getDouble(1, 1), 1e-15);
    }

    @Test
    void testSqrt() {
        NDArray a = Arrays.array(new double[]{1.0, 4.0, 9.0, 16.0});
        NDArray s = Arrays.sqrt(a);
        assertEquals(1.0, s.getDouble(0), 1e-15);
        assertEquals(2.0, s.getDouble(1), 1e-15);
        assertEquals(3.0, s.getDouble(2), 1e-15);
        assertEquals(4.0, s.getDouble(3), 1e-15);
    }

    @Test
    void testExp() {
        NDArray a = Arrays.array(new double[]{0.0, 1.0});
        NDArray e = Arrays.exp(a);
        assertEquals(1.0, e.getDouble(0), 1e-15);
        assertEquals(Math.E, e.getDouble(1), 1e-15);
    }

    @Test
    void testLog() {
        NDArray a = Arrays.array(new double[]{1.0, Math.E});
        NDArray l = Arrays.log(a);
        assertEquals(0.0, l.getDouble(0), 1e-15);
        assertEquals(1.0, l.getDouble(1), 1e-15);
    }

    @Test
    void testSinCos() {
        NDArray a = Arrays.array(new double[]{0.0, Math.PI / 2});
        NDArray s = Arrays.sin(a);
        NDArray c = Arrays.cos(a);
        assertEquals(0.0, s.getDouble(0), 1e-15);
        assertEquals(1.0, s.getDouble(1), 1e-15);
        assertEquals(1.0, c.getDouble(0), 1e-15);
        assertEquals(0.0, c.getDouble(1), 1e-10);
    }

    @Test
    void testPower() {
        NDArray a = Arrays.array(new double[]{2.0, 3.0, 4.0});
        NDArray b = Arrays.array(new double[]{3.0, 2.0, 1.0});
        NDArray p = Arrays.power(a, b);
        assertEquals(8.0, p.getDouble(0), 1e-15);
        assertEquals(9.0, p.getDouble(1), 1e-15);
        assertEquals(4.0, p.getDouble(2), 1e-15);
    }

    @Test
    void testBroadcasting() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray b = Arrays.array(new double[]{2.0});
        NDArray c = Arrays.multiply(a, b);
        assertEquals(2.0, c.getDouble(0), 1e-15);
        assertEquals(4.0, c.getDouble(1), 1e-15);
        assertEquals(6.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testConstants() {
        assertEquals(Math.PI, Arrays.PI, 1e-15);
        assertEquals(Math.E, Arrays.E, 1e-15);
        assertTrue(Double.isInfinite(Arrays.INF));
        assertTrue(Double.isNaN(Arrays.NAN));
    }

    @Test
    void testFlattenRavel() {
        NDArray a = Arrays.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray f = Arrays.flatten(a);
        NDArray r = Arrays.ravel(a);
        assertEquals(4, f.size());
        assertEquals(4, r.size());
        assertEquals(1.0, f.getDouble(0), 1e-15);
        assertEquals(4.0, f.getDouble(3), 1e-15);
    }

    @Test
    void testSqueeze() {
        NDArray a = Arrays.ones(1, 3, 1);
        NDArray s = Arrays.squeeze(a);
        assertArrayEquals(new int[]{3}, s.shape());
    }

    @Test
    void testExpandDims() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray e = Arrays.expandDims(a, 0);
        assertArrayEquals(new int[]{1, 3}, e.shape());
    }

    @Test
    void testClip() {
        NDArray a = Arrays.array(new double[]{-1.0, 0.5, 2.0});
        NDArray c = Arrays.clip(a, 0.0, 1.0);
        assertEquals(0.0, c.getDouble(0), 1e-15);
        assertEquals(0.5, c.getDouble(1), 1e-15);
        assertEquals(1.0, c.getDouble(2), 1e-15);
    }

    @Test
    void testWhere() {
        NDArray cond = Arrays.array(new double[]{1.0, 0.0, 1.0});
        NDArray x = Arrays.array(new double[]{10.0, 20.0, 30.0});
        NDArray y = Arrays.array(new double[]{-1.0, -2.0, -3.0});
        NDArray w = Arrays.where(cond, x, y);
        assertEquals(10.0, w.getDouble(0), 1e-15);
        assertEquals(-2.0, w.getDouble(1), 1e-15);
        assertEquals(30.0, w.getDouble(2), 1e-15);
    }

    @Test
    void testUnique() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 2.0, 1.0, 3.0});
        NDArray u = Arrays.unique(a);
        assertEquals(3, u.size());
        assertEquals(1.0, u.getDouble(0), 1e-15);
        assertEquals(2.0, u.getDouble(1), 1e-15);
        assertEquals(3.0, u.getDouble(2), 1e-15);
    }

    @Test
    void testSort() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 2.0});
        NDArray s = Arrays.sort(a);
        assertEquals(1.0, s.getDouble(0), 1e-15);
        assertEquals(2.0, s.getDouble(1), 1e-15);
        assertEquals(3.0, s.getDouble(2), 1e-15);
    }

    @Test
    void testArgmax() {
        NDArray a = Arrays.array(new double[]{1.0, 3.0, 2.0});
        NDArray am = Arrays.argmax(a);
        assertEquals(1.0, am.getDouble(0), 1e-15);
    }

    @Test
    void testArgmin() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 2.0});
        NDArray am = Arrays.argmin(a);
        assertEquals(1.0, am.getDouble(0), 1e-15);
    }

    @Test
    void testCountNonzero() {
        NDArray a = Arrays.array(new double[]{0.0, 1.0, 2.0, 0.0, 3.0});
        NDArray c = Arrays.countNonzero(a);
        assertEquals(3.0, c.getDouble(0), 1e-15);
    }

    @Test
    void testCumsum() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray c = Arrays.cumsum(a);
        assertEquals(1.0, c.getDouble(0), 1e-15);
        assertEquals(3.0, c.getDouble(1), 1e-15);
        assertEquals(6.0, c.getDouble(2), 1e-15);
        assertEquals(10.0, c.getDouble(3), 1e-15);
    }

    @Test
    void testCumprod() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray c = Arrays.cumprod(a);
        assertEquals(1.0, c.getDouble(0), 1e-15);
        assertEquals(2.0, c.getDouble(1), 1e-15);
        assertEquals(6.0, c.getDouble(2), 1e-15);
        assertEquals(24.0, c.getDouble(3), 1e-15);
    }

    @Test
    void testFlip() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray f = Arrays.flip(a);
        assertEquals(3.0, f.getDouble(0), 1e-15);
        assertEquals(2.0, f.getDouble(1), 1e-15);
        assertEquals(1.0, f.getDouble(2), 1e-15);
    }
}
