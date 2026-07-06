package jnumpy.statistics;

import jnumpy.ndarray.NDArray;
import jnumpy.core.Arrays;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class StatisticsTest {

    @Test
    void testSum() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray s = Statistics.sum(a);
        assertEquals(10.0, s.getDouble(0), 1e-12);
    }

    @Test
    void testSum2D() {
        NDArray a = Arrays.array(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        NDArray s0 = Statistics.sum(a, 0);
        assertEquals(4.0, s0.getDouble(0), 1e-12);
        assertEquals(6.0, s0.getDouble(1), 1e-12);
    }

    @Test
    void testMean() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray m = Statistics.mean(a);
        assertEquals(2.5, m.getDouble(0), 1e-12);
    }

    @Test
    void testProd() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray p = Statistics.prod(a);
        assertEquals(24.0, p.getDouble(0), 1e-12);
    }

    @Test
    void testVar() {
        NDArray a = Arrays.array(new double[]{2.0, 2.0, 2.0, 2.0});
        NDArray v = Statistics.var(a);
        assertEquals(0.0, v.getDouble(0), 1e-12);
    }

    @Test
    void testStd() {
        NDArray a = Arrays.array(new double[]{0.0, 1.0});
        NDArray s = Statistics.std(a);
        assertEquals(0.5, s.getDouble(0), 1e-12);
    }

    @Test
    void testMin() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 4.0, 1.0});
        NDArray m = Statistics.min(a);
        assertEquals(1.0, m.getDouble(0), 1e-12);
    }

    @Test
    void testMax() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 4.0, 1.0});
        NDArray m = Statistics.max(a);
        assertEquals(4.0, m.getDouble(0), 1e-12);
    }

    @Test
    void testCumsum() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0});
        NDArray c = Statistics.cumsum(a);
        assertEquals(1.0, c.getDouble(0), 1e-12);
        assertEquals(3.0, c.getDouble(1), 1e-12);
        assertEquals(6.0, c.getDouble(2), 1e-12);
    }

    @Test
    void testCumprod() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray c = Statistics.cumprod(a);
        assertEquals(1.0, c.getDouble(0), 1e-12);
        assertEquals(2.0, c.getDouble(1), 1e-12);
        assertEquals(6.0, c.getDouble(2), 1e-12);
        assertEquals(24.0, c.getDouble(3), 1e-12);
    }

    @Test
    void testArgmin() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 4.0, 1.0});
        NDArray am = Statistics.argmin(a);
        assertTrue(am.getDouble(0) == 1.0 || am.getDouble(0) == 3.0);
    }

    @Test
    void testArgmax() {
        NDArray a = Arrays.array(new double[]{3.0, 1.0, 4.0, 1.0});
        NDArray am = Statistics.argmax(a);
        assertEquals(2.0, am.getDouble(0), 1e-12);
    }

    @Test
    void testCountNonzero() {
        NDArray a = Arrays.array(new double[]{0.0, 1.0, 0.0, 2.0, 0.0});
        NDArray c = Statistics.countNonzero(a);
        assertEquals(2.0, c.getDouble(0), 1e-12);
    }

    @Test
    void testPercentile() {
        NDArray a = Arrays.array(new double[]{1.0, 2.0, 3.0, 4.0});
        NDArray p = Statistics.percentile(a, 50);
        assertEquals(2.5, p.getDouble(0), 1e-12);
    }
}
