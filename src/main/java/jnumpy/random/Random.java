package jnumpy.random;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.util.concurrent.ThreadLocalRandom;

public final class Random {

    private static final ThreadLocal<java.util.Random> generators =
            ThreadLocal.withInitial(java.util.Random::new);

    private Random() {}

    public static void seed(long seed) {
        generators.set(new java.util.Random(seed));
    }

    public static NDArray uniform(double low, double high, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++)
            buf.setDouble(i, low + (high - low) * generators.get().nextDouble());
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray randn(int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++)
            buf.setDouble(i, generators.get().nextGaussian());
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray normal(double mean, double std, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++)
            buf.setDouble(i, mean + std * generators.get().nextGaussian());
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray randint(int low, int high, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.INT64, size);
        for (long i = 0; i < size; i++)
            buf.setLong(i, low + generators.get().nextInt(high - low));
        return new NDArray(buf, shape, DType.INT64, 'C');
    }

    public static NDArray random(int... shape) {
        return uniform(0, 1, shape);
    }

    public static NDArray poisson(double lam, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) {
            double L = Math.exp(-lam);
            double p = 1.0;
            int k = 0;
            do {
                k++;
                p *= generators.get().nextDouble();
            } while (p > L);
            buf.setDouble(i, k - 1);
        }
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray exponential(double scale, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++)
            buf.setDouble(i, -scale * Math.log(1 - generators.get().nextDouble()));
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray gamma(double shape, double scale, int... outShape) {
        long size = NDArray.sizeOfShape(outShape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++)
            buf.setDouble(i, sampleGamma(shape, scale));
        return new NDArray(buf, outShape, DType.FLOAT64, 'C');
    }

    public static NDArray beta(double a, double b, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) {
            double x = sampleGamma(a, 1);
            double y = sampleGamma(b, 1);
            buf.setDouble(i, x / (x + y));
        }
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray binomial(int n, double p, int... shape) {
        long size = NDArray.sizeOfShape(shape);
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, size);
        for (long i = 0; i < size; i++) {
            int count = 0;
            for (int j = 0; j < n; j++)
                if (generators.get().nextDouble() < p) count++;
            buf.setDouble(i, count);
        }
        return new NDArray(buf, shape, DType.FLOAT64, 'C');
    }

    public static NDArray choice(NDArray a, int size, boolean replace) {
        int n = (int) a.size();
        NDArray result = NDArray.create(new int[]{ size }, DType.FLOAT64);
        if (replace) {
            for (int i = 0; i < size; i++)
                result.setDouble(a.getDouble(generators.get().nextInt(n)), i);
        } else {
            int[] available = new int[n];
            for (int i = 0; i < n; i++) available[i] = i;
            for (int i = 0; i < size; i++) {
                int idx = generators.get().nextInt(n - i);
                result.setDouble(a.getDouble(available[idx]), i);
                available[idx] = available[n - i - 1];
            }
        }
        return result;
    }

    public static NDArray shuffle(NDArray a) {
        int n = (int) a.size();
        for (int i = n - 1; i > 0; i--) {
            int j = generators.get().nextInt(i + 1);
            double tmp = a.getDouble(i);
            a.setDouble(a.getDouble(j), i);
            a.setDouble(tmp, j);
        }
        return a;
    }

    public static NDArray permutation(int n) {
        NDArray result = NDArray.create(new int[]{ n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) result.setDouble(i, i);
        return shuffle(result);
    }

    private static double sampleGamma(double shape, double scale) {
        if (shape < 1) {
            double u = generators.get().nextDouble();
            return sampleGamma(shape + 1, scale) * Math.pow(u, 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x, v;
            do {
                x = generators.get().nextGaussian();
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = generators.get().nextDouble();
            if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
        }
    }
}
