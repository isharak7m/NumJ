package jnumpy.simd;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jdk.incubator.vector.*;
import java.lang.foreign.MemorySegment;

public final class SIMD {

    private static final VectorSpecies<Double> DOUBLE_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Long> LONG_SPECIES = LongVector.SPECIES_PREFERRED;

    private SIMD() {}

    public static boolean isSupported() {
        return DOUBLE_SPECIES.length() >= 2;
    }

    public static int vectorLength() {
        return DOUBLE_SPECIES.length();
    }

    public static void add(double[] a, double[] b, double[] out) {
        int i = 0;
        int len = DOUBLE_SPECIES.loopBound(a.length);
        for (; i < len; i += DOUBLE_SPECIES.length()) {
            var va = DoubleVector.fromArray(DOUBLE_SPECIES, a, i);
            var vb = DoubleVector.fromArray(DOUBLE_SPECIES, b, i);
            va.add(vb).intoArray(out, i);
        }
        for (; i < a.length; i++) out[i] = a[i] + b[i];
    }

    public static void mul(double[] a, double[] b, double[] out) {
        int i = 0;
        int len = DOUBLE_SPECIES.loopBound(a.length);
        for (; i < len; i += DOUBLE_SPECIES.length()) {
            var va = DoubleVector.fromArray(DOUBLE_SPECIES, a, i);
            var vb = DoubleVector.fromArray(DOUBLE_SPECIES, b, i);
            va.mul(vb).intoArray(out, i);
        }
        for (; i < a.length; i++) out[i] = a[i] * b[i];
    }

    public static void fma(double[] a, double[] b, double[] c, double[] out) {
        int i = 0;
        int len = DOUBLE_SPECIES.loopBound(a.length);
        for (; i < len; i += DOUBLE_SPECIES.length()) {
            var va = DoubleVector.fromArray(DOUBLE_SPECIES, a, i);
            var vb = DoubleVector.fromArray(DOUBLE_SPECIES, b, i);
            var vc = DoubleVector.fromArray(DOUBLE_SPECIES, c, i);
            va.fma(vb, vc).intoArray(out, i);
        }
        for (; i < a.length; i++) out[i] = a[i] * b[i] + c[i];
    }

    public static void sqrt(double[] a, double[] out) {
        int i = 0;
        int len = DOUBLE_SPECIES.loopBound(a.length);
        for (; i < len; i += DOUBLE_SPECIES.length()) {
            var va = DoubleVector.fromArray(DOUBLE_SPECIES, a, i);
            va.sqrt().intoArray(out, i);
        }
        for (; i < a.length; i++) out[i] = Math.sqrt(a[i]);
    }

    public static double sum(double[] a) {
        double sum = 0;
        int i = 0;
        int len = DOUBLE_SPECIES.loopBound(a.length);
        var vSum = DoubleVector.zero(DOUBLE_SPECIES);
        for (; i < len; i += DOUBLE_SPECIES.length()) {
            var va = DoubleVector.fromArray(DOUBLE_SPECIES, a, i);
            vSum = vSum.add(va);
        }
        sum = vSum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) sum += a[i];
        return sum;
    }

    public static void add(float[] a, float[] b, float[] out) {
        int i = 0;
        int len = FLOAT_SPECIES.loopBound(a.length);
        for (; i < len; i += FLOAT_SPECIES.length()) {
            var va = FloatVector.fromArray(FLOAT_SPECIES, a, i);
            var vb = FloatVector.fromArray(FLOAT_SPECIES, b, i);
            va.add(vb).intoArray(out, i);
        }
        for (; i < a.length; i++) out[i] = a[i] + b[i];
    }
}
