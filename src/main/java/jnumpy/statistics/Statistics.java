package jnumpy.statistics;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.util.Arrays;

public final class Statistics {

    private Statistics() {}

    public static NDArray sum(NDArray a, int... axis) {
        return reduce(a, axis, "sum", 0.0, Double::sum);
    }

    public static NDArray prod(NDArray a, int... axis) {
        return reduce(a, axis, "prod", 1.0, (x, y) -> x * y);
    }

    public static NDArray mean(NDArray a, int... axis) {
        if (axis.length == 0) {
            double s = sum(a, 0).getDouble(0);
            return NDArray.create(new double[]{ s / a.size() });
        }
        NDArray s = sum(a, axis);
        long divisor = 1;
        for (int ax : axis) divisor *= a.shape(ax);
        double div = divisor;
        return applyScalarOp(s, "mean_div", v -> v / div);
    }

    public static NDArray var(NDArray a, int... axis) {
        NDArray m = mean(a, axis);
        NDArray diff = ufuncOp(a, m, (x, y) -> (x - y) * (x - y));
        NDArray s = sum(diff, axis);
        if (axis.length == 0) {
            return NDArray.create(new double[]{ s.getDouble(0) / a.size() });
        }
        long divisor = 1;
        for (int ax : axis) divisor *= a.shape(ax);
        double div = divisor;
        return applyScalarOp(s, "var_div", v -> v / div);
    }

    public static NDArray std(NDArray a, int... axis) {
        NDArray v = var(a, axis);
        return applyScalarOp(v, "std_sqrt", Math::sqrt);
    }

    public static NDArray min(NDArray a, int... axis) {
        return reduce(a, axis, "min", Double.POSITIVE_INFINITY, Math::min);
    }

    public static NDArray max(NDArray a, int... axis) {
        return reduce(a, axis, "max", Double.NEGATIVE_INFINITY, Math::max);
    }

    public static NDArray argmin(NDArray a, int... axis) {
        return argReduce(a, axis, "argmin", false);
    }

    public static NDArray argmax(NDArray a, int... axis) {
        return argReduce(a, axis, "argmax", true);
    }

    public static NDArray cumsum(NDArray a, int... axis) {
        return cumulativeOp(a, axis, "cumsum", 0.0, Double::sum);
    }

    public static NDArray cumprod(NDArray a, int... axis) {
        return cumulativeOp(a, axis, "cumprod", 1.0, (x, y) -> x * y);
    }

    public static NDArray countNonzero(NDArray a, int... axis) {
        NDArray nonzeros = applyScalarOp(a, "nonzero", v -> v != 0 ? 1.0 : 0.0);
        return sum(nonzeros, axis);
    }

    public static NDArray quantile(NDArray a, double q, int... axis) {
        NDArray sorted = jnumpy.sort.Sort.sort(a, axis);
        if (axis.length == 0) {
            long n = a.size();
            double idx = q * (n - 1);
            long lo = (long) Math.floor(idx);
            long hi = (long) Math.ceil(idx);
            double frac = idx - lo;
            double vlo = sorted.getDouble(new int[]{ (int) lo });
            double vhi = sorted.getDouble(new int[]{ (int) hi });
            return NDArray.create(new double[]{ vlo + frac * (vhi - vlo) });
        }
        return null;
    }

    public static NDArray percentile(NDArray a, double p, int... axis) {
        return quantile(a, p / 100.0, axis);
    }

    static NDArray reduce(NDArray a, int[] axis, String name, double identity, java.util.function.DoubleBinaryOperator op) {
        if (axis.length == 0) {
            double result = identity;
            int[] indices = new int[a.ndim()];
            for (long i = 0; i < a.size(); i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    indices[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                result = op.applyAsDouble(result, a.getDouble(indices));
            }
            return NDArray.create(new double[]{ result });
        }
        int[] shape = a.shape();
        java.util.HashSet<Integer> axisSet = new java.util.HashSet<>();
        for (int ax : axis) axisSet.add(ax < 0 ? ax + shape.length : ax);
        int[] finalShape = new int[shape.length - axisSet.size()];
        int[] finalStrides = new int[shape.length - axisSet.size()];
        int fi = 0;
        for (int i = 0; i < shape.length; i++) {
            if (!axisSet.contains(i)) {
                finalShape[fi] = shape[i];
                finalStrides[fi] = fi == 0 ? 1 : finalStrides[fi - 1] * finalShape[fi - 1];
                fi++;
            }
        }
        if (finalShape.length == 0) { finalShape = new int[]{ 1 }; finalStrides = new int[]{ 1 }; }
        NDArray result = NDArray.create(finalShape, a.dtype());
        int[] srcIdx = new int[shape.length];
        int[] dstIdx = new int[finalShape.length];
        for (long flat = 0; flat < a.size(); flat++) {
            long remaining = flat;
            for (int d = shape.length - 1; d >= 0; d--) {
                srcIdx[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            fi = 0;
            for (int d = 0; d < shape.length; d++) {
                if (!axisSet.contains(d)) dstIdx[fi++] = srcIdx[d];
            }
            int dstFlat = 0;
            for (int d = finalShape.length - 1; d >= 0; d--) {
                dstFlat = dstFlat * finalShape[d] + dstIdx[d];
            }
            if (flat == 0) {
                result.setDouble(identity, dstIdx);
            }
            double current = result.getDouble(dstIdx);
            result.setDouble(op.applyAsDouble(current, a.getDouble(srcIdx)), dstIdx);
        }
        return result;
    }

    private static NDArray argReduce(NDArray a, int[] axis, String name, boolean findMax) {
        if (axis.length == 0) {
            double bestVal = findMax ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
            long bestIdx = 0;
            int[] indices = new int[a.ndim()];
            for (long i = 0; i < a.size(); i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    indices[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                double val = a.getDouble(indices);
                boolean better = findMax ? val > bestVal : val < bestVal;
                if (better) { bestVal = val; bestIdx = i; }
            }
            return NDArray.create(new double[]{ bestIdx });
        }
        return a;
    }

    private static NDArray cumulativeOp(NDArray a, int[] axis, String name, double identity, java.util.function.DoubleBinaryOperator op) {
        if (axis.length == 0) {
            int n = (int) a.size();
            NDArray result = NDArray.create(new int[]{ n }, a.dtype());
            double acc = identity;
            int[] indices = new int[1];
            for (int i = 0; i < n; i++) {
                acc = op.applyAsDouble(acc, a.getDouble(i));
                result.setDouble(acc, i);
            }
            return result;
        }
        int ax = axis[0];
        int[] shape = a.shape();
        NDArray result = NDArray.create(shape, a.dtype());
        int outerSize = 1;
        for (int i = 0; i < ax; i++) outerSize *= shape[i];
        int innerSize = 1;
        for (int i = ax + 1; i < shape.length; i++) innerSize *= shape[i];
        int dimSize = shape[ax];
        int[] indices = new int[a.ndim()];
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                double acc = identity;
                int outerRemaining = outer;
                for (int i = ax - 1; i >= 0; i--) {
                    indices[i] = outerRemaining % shape[i];
                    outerRemaining /= shape[i];
                }
                int innerRemaining = inner;
                for (int i = shape.length - 1; i > ax; i--) {
                    indices[i] = innerRemaining % shape[i];
                    innerRemaining /= shape[i];
                }
                for (int d = 0; d < dimSize; d++) {
                    indices[ax] = d;
                    acc = op.applyAsDouble(acc, a.getDouble(indices));
                    result.setDouble(acc, indices);
                }
            }
        }
        return result;
    }

    static NDArray applyScalarOp(NDArray a, String name, java.util.function.DoubleUnaryOperator op) {
        NDArray result = NDArray.create(a.shape(), a.dtype());
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result.setDouble(op.applyAsDouble(a.getDouble(indices)), indices);
        }
        return result;
    }

    static NDArray ufuncOp(NDArray a, NDArray b, java.util.function.DoubleBinaryOperator op) {
        int[] shape = jnumpy.broadcast.Broadcast.broadcastShape(a.shape(), b.shape());
        NDArray ba = jnumpy.broadcast.Broadcast.broadcastTo(a, shape);
        NDArray bb = jnumpy.broadcast.Broadcast.broadcastTo(b, shape);
        NDArray result = NDArray.create(shape, a.dtype().promotedWith(b.dtype()));
        int[] indices = new int[shape.length];
        for (long i = 0; i < result.size(); i++) {
            long remaining = i;
            for (int d = shape.length - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            result.setDouble(op.applyAsDouble(ba.getDouble(indices), bb.getDouble(indices)), indices);
        }
        return result;
    }
}
