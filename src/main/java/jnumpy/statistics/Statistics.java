/*
 * Copyright 2026 JNumj Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jnumpy.statistics;

import jnumpy.dtype.DType;
import jnumpy.ndarray.NDArray;
import jnumpy.util.Util;

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
            return NDArray.create(new double[] {s / a.size()});
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
            return NDArray.create(new double[] {s.getDouble(0) / a.size()});
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
            double vlo = Util.readElement(sorted, (int) lo);
            double vhi = Util.readElement(sorted, (int) hi);
            return NDArray.create(new double[] {vlo + frac * (vhi - vlo)});
        }
        int[] shape = a.shape();
        java.util.HashSet<Integer> axisSet = new java.util.HashSet<>();
        for (int ax : axis) axisSet.add(ax < 0 ? ax + shape.length : ax);
        int[] resultShape = new int[shape.length - axisSet.size()];
        int fi = 0;
        for (int i = 0; i < shape.length; i++) {
            if (!axisSet.contains(i)) resultShape[fi++] = shape[i];
        }
        if (resultShape.length == 0) resultShape = new int[] {1};
        NDArray result = NDArray.create(resultShape, DType.FLOAT64);
        int[] outIdx = new int[resultShape.length];
        for (long i = 0; i < result.size(); i++) {
            long r = i;
            for (int d = resultShape.length - 1; d >= 0; d--) {
                outIdx[d] = (int) (r % resultShape[d]);
                r /= resultShape[d];
            }
            int[] fullIdx = new int[shape.length];
            fi = 0;
            for (int d = 0; d < shape.length; d++) {
                if (axisSet.contains(d)) fullIdx[d] = 0;
                else fullIdx[d] = outIdx[fi++];
            }
            java.util.ArrayList<Double> values = new java.util.ArrayList<>();
            int ax = axis[0];
            for (int k = 0; k < shape[ax]; k++) {
                fullIdx[ax] = k;
                values.add(Util.readElement(sorted, fullIdx));
            }
            double idx = q * (values.size() - 1);
            long lo = (long) Math.floor(idx);
            long hi = (long) Math.ceil(idx);
            double frac = idx - lo;
            double vlo = values.get((int) lo);
            double vhi = values.get((int) hi);
            Util.writeElement(result, vlo + frac * (vhi - vlo), outIdx);
        }
        return result;
    }

    public static NDArray percentile(NDArray a, double p, int... axis) {
        return quantile(a, p / 100.0, axis);
    }

    static NDArray reduce(
            NDArray a, int[] axis, String name, double identity, java.util.function.DoubleBinaryOperator op) {
        if (axis.length == 0) {
            double result = identity;
            int[] indices = new int[a.ndim()];
            for (long i = 0; i < a.size(); i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    indices[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                result = op.applyAsDouble(result, Util.readElement(a, indices));
            }
            return NDArray.create(new double[] {result});
        }
        int[] shape = a.shape();
        java.util.HashSet<Integer> axisSet = new java.util.HashSet<>();
        for (int ax : axis) axisSet.add(ax < 0 ? ax + shape.length : ax);
        int[] finalShape = new int[shape.length - axisSet.size()];
        int fi = 0;
        for (int i = 0; i < shape.length; i++) {
            if (!axisSet.contains(i)) {
                finalShape[fi++] = shape[i];
            }
        }
        if (finalShape.length == 0) {
            finalShape = new int[] {1};
        }
        NDArray result = NDArray.create(finalShape, a.dtype());
        int[] id = new int[finalShape.length];
        for (long i = 0; i < result.size(); i++) {
            long r = i;
            for (int d = finalShape.length - 1; d >= 0; d--) {
                id[d] = (int) (r % finalShape[d]);
                r /= finalShape[d];
            }
            Util.writeElement(result, identity, id);
        }
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
            double current = Util.readElement(result, dstIdx);
            Util.writeElement(result, op.applyAsDouble(current, Util.readElement(a, srcIdx)), dstIdx);
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
                double val = Util.readElement(a, indices);
                boolean better = findMax ? val > bestVal : val < bestVal;
                if (better) {
                    bestVal = val;
                    bestIdx = i;
                }
            }
            return NDArray.create(new double[] {bestIdx});
        }
        return a;
    }

    private static NDArray cumulativeOp(
            NDArray a, int[] axis, String name, double identity, java.util.function.DoubleBinaryOperator op) {
        if (axis.length == 0) {
            int n = (int) a.size();
            NDArray result = NDArray.create(new int[] {n}, a.dtype());
            double acc = identity;
            for (int i = 0; i < n; i++) {
                acc = op.applyAsDouble(acc, Util.readBuffer(a.buffer(), a.dtype(), i));
                Util.writeBuffer(result.buffer(), result.dtype(), i, acc);
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
                    acc = op.applyAsDouble(acc, Util.readElement(a, indices));
                    Util.writeElement(result, acc, indices);
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
            Util.writeElement(result, op.applyAsDouble(Util.readElement(a, indices)), indices);
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
            Util.writeElement(
                    result, op.applyAsDouble(Util.readElement(ba, indices), Util.readElement(bb, indices)), indices);
        }
        return result;
    }
}
