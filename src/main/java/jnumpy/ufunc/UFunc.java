package jnumpy.ufunc;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.broadcast.Broadcast;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.DoublePredicate;

public final class UFunc {

    private UFunc() {}

    public static NDArray add(NDArray a, NDArray b) { return binaryOp(a, b, "add", Double::sum); }
    public static NDArray sub(NDArray a, NDArray b) { return binaryOp(a, b, "sub", (x, y) -> x - y); }
    public static NDArray mul(NDArray a, NDArray b) { return binaryOp(a, b, "mul", (x, y) -> x * y); }
    public static NDArray div(NDArray a, NDArray b) { return binaryOp(a, b, "div", (x, y) -> x / y); }
    public static NDArray pow(NDArray a, NDArray b) { return binaryOp(a, b, "pow", Math::pow); }
    public static NDArray mod(NDArray a, NDArray b) { return binaryOp(a, b, "mod", (x, y) -> x % y); }

    public static NDArray neg(NDArray a) { return unaryOp(a, "neg", x -> -x); }
    public static NDArray abs(NDArray a) { return unaryOp(a, "abs", Math::abs); }
    public static NDArray sqrt(NDArray a) { return unaryOp(a, "sqrt", Math::sqrt); }
    public static NDArray sin(NDArray a) { return unaryOp(a, "sin", Math::sin); }
    public static NDArray cos(NDArray a) { return unaryOp(a, "cos", Math::cos); }
    public static NDArray tan(NDArray a) { return unaryOp(a, "tan", Math::tan); }
    public static NDArray asin(NDArray a) { return unaryOp(a, "asin", Math::asin); }
    public static NDArray acos(NDArray a) { return unaryOp(a, "acos", Math::acos); }
    public static NDArray atan(NDArray a) { return unaryOp(a, "atan", Math::atan); }
    public static NDArray sinh(NDArray a) { return unaryOp(a, "sinh", Math::sinh); }
    public static NDArray cosh(NDArray a) { return unaryOp(a, "cosh", Math::cosh); }
    public static NDArray tanh(NDArray a) { return unaryOp(a, "tanh", Math::tanh); }
    public static NDArray exp(NDArray a) { return unaryOp(a, "exp", Math::exp); }
    public static NDArray log(NDArray a) { return unaryOp(a, "log", Math::log); }
    public static NDArray log10(NDArray a) { return unaryOp(a, "log10", Math::log10); }
    public static NDArray log2(NDArray a) { return unaryOp(a, "log2", Math::log10); } // log2 via log10
    public static NDArray ceil(NDArray a) { return unaryOp(a, "ceil", Math::ceil); }
    public static NDArray floor(NDArray a) { return unaryOp(a, "floor", Math::floor); }
    public static NDArray round(NDArray a) { return unaryOp(a, "round", Math::rint); }
    public static NDArray sign(NDArray a) { return unaryOp(a, "sign", x -> x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0); }
    public static NDArray reciprocal(NDArray a) { return unaryOp(a, "reciprocal", x -> 1.0 / x); }

    public static NDArray maximum(NDArray a, NDArray b) { return binaryOp(a, b, "maximum", Math::max); }
    public static NDArray minimum(NDArray a, NDArray b) { return binaryOp(a, b, "minimum", Math::min); }
    public static NDArray clip(NDArray a, double min, double max) {
        return unaryOp(a, "clip", x -> Math.max(min, Math.min(max, x)));
    }

    public static NDArray logicalAnd(NDArray a, NDArray b) { return binaryOp(a, b, "logical_and", (x, y) -> (x != 0 && y != 0) ? 1.0 : 0.0); }
    public static NDArray logicalOr(NDArray a, NDArray b) { return binaryOp(a, b, "logical_or", (x, y) -> (x != 0 || y != 0) ? 1.0 : 0.0); }
    public static NDArray logicalXor(NDArray a, NDArray b) { return binaryOp(a, b, "logical_xor", (x, y) -> ((x != 0) != (y != 0)) ? 1.0 : 0.0); }
    public static NDArray logicalNot(NDArray a) { return unaryOp(a, "logical_not", x -> (x == 0) ? 1.0 : 0.0); }

    public static NDArray greater(NDArray a, NDArray b) { return binaryOp(a, b, "greater", (x, y) -> x > y ? 1.0 : 0.0); }
    public static NDArray greaterEqual(NDArray a, NDArray b) { return binaryOp(a, b, "greater_equal", (x, y) -> x >= y ? 1.0 : 0.0); }
    public static NDArray less(NDArray a, NDArray b) { return binaryOp(a, b, "less", (x, y) -> x < y ? 1.0 : 0.0); }
    public static NDArray lessEqual(NDArray a, NDArray b) { return binaryOp(a, b, "less_equal", (x, y) -> x <= y ? 1.0 : 0.0); }
    public static NDArray equal(NDArray a, NDArray b) { return binaryOp(a, b, "equal", (x, y) -> x == y ? 1.0 : 0.0); }
    public static NDArray notEqual(NDArray a, NDArray b) { return binaryOp(a, b, "not_equal", (x, y) -> x != y ? 1.0 : 0.0); }

    public static NDArray bitwiseAnd(NDArray a, NDArray b) { return binaryOp(a, b, "bitwise_and", (x, y) -> (double) ((long) x & (long) y)); }
    public static NDArray bitwiseOr(NDArray a, NDArray b) { return binaryOp(a, b, "bitwise_or", (x, y) -> (double) ((long) x | (long) y)); }
    public static NDArray bitwiseXor(NDArray a, NDArray b) { return binaryOp(a, b, "bitwise_xor", (x, y) -> (double) ((long) x ^ (long) y)); }
    public static NDArray bitwiseNot(NDArray a) { return unaryOp(a, "bitwise_not", x -> (double) ~((long) x)); }
    public static NDArray leftShift(NDArray a, NDArray b) { return binaryOp(a, b, "left_shift", (x, y) -> (double) (((long) x) << ((long) y))); }
    public static NDArray rightShift(NDArray a, NDArray b) { return binaryOp(a, b, "right_shift", (x, y) -> (double) (((long) x) >> ((long) y))); }

    public static NDArray isNan(NDArray a) { return unaryOp(a, "isnan", x -> Double.isNaN(x) ? 1.0 : 0.0); }
    public static NDArray isInf(NDArray a) { return unaryOp(a, "isinf", x -> Double.isInfinite(x) ? 1.0 : 0.0); }
    public static NDArray isFinite(NDArray a) { return unaryOp(a, "isfinite", x -> Double.isFinite(x) ? 1.0 : 0.0); }

    private static NDArray unaryOp(NDArray a, String name, DoubleUnaryOperator op) {
        NDArray result = NDArray.create(a.shape(), a.dtype());
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                int[] shape = a.shape();
                indices[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            double val = a.getDouble(indices);
            result.setDouble(op.applyAsDouble(val), indices);
        }
        return result;
    }

    private static NDArray binaryOp(NDArray a, NDArray b, String name, DoubleBinaryOperator op) {
        int[] shape = Broadcast.broadcastShape(a.shape(), b.shape());
        NDArray broadcastA = Broadcast.broadcastTo(a, shape);
        NDArray broadcastB = Broadcast.broadcastTo(b, shape);
        DType resultDtype = a.dtype().promotedWith(b.dtype());
        NDArray result = NDArray.create(shape, resultDtype);
        Broadcast.BroadcastIterator it = Broadcast.iterator(a, b);
        it.forEach((flat, aOff, bOff) -> {
            double va = a.buffer().getDouble(aOff);
            double vb = b.buffer().getDouble(bOff);
            long resultOff = result.offset();
            long remaining = flat;
            for (int d = result.ndim() - 1; d >= 0; d--) {
                int coord = (int) (remaining % shape[d]);
                remaining /= shape[d];
                resultOff += (long) coord * result.strides()[d];
            }
            result.buffer().setDouble(resultOff, op.applyAsDouble(va, vb));
        });
        return result;
    }
}
