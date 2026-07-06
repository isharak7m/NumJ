package jnumpy.core;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.creation.Creation;
import jnumpy.manipulation.Manipulation;
import jnumpy.ufunc.UFunc;
import jnumpy.statistics.Statistics;
import jnumpy.linalg.Linalg;
import jnumpy.random.Random;
import jnumpy.sort.Sort;
import jnumpy.search.Search;
import jnumpy.fft.FFT;
import jnumpy.polynomial.Polynomial;
import jnumpy.io.IO;
import jnumpy.string.StringOps;
import jnumpy.util.Util;

public final class Arrays {

    private Arrays() {}

    public static NDArray array(Object data) { return Creation.array(data); }
    public static NDArray zeros(int... shape) { return Creation.zeros(shape); }
    public static NDArray zeros(DType dtype, int... shape) { return Creation.zeros(dtype, shape); }
    public static NDArray ones(int... shape) { return Creation.ones(shape); }
    public static NDArray ones(DType dtype, int... shape) { return Creation.ones(dtype, shape); }
    public static NDArray empty(int... shape) { return Creation.empty(shape); }
    public static NDArray full(int[] shape, double fillValue) { return Creation.full(shape, fillValue, DType.FLOAT64); }
    public static NDArray full(int[] shape, double fillValue, DType dtype) { return Creation.full(shape, fillValue, dtype); }
    public static NDArray eye(int n) { return Creation.eye(n); }
    public static NDArray eye(int rows, int cols) { return Creation.eye(rows, cols); }
    public static NDArray identity(int n) { return Creation.identity(n); }
    public static NDArray diag(double[] diagonal) { return Creation.diag(diagonal); }
    public static NDArray diag(NDArray v) { return Creation.diag(v); }
    public static NDArray diagflat(NDArray a) { return Creation.diagFlat(a); }
    public static NDArray arange(double stop) { return Creation.arange(stop); }
    public static NDArray arange(double start, double stop) { return Creation.arange(start, stop); }
    public static NDArray arange(double start, double stop, double step) { return Creation.arange(start, stop, step); }
    public static NDArray linspace(double start, double stop, int num) { return Creation.linspace(start, stop, num); }
    public static NDArray logspace(double start, double stop, int num) { return Creation.logspace(start, stop, num); }
    public static NDArray geomspace(double start, double stop, int num) { return Creation.geomspace(start, stop, num); }
    public static NDArray meshgrid(NDArray x, NDArray y) { return Creation.meshgrid(x, y); }
    public static NDArray fromFunction(java.util.function.IntToDoubleFunction f, int... shape) { return Creation.fromFunction(f, shape); }
    public static NDArray copy(NDArray a) { return Creation.copy(a); }

    public static NDArray reshape(NDArray a, int... newShape) { return a.reshape(newShape); }
    public static NDArray ravel(NDArray a) { return a.ravel(); }
    public static NDArray flatten(NDArray a) { return a.flatten(); }
    public static NDArray transpose(NDArray a, int... axes) { return a.transpose(axes); }
    public static NDArray swapAxes(NDArray a, int axis1, int axis2) { return a.swapAxes(axis1, axis2); }
    public static NDArray moveAxis(NDArray a, int source, int destination) { return a.moveAxis(source, destination); }
    public static NDArray squeeze(NDArray a, int... axes) { return a.squeeze(axes); }
    public static NDArray expandDims(NDArray a, int axis) { return a.expandDims(axis); }
    public static NDArray repeat(NDArray a, int repeats, int axis) { return a.repeat(repeats, axis); }
    public static NDArray tile(NDArray a, int... reps) { return a.tile(reps); }
    public static NDArray flip(NDArray a, int... axes) { return a.flip(axes); }
    public static NDArray rot90(NDArray a, int k, int... axes) { return a.rot90(k, axes); }
    public static NDArray concatenate(NDArray[] arrays, int axis) { return Manipulation.concatenate(arrays, axis); }
    public static NDArray stack(NDArray[] arrays, int axis) { return Manipulation.stack(arrays, axis); }
    public static NDArray hstack(NDArray[] arrays) { return Manipulation.hstack(arrays); }
    public static NDArray vstack(NDArray[] arrays) { return Manipulation.vstack(arrays); }
    public static NDArray split(NDArray a, int sections, int axis) { return Manipulation.split(a, sections, axis); }
    public static NDArray columnStack(NDArray[] arrays) { return Manipulation.columnStack(arrays); }

    public static NDArray add(NDArray a, NDArray b) { return UFunc.add(a, b); }
    public static NDArray subtract(NDArray a, NDArray b) { return UFunc.sub(a, b); }
    public static NDArray multiply(NDArray a, NDArray b) { return UFunc.mul(a, b); }
    public static NDArray divide(NDArray a, NDArray b) { return UFunc.div(a, b); }
    public static NDArray power(NDArray a, NDArray b) { return UFunc.pow(a, b); }
    public static NDArray mod(NDArray a, NDArray b) { return UFunc.mod(a, b); }
    public static NDArray negative(NDArray a) { return UFunc.neg(a); }
    public static NDArray abs(NDArray a) { return UFunc.abs(a); }
    public static NDArray sqrt(NDArray a) { return UFunc.sqrt(a); }
    public static NDArray sin(NDArray a) { return UFunc.sin(a); }
    public static NDArray cos(NDArray a) { return UFunc.cos(a); }
    public static NDArray tan(NDArray a) { return UFunc.tan(a); }
    public static NDArray arcsin(NDArray a) { return UFunc.asin(a); }
    public static NDArray arccos(NDArray a) { return UFunc.acos(a); }
    public static NDArray arctan(NDArray a) { return UFunc.atan(a); }
    public static NDArray sinh(NDArray a) { return UFunc.sinh(a); }
    public static NDArray cosh(NDArray a) { return UFunc.cosh(a); }
    public static NDArray tanh(NDArray a) { return UFunc.tanh(a); }
    public static NDArray exp(NDArray a) { return UFunc.exp(a); }
    public static NDArray log(NDArray a) { return UFunc.log(a); }
    public static NDArray log10(NDArray a) { return UFunc.log10(a); }
    public static NDArray log2(NDArray a) { return UFunc.log2(a); }
    public static NDArray ceil(NDArray a) { return UFunc.ceil(a); }
    public static NDArray floor(NDArray a) { return UFunc.floor(a); }
    public static NDArray round(NDArray a) { return UFunc.round(a); }
    public static NDArray sign(NDArray a) { return UFunc.sign(a); }
    public static NDArray reciprocal(NDArray a) { return UFunc.reciprocal(a); }
    public static NDArray maximum(NDArray a, NDArray b) { return UFunc.maximum(a, b); }
    public static NDArray minimum(NDArray a, NDArray b) { return UFunc.minimum(a, b); }
    public static NDArray clip(NDArray a, double min, double max) { return UFunc.clip(a, min, max); }

    public static NDArray logicalAnd(NDArray a, NDArray b) { return UFunc.logicalAnd(a, b); }
    public static NDArray logicalOr(NDArray a, NDArray b) { return UFunc.logicalOr(a, b); }
    public static NDArray logicalXor(NDArray a, NDArray b) { return UFunc.logicalXor(a, b); }
    public static NDArray logicalNot(NDArray a) { return UFunc.logicalNot(a); }

    public static NDArray greater(NDArray a, NDArray b) { return UFunc.greater(a, b); }
    public static NDArray greaterEqual(NDArray a, NDArray b) { return UFunc.greaterEqual(a, b); }
    public static NDArray less(NDArray a, NDArray b) { return UFunc.less(a, b); }
    public static NDArray lessEqual(NDArray a, NDArray b) { return UFunc.lessEqual(a, b); }
    public static NDArray equal(NDArray a, NDArray b) { return UFunc.equal(a, b); }
    public static NDArray notEqual(NDArray a, NDArray b) { return UFunc.notEqual(a, b); }

    public static NDArray bitwiseAnd(NDArray a, NDArray b) { return UFunc.bitwiseAnd(a, b); }
    public static NDArray bitwiseOr(NDArray a, NDArray b) { return UFunc.bitwiseOr(a, b); }
    public static NDArray bitwiseXor(NDArray a, NDArray b) { return UFunc.bitwiseXor(a, b); }
    public static NDArray bitwiseNot(NDArray a) { return UFunc.bitwiseNot(a); }
    public static NDArray leftShift(NDArray a, NDArray b) { return UFunc.leftShift(a, b); }
    public static NDArray rightShift(NDArray a, NDArray b) { return UFunc.rightShift(a, b); }

    public static NDArray isnan(NDArray a) { return UFunc.isNan(a); }
    public static NDArray isinf(NDArray a) { return UFunc.isInf(a); }
    public static NDArray isfinite(NDArray a) { return UFunc.isFinite(a); }

    public static NDArray sum(NDArray a, int... axis) { return Statistics.sum(a, axis); }
    public static NDArray prod(NDArray a, int... axis) { return Statistics.prod(a, axis); }
    public static NDArray mean(NDArray a, int... axis) { return Statistics.mean(a, axis); }
    public static NDArray var(NDArray a, int... axis) { return Statistics.var(a, axis); }
    public static NDArray std(NDArray a, int... axis) { return Statistics.std(a, axis); }
    public static NDArray min(NDArray a, int... axis) { return Statistics.min(a, axis); }
    public static NDArray max(NDArray a, int... axis) { return Statistics.max(a, axis); }
    public static NDArray argmin(NDArray a, int... axis) { return Statistics.argmin(a, axis); }
    public static NDArray argmax(NDArray a, int... axis) { return Statistics.argmax(a, axis); }
    public static NDArray cumsum(NDArray a, int... axis) { return Statistics.cumsum(a, axis); }
    public static NDArray cumprod(NDArray a, int... axis) { return Statistics.cumprod(a, axis); }
    public static NDArray countNonzero(NDArray a, int... axis) { return Statistics.countNonzero(a, axis); }
    public static NDArray quantile(NDArray a, double q, int... axis) { return Statistics.quantile(a, q, axis); }
    public static NDArray percentile(NDArray a, double p, int... axis) { return Statistics.percentile(a, p, axis); }

    public static NDArray dot(NDArray a, NDArray b) { return Linalg.dot(a, b); }
    public static NDArray matmul(NDArray a, NDArray b) { return Linalg.matmul(a, b); }
    public static NDArray inner(NDArray a, NDArray b) { return Linalg.inner(a, b); }
    public static NDArray outer(NDArray a, NDArray b) { return Linalg.outer(a, b); }
    public static NDArray cross(NDArray a, NDArray b) { return Linalg.cross(a, b); }
    public static NDArray kron(NDArray a, NDArray b) { return Linalg.kron(a, b); }
    public static double trace(NDArray a) { return Linalg.trace(a); }
    public static double norm(NDArray a) { return Linalg.norm(a); }
    public static NDArray inv(NDArray a) { return Linalg.inv(a); }
    public static double det(NDArray a) { return Linalg.det(a); }
    public static int rank(NDArray a) { return Linalg.rank(a); }
    public static NDArray qr(NDArray a) { return Linalg.qr(a); }
    public static NDArray cholesky(NDArray a) { return Linalg.cholesky(a); }
    public static NDArray svd(NDArray a) { return Linalg.svd(a); }
    public static NDArray eigen(NDArray a) { return Linalg.eigen(a); }

    public static NDArray rand(int... shape) { return Random.random(shape); }
    public static NDArray randn(int... shape) { return Random.randn(shape); }
    public static NDArray randint(int low, int high, int... shape) { return Random.randint(low, high, shape); }
    public static NDArray uniform(double low, double high, int... shape) { return Random.uniform(low, high, shape); }
    public static NDArray normal(double mean, double std, int... shape) { return Random.normal(mean, std, shape); }
    public static NDArray poisson(double lam, int... shape) { return Random.poisson(lam, shape); }
    public static NDArray exponential(double scale, int... shape) { return Random.exponential(scale, shape); }
    public static NDArray gamma(double shape, double scale, int... outShape) { return Random.gamma(shape, scale, outShape); }
    public static NDArray beta(double a, double b, int... shape) { return Random.beta(a, b, shape); }
    public static NDArray binomial(int n, double p, int... shape) { return Random.binomial(n, p, shape); }
    public static NDArray choice(NDArray a, int size, boolean replace) { return Random.choice(a, size, replace); }
    public static NDArray shuffle(NDArray a) { return Random.shuffle(a); }
    public static NDArray permutation(int n) { return Random.permutation(n); }
    public static void seed(long s) { Random.seed(s); }

    public static NDArray sort(NDArray a, int... axis) { return Sort.sort(a, axis); }
    public static NDArray argsort(NDArray a, int axis) { return Sort.argsort(a, axis); }
    public static NDArray unique(NDArray a) { return Sort.unique(a); }
    public static NDArray searchsorted(NDArray a, double v) { return Sort.searchsorted(a, v); }
    public static NDArray partition(NDArray a, int kth, int axis) { return Sort.partition(a, kth, axis); }

    public static NDArray where(NDArray condition, NDArray x, NDArray y) { return Search.where(condition, x, y); }
    public static NDArray nonzero(NDArray a) { return Search.nonzero(a); }
    public static NDArray argwhere(NDArray a) { return Search.argwhere(a); }
    public static NDArray isin(NDArray element, NDArray testElements) { return Search.isin(element, testElements); }
    public static NDArray intersect1d(NDArray a, NDArray b) { return Search.intersect1d(a, b); }
    public static NDArray union1d(NDArray a, NDArray b) { return Search.union1d(a, b); }
    public static NDArray setdiff1d(NDArray a, NDArray b) { return Search.setdiff1d(a, b); }
    public static NDArray extract(NDArray condition, NDArray a) { return Search.extract(condition, a); }
    public static NDArray compress(NDArray condition, NDArray a, int axis) { return Search.compress(condition, a, axis); }

    public static NDArray fft(NDArray a) { return FFT.fft(a); }
    public static NDArray ifft(NDArray a) { return FFT.ifft(a); }
    public static NDArray fft2(NDArray a) { return FFT.fft2(a); }
    public static NDArray ifft2(NDArray a) { return FFT.ifft2(a); }
    public static NDArray rfft(NDArray a) { return FFT.rfft(a); }
    public static NDArray irfft(NDArray a) { return FFT.irfft(a); }

    public static NDArray polyfit(NDArray x, NDArray y, int deg) { return Polynomial.polyfit(x, y, deg); }
    public static NDArray polyval(NDArray p, NDArray x) { return Polynomial.polyval(p, x); }
    public static NDArray roots(NDArray p) { return Polynomial.roots(p); }
    public static NDArray polyadd(NDArray a, NDArray b) { return Polynomial.polyadd(a, b); }
    public static NDArray polymul(NDArray a, NDArray b) { return Polynomial.polymul(a, b); }
    public static NDArray polydiv(NDArray a, NDArray b) { return Polynomial.polydiv(a, b); }

    public static NDArray fromfile(String filename) { return IO.fromfile(filename); }
    public static void tofile(NDArray a, String filename) { IO.tofile(a, filename); }
    public static NDArray fromcsv(String filename) { return IO.fromcsv(filename); }
    public static void tocsv(NDArray a, String filename) { IO.tocsv(a, filename); }

    public static NDArray charUpper(NDArray a) { return StringOps.upper(a); }
    public static NDArray charLower(NDArray a) { return StringOps.lower(a); }
    public static NDArray charStrip(NDArray a) { return StringOps.strip(a); }
    public static NDArray charReplace(NDArray a, String old, String newStr) { return StringOps.replace(a, old, newStr); }

    public static double[] toDoubleArray(NDArray a) { return Util.toDoubleArray(a); }
    public static String toString(NDArray a) { return a.toString(); }
    public static NDArray ascontiguousarray(NDArray a) { return Util.asContiguous(a); }

    public static final DType BOOL = DType.BOOL;
    public static final DType INT8 = DType.INT8;
    public static final DType INT16 = DType.INT16;
    public static final DType INT32 = DType.INT32;
    public static final DType INT64 = DType.INT64;
    public static final DType UINT8 = DType.UINT8;
    public static final DType UINT16 = DType.UINT16;
    public static final DType UINT32 = DType.UINT32;
    public static final DType UINT64 = DType.UINT64;
    public static final DType FLOAT16 = DType.FLOAT16;
    public static final DType FLOAT32 = DType.FLOAT32;
    public static final DType FLOAT64 = DType.FLOAT64;
    public static final DType COMPLEX64 = DType.COMPLEX64;
    public static final DType COMPLEX128 = DType.COMPLEX128;
    public static final DType CHAR = DType.CHAR;
    public static final DType STRING = DType.STRING;
    public static final DType OBJECT = DType.OBJECT;

    public static final double E = Math.E;
    public static final double PI = Math.PI;
    public static final double INF = Double.POSITIVE_INFINITY;
    public static final double NEG_INF = Double.NEGATIVE_INFINITY;
    public static final double NAN = Double.NaN;
}
