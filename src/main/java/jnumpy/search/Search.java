package jnumpy.search;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class Search {

    private Search() {}

    public static NDArray where(NDArray condition, NDArray x, NDArray y) {
        int[] shape = jnumpy.broadcast.Broadcast.broadcastShape(condition.shape(), x.shape(), y.shape());
        NDArray bc = jnumpy.broadcast.Broadcast.broadcastTo(condition, shape);
        NDArray bx = jnumpy.broadcast.Broadcast.broadcastTo(x, shape);
        NDArray by = jnumpy.broadcast.Broadcast.broadcastTo(y, shape);
        NDArray result = NDArray.create(shape, DType.FLOAT64);
        int[] indices = new int[shape.length];
        for (long i = 0; i < result.size(); i++) {
            long remaining = i;
            for (int d = shape.length - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            result.setDouble(bc.getBoolean(indices) ? bx.getDouble(indices) : by.getDouble(indices), indices);
        }
        return result;
    }

    public static NDArray nonzero(NDArray a) {
        NDArray flat = a.ravel();
        int count = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[]{ (int) i })) count++;
        double[] result = new double[count];
        int idx = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[]{ (int) i })) result[idx++] = i;
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray argwhere(NDArray a) {
        NDArray flat = a.ravel();
        int count = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[]{ (int) i })) count++;
        double[][] coords = new double[count][a.ndim()];
        int idx = 0;
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            if (a.getBoolean(indices)) {
                for (int d = 0; d < a.ndim(); d++) coords[idx][d] = indices[d];
                idx++;
            }
        }
        double[] flatResult = new double[count * a.ndim()];
        for (int i = 0; i < count; i++)
            for (int d = 0; d < a.ndim(); d++)
                flatResult[i * a.ndim() + d] = coords[i][d];
        return new NDArray(MemoryBuffer.wrap(flatResult), new int[]{ count, a.ndim() }, DType.FLOAT64, 'C');
    }

    public static NDArray isin(NDArray element, NDArray testElements) {
        NDArray result = NDArray.create(element.shape(), DType.BOOL);
        int[] indices = new int[element.ndim()];
        for (long i = 0; i < element.size(); i++) {
            long remaining = i;
            for (int d = element.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % element.shape()[d]);
                remaining /= element.shape()[d];
            }
            double val = element.getDouble(indices);
            boolean found = false;
            for (long j = 0; j < testElements.size(); j++) {
                if (testElements.getDouble(new int[]{ (int) j }) == val) { found = true; break; }
            }
            result.setBoolean(found, indices);
        }
        return result;
    }

    public static NDArray intersect1d(NDArray a, NDArray b) {
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.ArrayList<Double> common = new java.util.ArrayList<>();
        int i = 0, j = 0;
        int nai = (int) sa.size(), nbj = (int) sb.size();
        while (i < nai && j < nbj) {
            double va = sa.getDouble(new int[]{ i });
            double vb = sb.getDouble(new int[]{ j });
            if (va < vb) i++;
            else if (va > vb) j++;
            else {
                common.add(va);
                i++; j++;
            }
        }
        double[] result = new double[common.size()];
        for (int k = 0; k < common.size(); k++) result[k] = common.get(k);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray union1d(NDArray a, NDArray b) {
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.TreeSet<Double> set = new java.util.TreeSet<>();
        int nsa = (int) sa.size();
        int nsb = (int) sb.size();
        for (int i = 0; i < nsa; i++) set.add(sa.getDouble(new int[]{ i }));
        for (int i = 0; i < nsb; i++) set.add(sb.getDouble(new int[]{ i }));
        double[] result = new double[set.size()];
        int k = 0;
        for (double v : set) result[k++] = v;
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray setdiff1d(NDArray a, NDArray b) {
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.ArrayList<Double> diff = new java.util.ArrayList<>();
        int j = 0;
        int nsa = (int) sa.size(), nsb = (int) sb.size();
        for (int i = 0; i < nsa; i++) {
            double va = sa.getDouble(new int[]{ i });
            while (j < nsb && sb.getDouble(new int[]{ j }) < va) j++;
            if (j >= nsb || sb.getDouble(new int[]{ j }) != va) diff.add(va);
        }
        double[] result = new double[diff.size()];
        for (int k = 0; k < diff.size(); k++) result[k] = diff.get(k);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray extract(NDArray condition, NDArray a) {
        NDArray flatCond = condition.ravel();
        NDArray flatA = a.ravel();
        int count = 0;
        for (long i = 0; i < flatCond.size(); i++)
            if (flatCond.getBoolean(new int[]{ (int) i })) count++;
        double[] result = new double[count];
        int idx = 0;
        int nf = (int) flatCond.size();
        for (int i = 0; i < nf; i++)
            if (flatCond.getBoolean(new int[]{ i })) result[idx++] = flatA.getDouble(new int[]{ i });
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray compress(NDArray condition, NDArray a, int axis) {
        NDArray flatCond = condition.ravel();
        int count = 0;
        for (long i = 0; i < flatCond.size(); i++)
            if (flatCond.getBoolean(new int[]{ (int) i })) count++;
        int[] newShape = a.shape().clone();
        newShape[axis] = count;
        NDArray result = NDArray.create(newShape, a.dtype());
        int[] srcIdx = new int[a.ndim()];
        int[] dstIdx = new int[a.ndim()];
        int dst = 0;
        for (int i = 0; i < a.shape(axis); i++) {
            if (flatCond.getBoolean(i)) {
                for (long j = 0; j < a.size() / a.shape(axis); j++) {
                    long remaining = j;
                    for (int d = a.ndim() - 1; d >= 0; d--) {
                        if (d == axis) continue;
                        srcIdx[d] = (int) (remaining % a.shape()[d]);
                        remaining /= a.shape()[d];
                    }
                    srcIdx[axis] = i;
                    dstIdx[axis] = dst;
                    for (int d = 0; d < a.ndim(); d++) if (d != axis) dstIdx[d] = srcIdx[d];
                    result.setDouble(a.getDouble(srcIdx), dstIdx);
                }
                dst++;
            }
        }
        return result;
    }
}
