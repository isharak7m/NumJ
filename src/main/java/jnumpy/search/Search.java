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

package jnumpy.search;

import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.ndarray.NDArray;
import jnumpy.util.Util;

public final class Search {

    private Search() {}

    public static NDArray where(NDArray condition, NDArray x, NDArray y) {
        int[] shape = jnumpy.broadcast.Broadcast.broadcastShape(condition.shape(), x.shape(), y.shape());
        NDArray bc = jnumpy.broadcast.Broadcast.broadcastTo(condition, shape);
        NDArray bx = jnumpy.broadcast.Broadcast.broadcastTo(x, shape);
        NDArray by = jnumpy.broadcast.Broadcast.broadcastTo(y, shape);
        DType resultDtype = x.dtype().promotedWith(y.dtype());
        NDArray result = NDArray.create(shape, resultDtype);
        int[] indices = new int[shape.length];
        for (long i = 0; i < result.size(); i++) {
            long remaining = i;
            for (int d = shape.length - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % shape[d]);
                remaining /= shape[d];
            }
            double val = bc.getBoolean(indices) ? Util.readElement(bx, indices) : Util.readElement(by, indices);
            Util.writeElement(result, val, indices);
        }
        return result;
    }

    public static NDArray nonzero(NDArray a) {
        if (a.size() == 0) return NDArray.create(new int[] {0}, DType.INT64);
        NDArray flat = a.ravel();
        int count = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[] {(int) i})) count++;
        long[] result = new long[count];
        int idx = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[] {(int) i})) result[idx++] = i;
        return new NDArray(MemoryBuffer.wrap(result), DType.INT64);
    }

    public static NDArray argwhere(NDArray a) {
        if (a.size() == 0) return NDArray.create(new int[] {0, a.ndim()}, DType.INT64);
        NDArray flat = a.ravel();
        int count = 0;
        for (long i = 0; i < flat.size(); i++) if (flat.getBoolean(new int[] {(int) i})) count++;
        long[] flatResult = new long[count * a.ndim()];
        int idx = 0;
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            if (a.getBoolean(indices)) {
                for (int d = 0; d < a.ndim(); d++) flatResult[idx * a.ndim() + d] = indices[d];
                idx++;
            }
        }
        return new NDArray(MemoryBuffer.wrap(flatResult), new int[] {count, a.ndim()}, DType.INT64, 'C');
    }

    public static NDArray isin(NDArray element, NDArray testElements) {
        NDArray result = NDArray.create(element.shape(), DType.BOOL);
        java.util.HashSet<Long> testSet = new java.util.HashSet<>();
        for (long j = 0; j < testElements.size(); j++) {
            testSet.add(Double.doubleToLongBits(Util.readElement(testElements, (int) j)));
        }
        int[] indices = new int[element.ndim()];
        for (long i = 0; i < element.size(); i++) {
            long remaining = i;
            for (int d = element.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % element.shape()[d]);
                remaining /= element.shape()[d];
            }
            double val = Util.readElement(element, indices);
            result.setBoolean(testSet.contains(Double.doubleToLongBits(val)), indices);
        }
        return result;
    }

    public static NDArray intersect1d(NDArray a, NDArray b) {
        if (a.size() == 0 || b.size() == 0) return NDArray.create(new int[] {0}, DType.FLOAT64);
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.ArrayList<Double> common = new java.util.ArrayList<>();
        int i = 0, j = 0;
        int nai = (int) sa.size(), nbj = (int) sb.size();
        while (i < nai && j < nbj) {
            double va = Util.readElement(sa, i);
            double vb = Util.readElement(sb, j);
            if (va < vb) i++;
            else if (va > vb) j++;
            else {
                common.add(va);
                double last = va;
                while (i < nai && Util.readElement(sa, i) == last) i++;
                while (j < nbj && Util.readElement(sb, j) == last) j++;
            }
        }
        double[] result = new double[common.size()];
        for (int k = 0; k < common.size(); k++) result[k] = common.get(k);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray union1d(NDArray a, NDArray b) {
        if (a.size() == 0 && b.size() == 0) return NDArray.create(new int[] {0}, DType.FLOAT64);
        if (a.size() == 0) return jnumpy.sort.Sort.sort(b).ravel();
        if (b.size() == 0) return jnumpy.sort.Sort.sort(a).ravel();
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.ArrayList<Double> merged = new java.util.ArrayList<>();
        int i = 0, j = 0;
        int nsa = (int) sa.size(), nsb = (int) sb.size();
        while (i < nsa || j < nsb) {
            double va = i < nsa ? Util.readElement(sa, i) : Double.POSITIVE_INFINITY;
            double vb = j < nsb ? Util.readElement(sb, j) : Double.POSITIVE_INFINITY;
            if (va < vb) {
                if (merged.isEmpty() || merged.get(merged.size() - 1) != va) merged.add(va);
                i++;
            } else if (vb < va) {
                if (merged.isEmpty() || merged.get(merged.size() - 1) != vb) merged.add(vb);
                j++;
            } else {
                if (merged.isEmpty() || merged.get(merged.size() - 1) != va) merged.add(va);
                i++;
                j++;
            }
        }
        double[] result = new double[merged.size()];
        for (int k = 0; k < merged.size(); k++) result[k] = merged.get(k);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray setdiff1d(NDArray a, NDArray b) {
        if (a.size() == 0) return NDArray.create(new int[] {0}, DType.FLOAT64);
        if (b.size() == 0) return jnumpy.sort.Sort.sort(a).ravel();
        NDArray sa = jnumpy.sort.Sort.sort(a);
        NDArray sb = jnumpy.sort.Sort.sort(b);
        java.util.ArrayList<Double> diff = new java.util.ArrayList<>();
        int j = 0;
        int nsa = (int) sa.size(), nsb = (int) sb.size();
        for (int i = 0; i < nsa; i++) {
            double va = Util.readElement(sa, i);
            if (i > 0 && Util.readElement(sa, i - 1) == va) continue;
            while (j < nsb && Util.readElement(sb, j) < va) j++;
            if (j >= nsb || Util.readElement(sb, j) != va) diff.add(va);
        }
        double[] result = new double[diff.size()];
        for (int k = 0; k < diff.size(); k++) result[k] = diff.get(k);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray extract(NDArray condition, NDArray a) {
        NDArray flatCond = condition.ravel();
        NDArray flatA = a.ravel();
        int count = 0;
        for (long i = 0; i < flatCond.size(); i++) if (flatCond.getBoolean(new int[] {(int) i})) count++;
        double[] result = new double[count];
        int idx = 0;
        int nf = (int) flatCond.size();
        for (int i = 0; i < nf; i++) if (flatCond.getBoolean(new int[] {i})) result[idx++] = Util.readElement(flatA, i);
        return new NDArray(MemoryBuffer.wrap(result), DType.FLOAT64);
    }

    public static NDArray compress(NDArray condition, NDArray a, int axis) {
        NDArray flatCond = condition.ravel();
        int count = 0;
        for (long i = 0; i < flatCond.size(); i++) if (flatCond.getBoolean(new int[] {(int) i})) count++;
        int[] newShape = a.shape().clone();
        newShape[axis] = count;
        NDArray result = NDArray.create(newShape, a.dtype());
        int[] srcIdx = new int[a.ndim()];
        int[] dstIdx = new int[a.ndim()];
        int dst = 0;
        int outerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= a.shape(d);
        int innerSize = 1;
        for (int d = axis + 1; d < a.ndim(); d++) innerSize *= a.shape(d);
        for (int o = 0; o < outerSize; o++) {
            int or = o;
            for (int d = axis - 1; d >= 0; d--) {
                srcIdx[d] = or % a.shape(d);
                or /= a.shape(d);
            }
            dst = 0;
            for (int i = 0; i < a.shape(axis); i++) {
                if (flatCond.getBoolean(i)) {
                    for (int inner = 0; inner < innerSize; inner++) {
                        int ir = inner;
                        for (int d = a.ndim() - 1; d > axis; d--) {
                            srcIdx[d] = ir % a.shape(d);
                            ir /= a.shape(d);
                        }
                        srcIdx[axis] = i;
                        for (int d = 0; d < a.ndim(); d++) dstIdx[d] = srcIdx[d];
                        dstIdx[axis] = dst;
                        Util.writeElement(result, Util.readElement(a, srcIdx), dstIdx);
                    }
                    dst++;
                }
            }
        }
        return result;
    }
}
