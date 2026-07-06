package jnumpy.string;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class StringOps {

    private StringOps() {}

    public static NDArray upper(NDArray a) {
        NDArray result = NDArray.create(a.shape(), DType.STRING);
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result.setString(a.getString(indices).toUpperCase(), indices);
        }
        return result;
    }

    public static NDArray lower(NDArray a) {
        NDArray result = NDArray.create(a.shape(), DType.STRING);
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result.setString(a.getString(indices).toLowerCase(), indices);
        }
        return result;
    }

    public static NDArray strip(NDArray a) {
        NDArray result = NDArray.create(a.shape(), DType.STRING);
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result.setString(a.getString(indices).strip(), indices);
        }
        return result;
    }

    public static NDArray replace(NDArray a, String oldStr, String newStr) {
        NDArray result = NDArray.create(a.shape(), DType.STRING);
        int[] indices = new int[a.ndim()];
        for (long i = 0; i < a.size(); i++) {
            long remaining = i;
            for (int d = a.ndim() - 1; d >= 0; d--) {
                indices[d] = (int) (remaining % a.shape()[d]);
                remaining /= a.shape()[d];
            }
            result.setString(a.getString(indices).replace(oldStr, newStr), indices);
        }
        return result;
    }
}
