package jnumpy.fft;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;

public final class FFT {

    private FFT() {}

    public static NDArray fft(NDArray a) {
        int n = (int) a.size();
        double[] re = new double[n];
        double[] im = new double[n];
        int[] idx = new int[1];
        for (int i = 0; i < n; i++) {
            idx[0] = i;
            re[i] = a.getDouble(idx);
        }
        fftCooleyTukey(re, im, false);
        double[] result = new double[n * 2];
        for (int i = 0; i < n; i++) {
            result[2 * i] = re[i];
            result[2 * i + 1] = im[i];
        }
        MemoryBuffer buf = MemoryBuffer.allocate(DType.COMPLEX64, n);
        for (int i = 0; i < n; i++) {
            buf.setFloat(2L * i, (float) re[i]);
            buf.setFloat(2L * i + 1, (float) im[i]);
        }
        return new NDArray(buf, new int[]{ n }, DType.COMPLEX64, 'C');
    }

    public static NDArray ifft(NDArray a) {
        int n = (int) a.size();
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) {
            re[i] = a.getFloat(new int[]{ i, 0 });
            im[i] = a.getFloat(new int[]{ i, 1 });
        }
        fftCooleyTukey(re, im, true);
        double invN = 1.0 / n;
        for (int i = 0; i < n; i++) { re[i] *= invN; im[i] *= invN; }
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, n);
        for (int i = 0; i < n; i++) buf.setDouble(i, re[i]);
        return new NDArray(buf, DType.FLOAT64);
    }

    public static NDArray fft2(NDArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.COMPLEX64);
        for (int i = 0; i < rows; i++) {
            NDArray row = jnumpy.indexing.Indexer.get(a, i);
            NDArray fftRow = fft(row);
            for (int j = 0; j < cols; j++) {
                result.setFloat(fftRow.getFloat(new int[]{ j, 0 }), i, j, 0);
                result.setFloat(fftRow.getFloat(new int[]{ j, 1 }), i, j, 1);
            }
        }
        return result;
    }

    public static NDArray ifft2(NDArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.FLOAT64);
        for (int i = 0; i < rows; i++) {
            NDArray row = jnumpy.indexing.Indexer.get(a, i);
            NDArray ifftRow = ifft(row);
            for (int j = 0; j < cols; j++) result.setDouble(ifftRow.getDouble(new int[]{ j }), i, j);
        }
        return result;
    }

    public static NDArray rfft(NDArray a) {
        NDArray full = fft(a);
        int n = (int) a.size();
        int m = n / 2 + 1;
        MemoryBuffer buf = MemoryBuffer.allocate(DType.COMPLEX64, m);
        for (int i = 0; i < m; i++) {
            buf.setFloat(2L * i, full.getFloat(i, 0));
            buf.setFloat(2L * i + 1, full.getFloat(i, 1));
        }
        return new NDArray(buf, new int[]{ m }, DType.COMPLEX64, 'C');
    }

    public static NDArray irfft(NDArray a) {
        int m = (int) a.size();
        int n = 2 * (m - 1);
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < m; i++) {
            re[i] = a.getFloat(i, 0);
            im[i] = a.getFloat(i, 1);
        }
        for (int i = m; i < n; i++) {
            re[i] = re[n - i];
            im[i] = -im[n - i];
        }
        fftCooleyTukey(re, im, true);
        double invN = 1.0 / n;
        MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, n);
        for (int i = 0; i < n; i++) buf.setDouble(i, re[i] * invN);
        return new NDArray(buf, DType.FLOAT64);
    }

    private static void fftCooleyTukey(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        if (n <= 1) return;
        int bits = Integer.SIZE - Integer.numberOfLeadingZeros(n - 1);
        int[] rev = new int[n];
        for (int i = 0; i < n; i++) {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1));
        }
        for (int i = 0; i < n; i++) {
            if (i < rev[i]) {
                double tr = re[i]; double ti = im[i];
                re[i] = re[rev[i]]; im[i] = im[rev[i]];
                re[rev[i]] = tr; im[rev[i]] = ti;
            }
        }
        double sign = inverse ? 2 * Math.PI : -2 * Math.PI;
        for (int len = 2; len <= n; len <<= 1) {
            double ang = sign / len;
            double wRe = Math.cos(ang);
            double wIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double curRe = 1, curIm = 0;
                int halfLen = len >> 1;
                for (int j = 0; j < halfLen; j++) {
                    int ia = i + j;
                    int ib = i + j + halfLen;
                    double tRe = curRe * re[ib] - curIm * im[ib];
                    double tIm = curRe * im[ib] + curIm * re[ib];
                    re[ib] = re[ia] - tRe; im[ib] = im[ia] - tIm;
                    re[ia] += tRe; im[ia] += tIm;
                    double nRe = curRe * wRe - curIm * wIm;
                    double nIm = curRe * wIm + curIm * wRe;
                    curRe = nRe; curIm = nIm;
                }
            }
        }
    }
}
