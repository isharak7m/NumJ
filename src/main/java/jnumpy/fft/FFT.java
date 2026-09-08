package jnumpy.fft;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import jnumpy.util.Util;

public final class FFT {

    private FFT() {}

    public static NDArray fft(NDArray a) {
        int n = (int) a.size();
        double[] re = new double[n];
        double[] im = new double[n];
        int[] idx = new int[1];
        for (int i = 0; i < n; i++) {
            idx[0] = i;
            re[i] = Util.readElement(a, idx);
        }
        fftCooleyTukey(re, im, false);
        NDArray result = NDArray.create(new int[]{ n, 2 }, DType.FLOAT64);
        for (int i = 0; i < n; i++) {
            result.setDouble(re[i], i, 0);
            result.setDouble(im[i], i, 1);
        }
        return result;
    }

    public static NDArray ifft(NDArray a) {
        int n = a.ndim() == 1 ? (int) a.size() : a.shape(0);
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) {
            if (a.ndim() == 2) {
                re[i] = Util.readElement(a, i, 0);
                im[i] = Util.readElement(a, i, 1);
            } else {
                re[i] = Util.readElement(a, i);
            }
        }
        fftCooleyTukey(re, im, true);
        double invN = 1.0 / n;
        NDArray result = NDArray.create(new int[]{ n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) result.setDouble(re[i] * invN, i);
        return result;
    }

    public static NDArray fft2(NDArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NDArray result = NDArray.create(new int[]{ rows, cols, 2 }, DType.FLOAT64);
        for (int i = 0; i < rows; i++) {
            NDArray row = jnumpy.indexing.Indexer.get(a, i);
            NDArray fftRow = fft(row);
            for (int j = 0; j < cols; j++) {
                result.setDouble(fftRow.getDouble(j, 0), i, j, 0);
                result.setDouble(fftRow.getDouble(j, 1), i, j, 1);
            }
        }
        return result;
    }

    public static NDArray ifft2(NDArray a) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        NDArray result = NDArray.create(new int[]{ rows, cols }, DType.FLOAT64);
        for (int i = 0; i < rows; i++) {
            NDArray slice = jnumpy.indexing.Indexer.get(a, i);
            double[] re = new double[cols];
            double[] im = new double[cols];
            for (int j = 0; j < cols; j++) {
                re[j] = slice.getDouble(j, 0);
                im[j] = slice.getDouble(j, 1);
            }
            fftCooleyTukey(re, im, true);
            double invN = 1.0 / cols;
            for (int j = 0; j < cols; j++) result.setDouble(re[j] * invN, i, j);
        }
        return result;
    }

    public static NDArray rfft(NDArray a) {
        NDArray full = fft(a);
        int n = (int) a.size();
        int m = n / 2 + 1;
        NDArray result = NDArray.create(new int[]{ m, 2 }, DType.FLOAT64);
        for (int i = 0; i < m; i++) {
            result.setDouble(full.getDouble(i, 0), i, 0);
            result.setDouble(full.getDouble(i, 1), i, 1);
        }
        return result;
    }

    public static NDArray irfft(NDArray a) {
        int m = a.shape(0);
        int n = 2 * (m - 1);
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < m; i++) {
            re[i] = a.getDouble(i, 0);
            im[i] = a.getDouble(i, 1);
        }
        for (int i = m; i < n; i++) {
            re[i] = re[n - i];
            im[i] = -im[n - i];
        }
        fftCooleyTukey(re, im, true);
        double invN = 1.0 / n;
        NDArray result = NDArray.create(new int[]{ n }, DType.FLOAT64);
        for (int i = 0; i < n; i++) result.setDouble(re[i] * invN, i);
        return result;
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
