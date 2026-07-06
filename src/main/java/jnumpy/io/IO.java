package jnumpy.io;

import jnumpy.ndarray.NDArray;
import jnumpy.dtype.DType;
import jnumpy.memory.MemoryBuffer;
import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;

public final class IO {

    private IO() {}

    public static void tofile(NDArray a, String filename) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename))) {
            int[] indices = new int[a.ndim()];
            for (long i = 0; i < a.size(); i++) {
                long remaining = i;
                for (int d = a.ndim() - 1; d >= 0; d--) {
                    indices[d] = (int) (remaining % a.shape()[d]);
                    remaining /= a.shape()[d];
                }
                dos.writeDouble(a.getDouble(indices));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write file: " + filename, e);
        }
    }

    public static NDArray fromfile(String filename) {
        try {
            byte[] data = Files.readAllBytes(Paths.get(filename));
            int n = data.length / 8;
            MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, n);
            for (int i = 0; i < n; i++) {
                long bits = 0;
                for (int j = 0; j < 8; j++) bits = (bits << 8) | (data[i * 8 + j] & 0xFF);
                buf.setDouble(i, Double.longBitsToDouble(bits));
            }
            return new NDArray(buf, DType.FLOAT64);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read file: " + filename, e);
        }
    }

    public static void tocsv(NDArray a, String filename) {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filename))) {
            if (a.ndim() == 1) {
                for (int i = 0; i < a.shape(0); i++) {
                    writer.write(Double.toString(a.getDouble(i)));
                    writer.newLine();
                }
            } else if (a.ndim() == 2) {
                for (int i = 0; i < a.shape(0); i++) {
                    for (int j = 0; j < a.shape(1); j++) {
                        if (j > 0) writer.write(",");
                        writer.write(Double.toString(a.getDouble(i, j)));
                    }
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write CSV: " + filename, e);
        }
    }

    public static NDArray fromcsv(String filename) {
        try {
            java.util.List<double[]> rows = new ArrayList<>();
            int cols = -1;
            for (String line : Files.readAllLines(Paths.get(filename))) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                if (cols < 0) cols = parts.length;
                double[] row = new double[cols];
                for (int j = 0; j < cols; j++) row[j] = Double.parseDouble(parts[j].trim());
                rows.add(row);
            }
            int nrows = rows.size();
            MemoryBuffer buf = MemoryBuffer.allocate(DType.FLOAT64, (long) nrows * cols);
            for (int i = 0; i < nrows; i++)
                for (int j = 0; j < cols; j++)
                    buf.setDouble((long) i * cols + j, rows.get(i)[j]);
            return new NDArray(buf, new int[]{ nrows, cols }, DType.FLOAT64, 'C');
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV: " + filename, e);
        }
    }
}
