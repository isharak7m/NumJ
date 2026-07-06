package jnumpy.parallel;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.IntConsumer;

public final class Parallel {

    private static final ForkJoinPool pool = ForkJoinPool.commonPool();
    private static final int THRESHOLD = 1024;

    private Parallel() {}

    public static void forEach(long start, long end, IntConsumer action) {
        if (end - start <= THRESHOLD) {
            for (long i = start; i < end; i++) action.accept((int) i);
            return;
        }
        long mid = (start + end) >>> 1;
        pool.invoke(new RecursiveAction() {
            @Override
            protected void compute() {
                var left = new ForEachAction(start, mid, action);
                var right = new ForEachAction(mid, end, action);
                invokeAll(left, right);
            }
        });
    }

    public static void parallelFor(int start, int end, IntConsumer action) {
        if (end - start <= THRESHOLD) {
            for (int i = start; i < end; i++) action.accept(i);
            return;
        }
        int mid = (start + end) >>> 1;
        var left = new ForEachAction(start, mid, action);
        var right = new ForEachAction(mid, end, action);
        left.fork();
        right.compute();
        left.join();
    }

    public static int getParallelThreshold() { return THRESHOLD; }

    public static ForkJoinPool getPool() { return pool; }

    private static final class ForEachAction extends RecursiveAction {
        private final long start;
        private final long end;
        private final IntConsumer action;

        ForEachAction(long start, long end, IntConsumer action) {
            this.start = start;
            this.end = end;
            this.action = action;
        }

        @Override
        protected void compute() {
            long length = end - start;
            if (length <= THRESHOLD) {
                for (long i = start; i < end; i++) action.accept((int) i);
            } else {
                long mid = (start + end) >>> 1;
                invokeAll(
                    new ForEachAction(start, mid, action),
                    new ForEachAction(mid, end, action)
                );
            }
        }
    }
}
