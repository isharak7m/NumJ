module jnumpy {
    requires jdk.incubator.vector;
    exports jnumpy.core;
    exports jnumpy.ndarray;
    exports jnumpy.dtype;
    exports jnumpy.memory;
    exports jnumpy.broadcast;
    exports jnumpy.indexing;
    exports jnumpy.ufunc;
    exports jnumpy.creation;
    exports jnumpy.manipulation;
    exports jnumpy.linalg;
    exports jnumpy.random;
    exports jnumpy.statistics;
    exports jnumpy.fft;
    exports jnumpy.polynomial;
    exports jnumpy.sort;
    exports jnumpy.search;
    exports jnumpy.io;
    exports jnumpy.string;
    exports jnumpy.parallel;
    exports jnumpy.simd;
    exports jnumpy.util;
}
