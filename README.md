# JNumj

A numerical computing library for the JVM, inspired by NumPy. Provides N-dimensional arrays, linear algebra, random number generation, FFT, statistical functions, and string operations with a NumPy-like API.

**Published on Maven Central** — `io.github.isharak7m:jnumj:0.1.0`

---

## Installation

### Gradle

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.isharak7m:jnumj:0.1.0")
}
```

### Maven

```xml
<dependency>
    <groupId>io.github.isharak7m</groupId>
    <artifactId>jnumj</artifactId>
    <version>0.1.0</version>
</dependency>
```

---

## Quick Start

```java
import jnumpy.Arrays;
import jnumpy.NDArray;
import static jnumpy.Arrays.*;

public class Example {
    public static void main(String[] args) {
        // Create arrays
        NDArray a = array(new double[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
        NDArray b = array(new double[]{6, 5, 4, 3, 2, 1}).reshape(2, 3);

        // Arithmetic
        System.out.println("a + b:\n" + add(a, b));
        System.out.println("a * b:\n" + multiply(a, b));

        // Universal functions
        System.out.println("sin(a):\n" + sin(a));
        System.out.println("exp(a):\n" + exp(a));

        // Statistics
        System.out.println("mean(a): " + mean(a));
        System.out.println("std(a): " + std(a));

        // Linear algebra
        System.out.println("dot(a.T, a):\n" + dot(a.T(), a));

        // Array creation
        NDArray x = linspace(0, 10, 5);
        System.out.println("linspace(0, 10, 5): " + x);
        NDArray r = arange(0, 10, 2);
        System.out.println("arange(0, 10, 2): " + r);
        NDArray e = eye(3);
        System.out.println("eye(3):\n" + e);
        NDArray o = ones(new int[]{2, 3});
        System.out.println("ones(2,3):\n" + o);
    }
}
```

---

## Features

### NDArray
N-dimensional array with support for:

| DType | Description |
|-------|-------------|
| `INT8`, `INT16`, `INT32`, `INT64` | Signed integers |
| `UINT8`, `UINT16`, `UINT32`, `UINT64` | Unsigned integers |
| `FLOAT16`, `FLOAT32`, `FLOAT64` | Floating point |
| `COMPLEX64`, `COMPLEX128` | Complex numbers |
| `BOOL` | Boolean |
| `STRING` | String |

### Array Creation
`array`, `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `diag`, `diagflat`, `arange`, `linspace`, `logspace`, `geomspace`, `copy`, `fromFunction`, `ascontiguousarray`

### Universal Functions
**Arithmetic**: `add`, `subtract`, `multiply`, `divide`, `mod`, `pow`, `negate`, `abs`

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`

**Exponential**: `exp`, `log`, `log10`, `log2`, `log1p`, `sqrt`, `cbrt`

**Rounding**: `ceil`, `floor`, `round`, `trunc`, `rint`

**Comparison**: `equal`, `notEqual`, `greater`, `greaterEqual`, `less`, `lessEqual`

**Bitwise**: `bitwiseAnd`, `bitwiseOr`, `bitwiseXor`, `bitwiseNot`

**Others**: `clip`, `where`, `isnan`, `isinf`, `isfinite`

### Linear Algebra
`dot`, `inner`, `outer`, `cross`, `matmul`, `inv`, `det`, `solve`, `cholesky`, `lu`, `qr`, `svd`, `eigen`, `eigvals`, `norm`, `trace`

### Statistics
`min`, `max`, `mean`, `median`, `var`, `std`, `sum`, `prod`, `cumsum`, `cumprod`, `argmin`, `argmax`, `countNonzero`, `percentile`

### Random
`uniform`, `normal`, `beta`, `gamma`, `binomial`, `poisson`, `exponential`, `choice`, `randint`, `shuffle`, `permutation`

### FFT
`fft`, `ifft`, `fft2`, `ifft2`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`

### Array Manipulation
`reshape`, `transpose`, `flatten`, `ravel`, `squeeze`, `expandDims`, `repeat`, `tile`, `flip`, `rot90`, `concatenate`, `hstack`, `vstack`, `columnStack`, `split`, `hsplit`, `vsplit`

### Search & Sort
`argwhere`, `where`, `nonzero`, `extract`, `compress`, `intersect1d`, `union1d`, `setdiff1d`, `isin`, `sort`, `argsort`, `searchsorted`

### String Operations
`upper`, `lower`, `strip`, `replace`, `split`, `join`, `contains`, `startswith`, `endswith`

### I/O
`fromfile`, `tofile`, `fromcsv`, `tocsv`

---

## SIMD Acceleration

JNumj uses `jdk.incubator.vector` for SIMD-accelerated operations on supported hardware, providing significant performance improvements for element-wise operations on large arrays.

---

## Building from Source

### Requirements
- Java 21+
- Gradle (via wrapper)

### Commands

```powershell
# Build
.\gradlew.bat build

# Run tests
.\gradlew.bat test

# Run benchmarks
.\gradlew.bat jmh

# Generate coverage report
.\gradlew.bat jacocoTestReport
```

---

## License

MIT License — see [LICENSE](LICENSE)
