# JNumj

A numerical computing library for the JVM, inspired by NumPy. Provides N-dimensional arrays, linear algebra operations, random number generation, FFT, and statistical functions with a NumPy-like API.

## Features

- **NDArray**: N-dimensional array with support for int8/16/32/64, uint8/16/32/64, float16/32/64, complex64/128, bool, and string dtypes
- **Universal Functions (UFunc)**: Element-wise operations (add, multiply, sin, cos, exp, log, etc.)
- **Linear Algebra**: Matrix operations, decomposition (LU, Cholesky, QR, SVD), eigenvalues, solve
- **Random**: Distributions (normal, uniform, beta, gamma, binomial, Poisson) and random sampling
- **FFT**: 1D and 2D Fast Fourier Transform with real transforms
- **Statistics**: Mean, median, variance, std, sum, prod, cumsum, cumprod, argmin, argmax
- **Manipulation**: Reshape, transpose, concatenate, stack, split, flip, squeeze, expand_dims
- **String Operations**: Upper, lower, strip, replace, contains, split, join
- **I/O**: Save/load from binary files and CSV

## Installation

### Gradle

```kotlin
implementation("io.github.isharak7m:jnumj:0.1.0")
```

### Maven

```xml
<dependency>
    <groupId>io.github.isharak7m</groupId>
    <artifactId>jnumj</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

```java
import jnumpy.Arrays;
import jnumpy.NDArray;
import static jnumpy.Arrays.*;

public class Example {
    public static void main(String[] args) {
        NDArray a = array(new double[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
        NDArray b = array(new double[]{6, 5, 4, 3, 2, 1}).reshape(2, 3);

        System.out.println("a:\n" + a);
        System.out.println("a + b:\n" + add(a, b));
        System.out.println("sin(a):\n" + sin(a));
        System.out.println("mean(a): " + mean(a));
        System.out.println("dot(a.T, a):\n" + dot(a.T(), a));

        NDArray x = linspace(0, 10, 5);
        System.out.println("linspace: " + x);
    }
}
```

## Build

```powershell
.\gradlew.bat build
```

### Requirements

- Java 21+
- Gradle (via wrapper)

## License

MIT License - see [LICENSE](LICENSE)
