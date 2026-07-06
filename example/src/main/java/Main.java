import jnumpy.core.Arrays;
import jnumpy.ndarray.NDArray;
import static jnumpy.core.Arrays.*;

public class Main {
    public static void main(String[] args) {
        System.out.println("=== JNumj Demo ===");
        System.out.println();

        NDArray a = array(new double[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
        NDArray b = array(new double[]{6, 5, 4, 3, 2, 1}).reshape(2, 3);

        System.out.println("a =\n" + a);
        System.out.println("b =\n" + b);
        System.out.println("a + b =\n" + add(a, b));
        System.out.println("a * b =\n" + multiply(a, b));
        System.out.println("mean(a) = " + mean(a));
        System.out.println("sum(a) = " + sum(a));
        System.out.println("sin(a) =\n" + sin(a));
        System.out.println("a.T =\n" + a.T());
        System.out.println("dot(a.T, a) =\n" + dot(a.T(), a));
        System.out.println();

        NDArray eye = eye(4);
        System.out.println("eye(4) =\n" + eye);

        NDArray lin = linspace(0, 1, 5);
        System.out.println("linspace(0, 1, 5) = " + lin);

        NDArray rand = normal(0, 1, new int[]{2, 4});
        System.out.println("random normal(2x4) =\n" + rand);
    }
}
