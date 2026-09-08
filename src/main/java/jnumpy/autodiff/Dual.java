package jnumpy.autodiff;

public final class Dual {

    private final double value;
    private final double derivative;

    public Dual(double value, double derivative) {
        this.value = value;
        this.derivative = derivative;
    }

    public static Dual variable(double value) {
        return new Dual(value, 1.0);
    }

    public static Dual constant(double value) {
        return new Dual(value, 0.0);
    }

    public double val() { return value; }
    public double dx() { return derivative; }

    public Dual add(Dual other) {
        return new Dual(value + other.value, derivative + other.derivative);
    }

    public Dual sub(Dual other) {
        return new Dual(value - other.value, derivative - other.derivative);
    }

    public Dual mul(Dual other) {
        return new Dual(value * other.value, derivative * other.value + value * other.derivative);
    }

    public Dual div(Dual other) {
        return new Dual(value / other.value, (derivative * other.value - value * other.derivative) / (other.value * other.value));
    }

    public Dual neg() {
        return new Dual(-value, -derivative);
    }

    public Dual pow(double n) {
        return new Dual(Math.pow(value, n), n * Math.pow(value, n - 1) * derivative);
    }

    public Dual sin() {
        return new Dual(Math.sin(value), Math.cos(value) * derivative);
    }

    public Dual cos() {
        return new Dual(Math.cos(value), -Math.sin(value) * derivative);
    }

    public Dual tan() {
        double c = Math.cos(value);
        return new Dual(Math.tan(value), derivative / (c * c));
    }

    public Dual exp() {
        double e = Math.exp(value);
        return new Dual(e, e * derivative);
    }

    public Dual log() {
        return new Dual(Math.log(value), derivative / value);
    }

    public Dual sqrt() {
        double s = Math.sqrt(value);
        return new Dual(s, derivative / (2.0 * s));
    }

    public Dual abs() {
        return new Dual(Math.abs(value), value >= 0 ? derivative : -derivative);
    }

    @Override
    public String toString() {
        return "Dual(" + value + ", " + derivative + ")";
    }
}
