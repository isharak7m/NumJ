package jnumpy.autodiff;

public final class Var {

    private final int id;
    private final Tape tape;

    Var(int id, Tape tape) {
        this.id = id;
        this.tape = tape;
    }

    int id() { return id; }
    Tape tape() { return tape; }

    public double val() {
        return tape.getValue(id);
    }

    public Var add(Var other) {
        double result = val() + other.val();
        int[] inputs = { id, other.id };
        tape.addNode(new Tape.Node(new double[]{ result }, "add", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var sub(Var other) {
        double result = val() - other.val();
        int[] inputs = { id, other.id };
        tape.addNode(new Tape.Node(new double[]{ result }, "sub", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var mul(Var other) {
        double result = val() * other.val();
        int[] inputs = { id, other.id };
        tape.addNode(new Tape.Node(new double[]{ result }, "mul", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var div(Var other) {
        double result = val() / other.val();
        int[] inputs = { id, other.id };
        tape.addNode(new Tape.Node(new double[]{ result }, "div", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var neg() {
        double result = -val();
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "neg", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var pow(double n) {
        double result = Math.pow(val(), n);
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "pow", inputs, new double[]{ n }));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var sin() {
        double result = Math.sin(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "sin", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var cos() {
        double result = Math.cos(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "cos", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var tan() {
        double result = Math.tan(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "tan", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var exp() {
        double result = Math.exp(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "exp", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var log() {
        double result = Math.log(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "log", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var sqrt() {
        double result = Math.sqrt(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "sqrt", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }

    public Var abs() {
        double result = Math.abs(val());
        int[] inputs = { id };
        tape.addNode(new Tape.Node(new double[]{ result }, "abs", inputs, null));
        return new Var(tape.lastNodeId(), tape);
    }
}
