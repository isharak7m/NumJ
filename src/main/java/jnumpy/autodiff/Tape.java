package jnumpy.autodiff;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.function.DoubleBinaryOperator;

public final class Tape {

    public static final class Node {
        final double[] value;
        final String op;
        final int[] inputs;
        final double[] extra;
        double[] grad;

        Node(double[] value, String op, int[] inputs, double[] extra) {
            this.value = value;
            this.op = op;
            this.inputs = inputs;
            this.extra = extra;
            this.grad = new double[value.length];
        }
    }

    private final List<Node> nodes = new ArrayList<>();
    private final List<double[]> constants = new ArrayList<>();

    public Var var(double value) {
        double[] buf = new double[]{ value };
        nodes.add(new Node(buf, "const", new int[0], null));
        return new Var(nodes.size() - 1, this);
    }

    public Var const_(double value) {
        double[] buf = new double[]{ value };
        constants.add(buf);
        return new Var(-(constants.size()), this);
    }

    double getValue(int id) {
        if (id >= 0) return nodes.get(id).value[0];
        return constants.get(-id - 1)[0];
    }

    void addNode(Node node) {
        nodes.add(node);
    }

    int lastNodeId() {
        return nodes.size() - 1;
    }

    public double[] backward(int targetId, double[] targetGrad) {
        Map<Integer, double[]> grads = new HashMap<>();
        grads.put(targetId, targetGrad);

        for (int i = nodes.size() - 1; i >= 0; i--) {
            Node node = nodes.get(i);
            double[] g = grads.get(i);
            if (g == null) continue;

            switch (node.op) {
                case "add" -> {
                    accumulate(grads, node.inputs[0], g[0]);
                    accumulate(grads, node.inputs[1], g[0]);
                }
                case "sub" -> {
                    accumulate(grads, node.inputs[0], g[0]);
                    accumulate(grads, node.inputs[1], -g[0]);
                }
                case "mul" -> {
                    double a = getValue(node.inputs[0]);
                    double b = getValue(node.inputs[1]);
                    accumulate(grads, node.inputs[0], g[0] * b);
                    accumulate(grads, node.inputs[1], g[0] * a);
                }
                case "div" -> {
                    double num = getValue(node.inputs[0]);
                    double den = getValue(node.inputs[1]);
                    accumulate(grads, node.inputs[0], g[0] / den);
                    accumulate(grads, node.inputs[1], -g[0] * num / (den * den));
                }
                case "neg" -> accumulate(grads, node.inputs[0], -g[0]);
                case "pow" -> {
                    double base = getValue(node.inputs[0]);
                    double n = node.extra[0];
                    accumulate(grads, node.inputs[0], g[0] * n * Math.pow(base, n - 1));
                }
                case "sin" -> {
                    double x = getValue(node.inputs[0]);
                    accumulate(grads, node.inputs[0], g[0] * Math.cos(x));
                }
                case "cos" -> {
                    double x = getValue(node.inputs[0]);
                    accumulate(grads, node.inputs[0], -g[0] * Math.sin(x));
                }
                case "tan" -> {
                    double x = getValue(node.inputs[0]);
                    double c = Math.cos(x);
                    accumulate(grads, node.inputs[0], g[0] / (c * c));
                }
                case "exp" -> {
                    double v = node.value[0];
                    accumulate(grads, node.inputs[0], g[0] * v);
                }
                case "log" -> {
                    double x = getValue(node.inputs[0]);
                    accumulate(grads, node.inputs[0], g[0] / x);
                }
                case "sqrt" -> {
                    double v = node.value[0];
                    accumulate(grads, node.inputs[0], g[0] / (2.0 * v));
                }
                case "abs" -> {
                    double x = getValue(node.inputs[0]);
                    accumulate(grads, node.inputs[0], g[0] * (x >= 0 ? 1.0 : -1.0));
                }
            }
        }

        double[] result = new double[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            double[] g = grads.get(i);
            if (g != null) result[i] = g[0];
        }
        return result;
    }

    private void accumulate(Map<Integer, double[]> grads, int id, double val) {
        if (id < 0) return;
        grads.merge(id, new double[]{ val }, (a, b) -> new double[]{ a[0] + b[0] });
    }

    public void clear() {
        nodes.clear();
        constants.clear();
    }
}
