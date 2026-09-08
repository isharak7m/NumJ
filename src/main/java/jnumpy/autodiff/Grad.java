/*
 * Copyright 2026 JNumj Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jnumpy.autodiff;

public final class Grad {

    private Grad() {}

    public static double grad(java.util.function.Function<Var[], Var> f, double... xs) {
        Tape tape = new Tape();
        Var[] vars = new Var[xs.length];
        for (int i = 0; i < xs.length; i++) vars[i] = tape.var(xs[i]);
        Var y = f.apply(vars);
        double[] grads = tape.backward(tape.lastNodeId(), new double[] {1.0});
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            result[i] = grads[vars[i].id()];
        }
        return result.length == 1 ? result[0] : result[0];
    }

    public static double[] gradAll(java.util.function.Function<Var[], Var> f, double... xs) {
        Tape tape = new Tape();
        Var[] vars = new Var[xs.length];
        for (int i = 0; i < xs.length; i++) vars[i] = tape.var(xs[i]);
        Var y = f.apply(vars);
        double[] grads = tape.backward(tape.lastNodeId(), new double[] {1.0});
        double[] result = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            result[i] = grads[vars[i].id()];
        }
        return result;
    }

    public static double[] valueAndGrad(java.util.function.Function<Var[], Var> f, double... xs) {
        Tape tape = new Tape();
        Var[] vars = new Var[xs.length];
        for (int i = 0; i < xs.length; i++) vars[i] = tape.var(xs[i]);
        Var y = f.apply(vars);
        double value = y.val();
        double[] grads = tape.backward(tape.lastNodeId(), new double[] {1.0});
        double[] result = new double[xs.length + 1];
        result[0] = value;
        for (int i = 0; i < xs.length; i++) {
            result[i + 1] = grads[vars[i].id()];
        }
        return result;
    }

    public static double[] hessian(java.util.function.Function<Var[], Var> f, double... xs) {
        int n = xs.length;
        double[][] hess = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] partialGrad = new double[n];
            for (int j = 0; j < n; j++) {
                Tape tape = new Tape();
                Var[] vars = new Var[n];
                for (int k = 0; k < n; k++) vars[k] = tape.var(xs[k]);
                Var y = f.apply(vars);
                double[] grads = tape.backward(tape.lastNodeId(), new double[] {1.0});
                partialGrad[j] = grads[vars[j].id()];
            }
            for (int j = 0; j < n; j++) hess[i][j] = partialGrad[j];
        }
        double[] result = new double[n * n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) result[i * n + j] = hess[i][j];
        return result;
    }

    public static class DualForward {

        public static double eval(java.util.function.Function<Dual, Dual> f, double x) {
            Dual result = f.apply(Dual.variable(x));
            return result.dx();
        }

        public static double[] evalMulti(java.util.function.Function<Dual[], Dual> f, double... xs) {
            double[] result = new double[xs.length];
            for (int i = 0; i < xs.length; i++) {
                Dual[] vars = new Dual[xs.length];
                for (int j = 0; j < xs.length; j++) {
                    vars[j] = (j == i) ? Dual.variable(xs[j]) : Dual.constant(xs[j]);
                }
                Dual y = f.apply(vars);
                result[i] = y.dx();
            }
            return result;
        }

        public static double valueAndDeriv(java.util.function.Function<Dual, Dual> f, double x) {
            Dual result = f.apply(Dual.variable(x));
            return result.val();
        }
    }
}
