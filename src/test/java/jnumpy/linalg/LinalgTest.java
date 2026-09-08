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

package jnumpy.linalg;

import static org.junit.jupiter.api.Assertions.*;

import jnumpy.core.Arrays;
import jnumpy.ndarray.NDArray;
import org.junit.jupiter.api.Test;

class LinalgTest {

    @Test
    void testDot() {
        NDArray a = Arrays.array(new double[] {1.0, 2.0, 3.0});
        NDArray b = Arrays.array(new double[] {4.0, 5.0, 6.0});
        NDArray c = Linalg.dot(a, b);
        assertEquals(32.0, c.getDouble(0), 1e-12);
    }

    @Test
    void testMatmul() {
        NDArray a = Arrays.array(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        NDArray b = Arrays.array(new double[][] {{5.0, 6.0}, {7.0, 8.0}});
        NDArray c = Linalg.matmul(a, b);
        assertEquals(19.0, c.getDouble(0, 0), 1e-12);
        assertEquals(50.0, c.getDouble(1, 1), 1e-12);
    }

    @Test
    void testOuter() {
        NDArray a = Arrays.array(new double[] {1.0, 2.0});
        NDArray b = Arrays.array(new double[] {3.0, 4.0, 5.0});
        NDArray c = Linalg.outer(a, b);
        assertEquals(3.0, c.getDouble(0, 0), 1e-12);
        assertEquals(4.0, c.getDouble(0, 1), 1e-12);
        assertEquals(6.0, c.getDouble(1, 0), 1e-12);
        assertEquals(10.0, c.getDouble(1, 2), 1e-12);
    }

    @Test
    void testNorm() {
        NDArray a = Arrays.array(new double[] {3.0, 4.0});
        assertEquals(5.0, Linalg.norm(a), 1e-12);
    }

    @Test
    void testDet() {
        NDArray a = Arrays.array(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        assertEquals(-2.0, Linalg.det(a), 1e-12);
    }

    @Test
    void testInv() {
        NDArray a = Arrays.array(new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        NDArray invA = Linalg.inv(a);
        NDArray identity = Linalg.matmul(a, invA);
        assertEquals(1.0, identity.getDouble(0, 0), 1e-12);
        assertEquals(0.0, identity.getDouble(0, 1), 1e-12);
        assertEquals(0.0, identity.getDouble(1, 0), 1e-12);
        assertEquals(1.0, identity.getDouble(1, 1), 1e-12);
    }

    @Test
    void testCross() {
        NDArray a = Arrays.array(new double[] {1.0, 0.0, 0.0});
        NDArray b = Arrays.array(new double[] {0.0, 1.0, 0.0});
        NDArray c = Linalg.cross(a, b);
        assertEquals(0.0, c.getDouble(0), 1e-12);
        assertEquals(0.0, c.getDouble(1), 1e-12);
        assertEquals(1.0, c.getDouble(2), 1e-12);
    }
}
