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

package jnumpy.random;

import static org.junit.jupiter.api.Assertions.*;

import jnumpy.core.Arrays;
import jnumpy.ndarray.NDArray;
import org.junit.jupiter.api.Test;

class RandomTest {

    @Test
    void testRandom() {
        NDArray a = Random.random(1000);
        assertEquals(1000, a.size());
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (int i = 0; i < 1000; i++) {
            double v = a.getDouble(i);
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        assertTrue(min >= 0.0);
        assertTrue(max <= 1.0);
    }

    @Test
    void testRandn() {
        NDArray a = Random.randn(1000);
        assertEquals(1000, a.size());
        double mean = 0;
        for (int i = 0; i < 1000; i++) mean += a.getDouble(i);
        mean /= 1000;
        assertTrue(Math.abs(mean) < 0.3);
    }

    @Test
    void testRandint() {
        NDArray a = Random.randint(0, 10, 1000);
        for (int i = 0; i < 1000; i++) {
            long v = a.getLong(i);
            assertTrue(v >= 0 && v < 10);
        }
    }

    @Test
    void testUniform() {
        NDArray a = Random.uniform(-1.0, 1.0, 100);
        for (int i = 0; i < 100; i++) {
            double v = a.getDouble(i);
            assertTrue(v >= -1.0 && v <= 1.0);
        }
    }

    @Test
    void testShuffle() {
        NDArray a = Arrays.array(new double[] {1.0, 2.0, 3.0, 4.0, 5.0});
        NDArray original = a.copy();
        Random.shuffle(a);
        boolean sameOrder = true;
        for (int i = 0; i < 5; i++) {
            if (a.getDouble(i) != original.getDouble(i)) {
                sameOrder = false;
                break;
            }
        }
        assertFalse(sameOrder);
    }

    @Test
    void testPermutation() {
        NDArray a = Random.permutation(10);
        assertEquals(10, a.size());
        double sum = 0;
        for (int i = 0; i < 10; i++) sum += a.getDouble(i);
        assertEquals(45.0, sum, 1e-12);
    }

    @Test
    void testSeed() {
        Random.seed(42);
        NDArray a = Random.randn(5);
        Random.seed(42);
        NDArray b = Random.randn(5);
        for (int i = 0; i < 5; i++) {
            assertEquals(a.getDouble(i), b.getDouble(i), 1e-15);
        }
    }
}
