plugins {
    id("java-library")
    id("maven-publish")
    id("signing")
    id("jacoco")
    id("me.champeau.jmh") version "0.7.2"
    id("com.diffplug.spotless") version "6.25.0"
}

group = "io.github.isharak7m"
version = properties["VERSION"] as? String ?: "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
    withJavadocJar()
    withSourcesJar()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.11.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.junit.jupiter:junit-jupiter-params")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "-Xlint:all",
        "-Xlint:-processing",
        "-Xlint:-serial",
        "-Xlint:-path",
        "-Xlint:-missing-explicit-ctor",
        "--enable-preview"
    ))
    options.encoding = "UTF-8"
    options.isFork = true
}

tasks.withType<JavaExec> {
    jvmArgs("--enable-preview")
}

tasks.withType<Test> {
    useJUnitPlatform()
    maxHeapSize = "4g"
    jvmArgs("--enable-preview", "--add-modules=jdk.incubator.vector")
}

tasks.withType<Javadoc> {
    options.encoding = "UTF-8"
    options.memberLevel = JavadocMemberLevel.PUBLIC
    (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    (options as StandardJavadocDocletOptions).addBooleanOption("-enable-preview", true)
    (options as StandardJavadocDocletOptions).addStringOption("-source", "21")
    (options as StandardJavadocDocletOptions).addStringOption("-add-modules", "jdk.incubator.vector")
    isFailOnError = false
}

jacoco {
    toolVersion = "0.8.12"
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}

spotless {
    java {
        palantirJavaFormat()
        formatAnnotations()
        licenseHeader("/*\n * Copyright \$YEAR JNumj Contributors\n *\n * Licensed under the Apache License, Version 2.0 (the \"License\");\n * you may not use this file except in compliance with the License.\n * You may obtain a copy of the License at\n *\n *     http://www.apache.org/licenses/LICENSE-2.0\n *\n * Unless required by applicable law or agreed to in writing, software\n * distributed under the License is distributed on an \"AS IS\" BASIS,\n * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n * See the License for the specific language governing permissions and\n * limitations under the License.\n */\n\n")
    }
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                name.set("JNumj")
                description.set("Numerical computing library for the JVM, inspired by NumPy")
                url.set("https://github.com/isharak7m/NumJ")
                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
                developers {
                    developer {
                        id.set("isharak7m")
                        name.set("Isharak7m")
                        email.set("isharak7m@users.noreply.github.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/isharak7m/NumJ.git")
                    developerConnection.set("scm:git:ssh://github.com/isharak7m/NumJ.git")
                    url.set("https://github.com/isharak7m/NumJ")
                }
            }
        }
    }
}

signing {
    sign(publishing.publications["mavenJava"])
}

jmh {
    warmupIterations.set(3)
    iterations.set(5)
    fork.set(2)
    benchmarkMode.set(listOf("thrpt", "avgt"))
    timeUnit.set("ms")
    resultFormat.set("JSON")
    jvmArgs.set(listOf("--add-modules=jdk.incubator.vector"))
}
