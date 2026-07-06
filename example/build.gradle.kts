plugins {
    id("application")
}

group = "com.example"
version = "1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation("io.github.isharak7m:jnumj:0.1.0")
}

application {
    mainClass = "Main"
}
