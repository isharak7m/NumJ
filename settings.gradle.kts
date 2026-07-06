rootProject.name = "jnumj"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("com.gradleup.nmcp.settings") version "1.6.1"
}

nmcpSettings {
    centralPortal {
        username = providers.gradleProperty("sonatypeUser").orNull ?: ""
        password = providers.gradleProperty("sonatypePassword").orNull ?: ""
        publishingType = "AUTOMATIC"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
    }
}
