plugins { alias(libs.plugins.kotlin.jvm) }
kotlin {
    jvmToolchain(25)
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}
java { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
dependencies { testImplementation(libs.junit) }
tasks.test { systemProperty("shared.dir", rootProject.file("../shared").absolutePath) }
