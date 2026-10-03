// app/build.gradle.kts - the chat app module.
// Single APK: native llama shell (shared libs staged via jniLibs) + Compose UI.
// git SHA provenance: configuration cache forbids external processes at configure time,
// so the build takes E0_GIT_SHA from the environment:
// unset falls back to "unknown" (never blocks the build).
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
    alias(libs.plugins.kotlinx.serialization)
}

android {
    namespace = "dev.edge0.runtime.app"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "dev.edge0.runtime.app"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.compileSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" } // arm64 only
        externalNativeBuild {
            cmake {
                abiFilters += "arm64-v8a"
                arguments += "-DLLAMA_LIBS=" + rootProject.file("build-dl/llama-libs/arm64-v8a").absolutePath
            }
        }
        // instrumented suites follow install -> stage assets -> instrument; the first step is gradle.
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "GIT_SHA",
            "\"${System.getenv("E0_GIT_SHA") ?: "unknown"}\"")
    }
    buildTypes {
        debug { isMinifyEnabled = false }
        release {
            isMinifyEnabled = false // no minification at this stage
        }
    }
    // llama native shell; .so assembly point is build-dl/llama-libs (gitignored)
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    sourceSets {
        getByName("main") {
            val ll = rootProject.file("build-dl/llama-libs/arm64-v8a")
            if (ll.isDirectory) jniLibs.srcDirs(rootProject.file("build-dl/llama-libs"))
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

ksp {
    // Room schema export into app/schemas/ for migration traceability
    arg("room.schemaLocation", "$projectDir/schemas".toString())
    arg("room.generateKotlin", "true")
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.extended)
    debugImplementation(libs.compose.ui.tooling)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.process) // lifecycle hooks for trim mapping
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.room.ktx)
    ksp(libs.androidx.room.compiler)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.markdown.renderer.m3) // markdown rendering (tables/links/strikethrough)

    testImplementation("junit:junit:4.13.2") // JVM unit tests
    androidTestImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
