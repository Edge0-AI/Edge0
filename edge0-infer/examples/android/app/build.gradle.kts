@file:Suppress("DEPRECATION")

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}
val appId = providers.environmentVariable("MOE_ANDROID_EXAMPLE_APP_ID").orElse("dev.edge0.example.android")
android {
    namespace = "dev.edge0.chat"
    compileSdk = 37
    compileSdkMinor = 2
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = appId.orNull
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.2.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    buildFeatures { compose = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("sharedAssets").get().asFile)
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
val copyShared = tasks.register<Copy>("copyShared") {
    from(rootProject.file("../shared/app.json"))
    into(layout.buildDirectory.dir("sharedAssets"))
}
tasks.named("preBuild") {
    dependsOn(copyShared)
}
dependencies {
    implementation(project(":core"))
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.material3)
    implementation(libs.activity.compose)
    implementation(libs.viewmodel.compose)
    implementation(libs.markdown)
}
