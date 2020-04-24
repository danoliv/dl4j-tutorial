name := "dl4j-tutorial"

version := "0.1"

scalaVersion := "2.13.2"

val dl4jVersion = "1.0.0-beta6"

// specify the correct cuda version installed on your system
val cudaVersion = "10.2"

// CUDA : set to true to use GPU for training (CUDA) instead of CPU
val enableCuda = true

// CUDA depencencies
val cudaDependencies = Seq(
  "org.nd4j" % "nd4j-cuda-10.2-platform" % dl4jVersion,

  // Optional, but recommended: if you use CUDA, also use CuDNN. To use this, CuDNN must also be installed
  // https://deeplearning4j.konduit.ai/config/backends/config-cudnn
  "org.deeplearning4j" % "deeplearning4j-cuda-10.2" % dl4jVersion
)

// CPU dependencies, detect os to choose the correct backend configuration
// https://deeplearning4j.konduit.ai/config/backends/cpu#configuring-avx-in-nd-4-j-dl-4-j
val cpuDependencies = {
  val classifierByOs = System.getProperty("os.name").toLowerCase match {
    case mac if mac.contains("mac")  => "macosx-x86_64-avx2"
    case win if win.contains("win") => "windows-x86_64-avx2"
    case linux if linux.contains("linux") => "linux-x86_64-avx2"
    case osName => throw new RuntimeException(s"Unknown operating system $osName")
  }

  Seq(
    "org.nd4j" % "nd4j-native" % dl4jVersion,
    "org.nd4j" % "nd4j-native" % dl4jVersion classifier classifierByOs
  )
}

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
) ++ {
  if (enableCuda)
    cudaDependencies
  else
    cpuDependencies
}


