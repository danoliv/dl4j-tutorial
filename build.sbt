name := "dl4j-tutorial"

version := "0.1"

scalaVersion := "2.13.1"

val dl4jVersion = "1.0.0-beta6"

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.nd4j" % "nd4j-native" % dl4jVersion

  // CUDA: to use GPU for training (CUDA) instead of CPU, uncomment this, and remove nd4j-native-platform
  //   Requires CUDA to be installed to use. Change the version (8.0, 9.0, 9.1) to change the CUDA version
  // "org.nd4j" % "nd4j-cuda-9.2-platform" % dl4jVersion,

  // Optional, but recommended: if you use CUDA, also use CuDNN. To use this, CuDNN must also be installed
  // "org.deeplearning4j" % "deeplearning4j-cuda-9.2" % dl4jVersion
)


