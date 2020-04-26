package tutorial

import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration}
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object ComputationGraph extends App {

  val computationGraphConf : ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
    .seed(123)
    .updater(new Nesterovs(0.1, 0.9)) //High Level Configuration
    .graphBuilder()  //For configuring ComputationGraph we call the graphBuilder method
    .addInputs("input") //Configuring Layers
    .addLayer("L1",
      new DenseLayer
        .Builder()
        .nIn(3)
        .nOut(4)
        .build(), "input"
    )
    .addLayer("out1",
      new OutputLayer
        .Builder()
        .lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(4)
        .nOut(3)
        .build(),
      "L1"
    )
    .addLayer("out2",
      new OutputLayer
        .Builder()
        .lossFunction(LossFunction.MSE)
        .nIn(4)
        .nOut(2)
        .build(), "L1"
    )
    .setOutputs("out1","out2")
    .build() //Building configuration

  println(computationGraphConf)
}
