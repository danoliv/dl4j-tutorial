package tutorial

import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object MultiLayerNetwork extends App {

  val multiLayerConf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .seed(123)
    .updater(new Nesterovs(0.1, 0.9)) //High Level Configuration
    .list() //For configuring MultiLayerNetwork we call the list method
    .layer(0,
      new DenseLayer
        .Builder()
        .nIn(784)
        .nOut(100)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build()
    ) //Configuring Layers
    .layer(1,
      new OutputLayer
        .Builder(LossFunction.XENT)
        .nIn(100).nOut(10)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SIGMOID)
        .build()
    )
    .build() //Building Configuration


  println(multiLayerConf.toJson)

}
