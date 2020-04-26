package tutorial

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

object LogisticRegression extends App {
  //Building the output layer
  val outputLayer : OutputLayer = new OutputLayer.Builder()
    .nIn(784) //The number of inputs feed from the input layer
    .nOut(10) //The number of output values the output layer is supposed to take
    .weightInit(WeightInit.XAVIER) //The algorithm to use for weights initialization
    .activation(Activation.SOFTMAX) //Softmax activate converts the output layer into a probability distribution
    .build() //Building our output layer

  //Since this is a simple network with a stack of layers we're going to configure a MultiLayerNetwork
  val logisticRegressionConf : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    //High Level Configuration
    .seed(123)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(new Nesterovs(0.1, 0.9))
    //For configuring MultiLayerNetwork we call the list method
    .list()
    .layer(0, outputLayer) //    <----- output layer fed here
    .build() //Building Configuration
}
