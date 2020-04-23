package tutorial

import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer._
import org.deeplearning4j.nn.weights._
import org.deeplearning4j.optimize.listeners._
import org.nd4j.evaluation.classification._
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config._
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction // mean squared error, multiclass cross entropy, etc.

object Mnist extends App {

  import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator

  val batchSize = 128 // how many examples to simultaneously train in the network
  val emnistSet = EmnistDataSetIterator.Set.BALANCED
  val emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true)
  val emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false)

  val outputNum = EmnistDataSetIterator.numLabels(emnistSet) // total output classes
  val rngSeed = 123 // integer for reproducability of a random number generator
  val numRows = 28 // number of "pixel rows" in an mnist digit
  val numColumns = 28

  val conf = new NeuralNetConfiguration.Builder()
    .seed(rngSeed)
    .updater(new Adam())
    .l2(1e-4)
    .list()
    .layer(new DenseLayer.Builder()
      .nIn(numRows * numColumns) // Number of input datapoints.
      .nOut(1000) // Number of output datapoints.
      .activation(Activation.RELU) // Activation function.
      .weightInit(WeightInit.XAVIER) // Weight initialization.
      .build()
    )
    .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
      .nIn(1000)
      .nOut(outputNum)
      .activation(Activation.SOFTMAX)
      .weightInit(WeightInit.XAVIER)
      .build()
    )
    .build()

  // create the MLN
  val network = new MultiLayerNetwork(conf)
  network.init()

  // pass a training listener that reports score every 10 iterations
  val eachIterations = 100
  network.addListeners(new ScoreIterationListener(eachIterations))

  // fit for multiple epochs
  val numEpochs = 2
  // network.fit(emnistTrain, numEpochs)

  //or simply use for loop
  for (i <- 1 to numEpochs) {
    println("Epoch " + i + " / " + numEpochs)
    network.fit(emnistTrain)
  }

  // evaluate basic performance
  val eval = network.evaluate[Evaluation](emnistTest)
  println(eval.accuracy())
  println(eval.precision())
  println(eval.recall())

  // evaluate ROC and calculate the Area Under Curve
  val roc = network.evaluateROCMultiClass[ROCMultiClass](emnistTest, 0)
  println(roc.calculateAUC(0))

  // optionally, you can print all stats from the evaluations
  print(eval.stats())
  print(roc.stats())

}
