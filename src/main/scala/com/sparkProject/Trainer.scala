package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    /** CHARGER LE DATASET **/

    val df: DataFrame = spark
      .read
      .option("header", value = true)  // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .parquet("data/prepared_trainingset")


    /** TF-IDF **/

    // Split 'text' into words
    val tokenizer = new RegexTokenizer()
        .setPattern("\\W+")
        .setGaps(true)
        .setInputCol("text")
        .setOutputCol("tokens")

    // Remove stop words
    val swr = new StopWordsRemover()
        .setCaseSensitive(false)
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("tokens_no_stopwords")

    // TF
    val tf = new CountVectorizer()
        .setInputCol("tokens_no_stopwords")
        .setOutputCol("tf")

    // IDF
    val idf = new IDF()
        .setInputCol("tf")
        .setOutputCol("tfidf")

    // Index 'country'
    val country_indexer = new StringIndexer()
        .setInputCol("country2")
        .setOutputCol("country_indexed")
        .setHandleInvalid("skip")

    // Index 'currency'
    val currency_indexer = new StringIndexer()
        .setInputCol("currency2")
        .setOutputCol("currency_indexed")
        .setHandleInvalid("skip")


    /** VECTOR ASSEMBLER **/

    // Assemble features
    val assembler = new VectorAssembler()
        .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
        .setOutputCol("features")


    /** MODEL **/

    // Create model
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/

    // Create pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, swr, tf, idf, country_indexer, currency_indexer, assembler, lr))


    /** TRAINING AND GRID-SEARCH **/

    // Split DataFrame into training and test sets
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))

    // Create cross-validation grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(tf.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(0.00000001, 0.000001, 0.0001, 0.01))
      .build()

    // Create evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Define cross-validator
    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Run cross-validation on training set, and choose the best set of parameters.
    val cvModel = cv.fit(training)

    // Apply model on test set
    val df_WithPredictions = cvModel.transform(test)

    // Calculate score
    val f1_score = evaluator.evaluate(df_WithPredictions)

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    println(f1_score)

    //df_WithPredictions.select("final_status", "predictions").write.mode(SaveMode.Overwrite).csv("data/output")

    // Save model
    cvModel.write.overwrite().save("data/model")

    // To load the model in the future
    val model = TrainValidationSplitModel.read.load("data/model")
  }
}
