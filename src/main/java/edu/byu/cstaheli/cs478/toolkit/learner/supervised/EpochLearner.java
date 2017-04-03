package edu.byu.cstaheli.cs478.toolkit.learner.supervised;

import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Created by cstaheli on 3/5/2017.
 */
public abstract class EpochLearner extends RandomLearner
{
    private static final boolean OUTPUT_EACH_EPOCH = false;

    private int totalEpochs;
    private double learningRate;

    public EpochLearner(Random random)
    {
        super(random);
        totalEpochs = 0;
        learningRate = .1;
    }

    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        initializeWeights(trainingFeatures.cols(), trainingLabels.valueCount(0));
        //Get a baseline accuracy
        double trainingMSE = calcMeanSquaredError(trainingFeatures, trainingLabels);
        double validationMSE = calcMeanSquaredError(strategy.getValidationFeatures(), strategy.getValidationLabels());
        double validationAccuracy = measureAccuracy(strategy.getValidationFeatures(), strategy.getValidationLabels(), new Matrix());
        double bestAccuracy = validationAccuracy;
        completeEpoch(0, trainingMSE, validationMSE, validationAccuracy);
        boolean keepTraining = true;
        //for each epoch
        while (keepTraining)
        {
            //for each training data instance
            trainingFeatures = strategy.getTrainingFeatures();
            trainingLabels = strategy.getTrainingLabels();
            for (int i = 0; i < trainingFeatures.rows(); ++i)
            {
                analyzeInputRow(trainingFeatures.row(i), trainingLabels.get(i, 0));
                //propagate error through the network
                //adjust the weights
            }
            //calculate the accuracy over training data
            trainingMSE = calcMeanSquaredError(trainingFeatures, trainingLabels);
            //for each validation data instance
            //calculate the accuracy over the validation data
            validationMSE = calcMeanSquaredError(strategy.getValidationFeatures(), strategy.getValidationLabels());
            validationAccuracy = measureAccuracy(strategy.getValidationFeatures(), strategy.getValidationLabels(), new Matrix());
            //if the threshold validation accuracy is met, stop training, else continue
            keepTraining = !isThresholdValidationAccuracyMet(validationAccuracy, bestAccuracy);
            bestAccuracy = getBestAccuracy(validationAccuracy, bestAccuracy);
            incrementTotalEpochs();
            completeEpoch(getTotalEpochs(), trainingMSE, validationMSE, validationAccuracy);
        }
        double testMSE = calcMeanSquaredError(strategy.getTestingFeatures(), strategy.getTestingLabels());
        double testAccuracy = measureAccuracy(strategy.getTestingFeatures(), strategy.getTestingLabels(), new Matrix());
        outputFinalAccuracies(getTotalEpochs(), trainingMSE, validationMSE, testMSE, validationAccuracy, testAccuracy);
    }

    @Override
    public abstract void predict(double[] features, double[] labels) throws Exception;

    protected abstract void initializeWeights(int features, int outputs);

    protected abstract void analyzeInputRow(double[] row, double expectedOutput);

    protected abstract boolean isThresholdValidationAccuracyMet(double validationAccuracy, double bestAccuracy);

    protected void outputFinalAccuracies(int epoch, double trainingMSE, double validationMSE, double testMSE, double validationClassificationAccuracy, double testClassificationAccuracy)
    {
        if (getOutputFile() != null)
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                writer.append(String.format("%s, %s, %s, %s, %s, %s\n", epoch, trainingMSE, validationMSE, testMSE, validationClassificationAccuracy, testClassificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    public int getTotalEpochs()
    {
        return totalEpochs;
    }

    protected void completeEpoch(int epoch, double classificationAccuracy)
    {
        if (shouldOutput() && OUTPUT_EACH_EPOCH)
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                writer.append(String.format("%s, %s\n", epoch, classificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    protected void completeEpoch(int epoch, double trainingMSE, double validationMSE, double classificationAccuracy)
    {
        if (shouldOutput() && OUTPUT_EACH_EPOCH)
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                writer.append(String.format("%s, %s, %s, %s\n", epoch, trainingMSE, validationMSE, classificationAccuracy));
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }

    public void incrementTotalEpochs()
    {
        ++this.totalEpochs;
    }

}
