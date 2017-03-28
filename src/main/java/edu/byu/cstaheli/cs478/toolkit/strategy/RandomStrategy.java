package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.learner.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.Random;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class RandomStrategy extends LearningStrategy
{
    private double trainPercent;
    private Random rand;
    private boolean shouldShuffle;
    private Matrix trainingData;

    public RandomStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
        rand = learnerData.getRandom();
        System.out.println("Calculating accuracy on a random hold-out set...");
        trainPercent = Double.parseDouble(learnerData.getEvalParameter());
        if (getTrainPercent() < 0 || getTrainPercent() > 1)
        {
            throw new Exception("Percentage for random evaluation must be between 0 and 1");
        }
        System.out.println("Percentage used for training: " + getTrainPercent());
        System.out.println("Percentage used for testing: " + (1 - getTrainPercent()));
        getArffData().shuffle(rand);
        shouldShuffle = true;
        trainingData = getInitialTrainingData();
    }

    private Matrix getInitialTrainingData()
    {
        if (isUsingValidationSet())
        {
            return new Matrix(getArffData(), 0, 0, getTrainingSetSize(), getArffData().cols());
        }
        else
        {
            return new Matrix(getArffData());
        }
    }

    @Override
    public Matrix getTrainingFeatures()
    {
        shouldShuffle = true;
        return super.getTrainingFeatures();
    }

    @Override
    public Matrix getTrainingData()
    {
        if (shouldShuffle)
        {
            trainingData.shuffle(rand);
            shouldShuffle = false;
        }
        return trainingData;
    }

    @Override
    public Matrix getTestingData()
    {
        return new Matrix(getArffData(), getTrainSize(), 0, getArffData().rows() - getTrainSize(), getArffData().cols());
    }

    @Override
    public Matrix getValidationData()
    {
        if (isUsingValidationSet())
        {
            return new Matrix(getArffData(), getTrainingSetSize(), 0, getValidationSetSize(), getArffData().cols());
        }
        else
        {
            return new Matrix();
        }
    }

    protected double getTrainPercent()
    {
        return trainPercent;
    }

    protected int getTrainSize()
    {
        return (int) (getTrainPercent() * getArffData().rows());
    }

}
