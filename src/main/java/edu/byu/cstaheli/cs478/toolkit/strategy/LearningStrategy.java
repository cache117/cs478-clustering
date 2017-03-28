package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.learner.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

/**
 * Created by cstaheli on 1/20/2017.
 */
public abstract class LearningStrategy
{
    private final static double TRAINING_PERCENT = .8;
    private LearnerData learnerData;
    private boolean useValidationSet;

    public LearningStrategy(LearnerData learnerData) throws Exception
    {
        this.learnerData = learnerData;
        this.useValidationSet = true;
    }

    public Matrix getTrainingFeatures()
    {
        return getFeaturesFromMatrix(getTrainingData());
    }

    public abstract Matrix getTrainingData();

    private Matrix getFeaturesFromMatrix(Matrix data)
    {
        return new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
    }

    public Matrix getTrainingLabels()
    {
        return getLabelsFromMatrix(getTrainingData());
    }

    private Matrix getLabelsFromMatrix(Matrix data)
    {
        return new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
    }

    public Matrix getTestingFeatures()
    {
        return getFeaturesFromMatrix(getTestingData());
    }

    public abstract Matrix getTestingData();

    public Matrix getTestingLabels()
    {
        return getLabelsFromMatrix(getTestingData());
    }

    public Matrix getValidationFeatures()
    {
        return getFeaturesFromMatrix(getValidationData());
    }

    public abstract Matrix getValidationData();

    public Matrix getValidationLabels()
    {
        return getLabelsFromMatrix(getValidationData());
    }

    protected int getTrainingSetSize()
    {
        return (int) (getTrainSize() * getTrainPercent());
    }

    protected double getTrainPercent()
    {
        return TRAINING_PERCENT;
    }

    protected int getTrainSize()
    {
        return getArffData().rows();
    }

    Matrix getArffData()
    {
        return learnerData.getArffData();
    }

    protected int getValidationSetSize()
    {
        if (isUsingValidationSet())
        {
            return (int) (getTrainSize() * (1 - getTrainPercent()));
        }
        else
        {
            return 0;
        }
    }

    public boolean isUsingValidationSet()
    {
        return useValidationSet;
    }

    public void setUseValidationSet(boolean useValidationSet)
    {
        this.useValidationSet = useValidationSet;
    }
}
