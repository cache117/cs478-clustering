package edu.byu.cstaheli.cs478.toolkit.strategy;

import edu.byu.cstaheli.cs478.toolkit.learner.LearnerData;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.Random;

/**
 * Created by cstaheli on 1/20/2017.
 */
public class StaticStrategy extends LearningStrategy
{
    private Matrix testData;

    public StaticStrategy(LearnerData learnerData) throws Exception
    {
        super(learnerData);
        testData = new Matrix();
        testData.loadArff(learnerData.getEvalParameter());
        if (learnerData.isNormalized())
        {
            testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!
        }
        System.out.println("Calculating accuracy on separate test set...");
        System.out.println("Test set name: " + learnerData.getEvalParameter());
        System.out.println("Number of test instances: " + getTestingData().rows());
    }

    @Override
    public Matrix getTrainingData()
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
    public Matrix getTestingData()
    {
//        return testData;
        testData.shuffle(new Random());
        return new Matrix(testData, 0, 0, 2000, testData.cols());
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
}
