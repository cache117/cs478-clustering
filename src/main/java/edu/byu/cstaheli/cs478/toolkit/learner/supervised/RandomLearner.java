package edu.byu.cstaheli.cs478.toolkit.learner.supervised;

import edu.byu.cstaheli.cs478.toolkit.utility.RandomWeightGenerator;

import java.util.Random;

/**
 * Created by cstaheli on 2/16/2017.
 */
public abstract class RandomLearner extends SupervisedLearner
{
    private RandomWeightGenerator random;
    private double learningRate;

    public RandomLearner(Random random)
    {
        super();
        this.setRandom(random);
    }

    protected double getRandomWeight()
    {
        // Gives numbers between -.5 and .5
        return random.getRandomWeight();
    }

    protected RandomWeightGenerator getRandom()
    {
        return random;
    }

    protected void setRandom(Random random)
    {
        this.random = RandomWeightGenerator.getInstance(random);
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }
}
