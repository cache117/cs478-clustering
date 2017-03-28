package edu.byu.cstaheli.cs478.perceptron;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;

import java.util.Random;

/**
 * Created by cstaheli on 1/27/2017.
 */
public class MultipleOutputPerceptron extends Perceptron
{
    private double wantedLabelIndex;

    public MultipleOutputPerceptron(Random rand, MLSystemManager manager, double wantedLabelIndex)
    {
        super(rand);
        this.wantedLabelIndex = wantedLabelIndex;
    }

    public void setWantedLabelIndex(double wantedLabelIndex)
    {
        this.wantedLabelIndex = wantedLabelIndex;
    }

    @Override
    protected double getActivation(double[] rowWeights, double[] row)
    {
        return super.getActivation(rowWeights, row);
    }

    protected double getExpected(double expected)
    {
        return expected == wantedLabelIndex ? 1 : 0;
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        double label = getActivation(getWeights(), features);
        labels[0] = label == 1 ? wantedLabelIndex : getRandom().getRandomInt(2);
    }
}
