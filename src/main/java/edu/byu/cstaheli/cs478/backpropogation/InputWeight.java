package edu.byu.cstaheli.cs478.backpropogation;

/**
 * Created by cstaheli on 2/18/2017.
 */
public class InputWeight
{
    private double input;
    private double weight;
    private double previousWeightDelta;

    public InputWeight(double weight)
    {
        this.weight = weight;
    }

    public InputWeight(double input, double weight)
    {
        this(weight);
        this.input = input;
        previousWeightDelta = 0;
    }

    public double getInput()
    {
        return input;
    }

    public void setInput(double input)
    {
        this.input = input;
    }

    public double getWeight()
    {
        return weight;
    }

    public double calcNet()
    {
        return input * weight;
    }

    public void changeWeight(double learningRate, double outputError, double momentum)
    {
        previousWeightDelta = Node.calculateWeightDelta(learningRate, outputError, input, previousWeightDelta, momentum);
        applyWeightChange(previousWeightDelta);
    }

    private void applyWeightChange(double deltaWeight)
    {
        this.weight += deltaWeight;
    }
}
