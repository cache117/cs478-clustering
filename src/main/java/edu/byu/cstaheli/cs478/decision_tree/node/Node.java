package edu.byu.cstaheli.cs478.decision_tree.node;

/**
 * Created by cstaheli on 3/9/2017.
 */
public abstract class Node
{
    private double primaryColumnValue;
    private String attributeName;

    protected Node()
    {
        primaryColumnValue = -1;
    }

    public abstract double getOutputClass(double[] row);

    public double getPrimaryColumnValue()
    {
        return primaryColumnValue;
    }

    public void setPrimaryColumnValue(double primaryColumnValue)
    {
        this.primaryColumnValue = primaryColumnValue;
    }

    public String getAttributeName()
    {
        return attributeName;
    }

    public void setAttributeName(String attributeName)
    {
        this.attributeName = attributeName;
    }

    public abstract int getMaxTreeDepth();
}
