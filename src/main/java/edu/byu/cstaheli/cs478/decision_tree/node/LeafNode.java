package edu.byu.cstaheli.cs478.decision_tree.node;

import java.util.List;

/**
 * Created by cstaheli on 3/9/2017.
 */
public class LeafNode extends Node
{
    private double outputClass;
    private List<Node> children;
    private boolean pruneAttempted;

    public LeafNode(double outputClass)
    {
        this.outputClass = outputClass;
        pruneAttempted = false;
    }

    @Override
    public double getOutputClass(double[] row)
    {
        return outputClass;
    }

    @Override
    public int getMaxTreeDepth()
    {
        return 1;
    }

    @Override
    public String toString()
    {
        return String.format("Leaf: Attr:\"%s\", Output:\"%s\"", getAttributeName(), getOutputClass());
    }

    private double getOutputClass()
    {
        return outputClass;
    }

    public void setChildren(List<Node> children)
    {
        this.children = children;
    }

    public Node getChild(int index)
    {
        return children.get(index);
    }

    public Node findChildWithPrimaryColumnValue(double primaryColumnValue)
    {
        for (Node node : children)
        {
            if (Double.compare(node.getPrimaryColumnValue(), primaryColumnValue) == 0)
            {
                return node;
            }
        }
        return null;
    }

    protected int getNumberOfChildren()
    {
        return children.size();
    }

    public boolean isPruneAttempted()
    {
        return pruneAttempted;
    }

    public void attemptToPrune()
    {
        this.pruneAttempted = true;
    }
}
