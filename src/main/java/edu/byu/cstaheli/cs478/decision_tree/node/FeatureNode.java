package edu.byu.cstaheli.cs478.decision_tree.node;

import edu.byu.cstaheli.cs478.decision_tree.DecisionTree;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Utility;

import java.util.*;

/**
 * Created by cstaheli on 3/9/2017.
 */
public class FeatureNode extends Node
{
    private int splitOn;
    private String splitAttributeName;
    private List<Node> children;
    private boolean pruned;
    private double mostCommonChildOutputClass;

    public FeatureNode(int column, String splitAttributeName)
    {
        splitOn = column;
        this.splitAttributeName = splitAttributeName;
        children = new ArrayList<>();
        pruned = false;
    }

    @Override
    public double getOutputClass(double[] row)
    {
        if (!pruned)
        {
            Node wantedNode = this.findChildWithPrimaryColumnValue(row[splitOn]);
            double[] transformedRow = Utility.removeColumnFromRow(splitOn, row);
            if (wantedNode != null)
            {
                return wantedNode.getOutputClass(transformedRow);
            }
            else
            {
                Random random = new Random();
                int child = random.nextInt(getNumberOfChildren());
                return getChild(child).getOutputClass(transformedRow);
            }
        }
        else
        {
            return mostCommonChildOutputClass;
        }
    }

    @Override
    public int getMaxTreeDepth()
    {
        int maxDepth = 0;
        for (Node node : children)
        {
            int childMaxDepth = node.getMaxTreeDepth();
            if (childMaxDepth > maxDepth)
            {
                maxDepth = childMaxDepth;
            }
        }
        return 1 + maxDepth;
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

    public void tryPruning(LearningStrategy strategy, DecisionTree tree) throws Exception
    {
        double beforeAccuracy = tree.measureAccuracy(strategy.getValidationFeatures(), strategy.getValidationLabels(), null);
        pruned = true;
        mostCommonChildOutputClass = getMostCommonChildOutputClass();
        double afterAccuracy = tree.measureAccuracy(strategy.getValidationFeatures(), strategy.getValidationLabels(), null);
        if (afterAccuracy > beforeAccuracy)
        {
            children.clear();
        }
        else
        {
            pruned = false;
            mostCommonChildOutputClass = -1;
        }
    }

    public int getSplitColumn()
    {
        return splitOn;
    }

    @Override
    public String toString()
    {
        return String.format("Feature: Attr:\"%s(%s)\", SplitAttr:\"%s\"", getAttributeName(), getPrimaryColumnValue(), getSplitAttributeName());
    }

    public String getSplitAttributeName()
    {
        return splitAttributeName;
    }

    public List<Node> getChildren()
    {
        return Collections.unmodifiableList(children);
    }

    public void setChildren(List<Node> children)
    {
        this.children = children;
    }

    public int getNumberOfDescendants()
    {
        int counter = getNumberOfChildren();
        for (Node node : children)
        {
            if (node instanceof FeatureNode)
            {
                counter += ((FeatureNode) node).getNumberOfDescendants();
            }
            else if (node instanceof LeafNode)
            {
                counter += 1;
            }
        }
        return counter;
    }

    public double getMostCommonChildOutputClass()
    {
        Map<Double, Integer> map = new TreeMap<>();
        for (Node child : children)
        {
            double childOutputClass = -1;
            if (child instanceof LeafNode)
            {
                childOutputClass = child.getOutputClass(null);
            }
            else if (child instanceof FeatureNode)
            {
                childOutputClass = ((FeatureNode) child).getMostCommonChildOutputClass();
            }
            Integer count = map.get(childOutputClass);
            if (count == null)
            {
                map.put(childOutputClass, 1);
            }
            else
            {
                map.put(childOutputClass, count + 1);
            }
        }
        int maxCount = 0;
        double val = -1;
        for (Map.Entry<Double, Integer> entry : map.entrySet())
        {
            if (entry.getValue() > maxCount)
            {
                maxCount = entry.getValue();
                val = entry.getKey();
            }
        }
        return val;
    }
}
