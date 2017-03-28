package edu.byu.cstaheli.cs478.decision_tree.node;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by cstaheli on 3/11/2017.
 */
class NodeTest
{
    @Test
    void getOutputClass()
    {

    }

    @Test
    void findChildWithPrimaryColumnValue()
    {
        FeatureNode node = new FeatureNode(2, "Veggies");
        Node child1 = new LeafNode(1.0);
        child1.setPrimaryColumnValue(0.0);
        Node child2 = new LeafNode(1.0);
        child2.setPrimaryColumnValue(1.0);
        Node child3 = new LeafNode(0.0);
        child3.setPrimaryColumnValue(2.0);
        List<Node> children = new ArrayList<>(3);
        children.add(child1);
        children.add(child2);
        children.add(child3);
        node.setChildren(children);
        Node child = node.findChildWithPrimaryColumnValue(1.0);
        double outputClass = child.getOutputClass(new double[]{1.0});
        assertEquals(1.0, outputClass);
    }
}