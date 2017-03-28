package edu.byu.cstaheli.cs478.decision_tree;

import edu.byu.cstaheli.cs478.decision_tree.node.FeatureNode;
import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Map;
import java.util.TreeMap;

import static edu.byu.cstaheli.cs478.other.Other.assertNumberBetween;
import static edu.byu.cstaheli.cs478.other.Other.assertNumbersEqualWithEpsilon;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 3/1/2017.
 */
public class DecisionTreeTest
{
    private static String datasetsLocation = "src/test/resources/datasets/decision_tree/";

    @Test
    void getFeatureNode() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
        FeatureNode featureNode = (FeatureNode) tree.getFeatureNode(matrix);
        int splitOn = featureNode.getSplitColumn();
        assertEquals(0, splitOn);
        double primaryColumnValue = featureNode.getPrimaryColumnValue();
        assertEquals(0, primaryColumnValue);

        Matrix noMeatMatrix = matrix.getRowsWithColumnClass(0, 0.0);
        featureNode = (FeatureNode) tree.getFeatureNode(noMeatMatrix);
        splitOn = featureNode.getSplitColumn();
        assertEquals(1, splitOn);

        Matrix yesMeatMatrix = matrix.getRowsWithColumnClass(0, 0.0);
        featureNode = (FeatureNode) tree.getFeatureNode(yesMeatMatrix);
        splitOn = featureNode.getSplitColumn();
        assertEquals(1, splitOn);
    }

    @Test
    void getNodeChildren() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
    }

    @Test
    void getLeafNodeIfAny() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
    }

    @Test
    void getMostProbable() throws Exception
    {
        Map<Double, Integer> outputDistribution = new TreeMap<>();
        outputDistribution.put(0.0, 5);
        outputDistribution.put(1.0, 15);
        outputDistribution.put(2.0, 14);
        DecisionTree tree = new DecisionTree();
        double mostProbable = tree.getMostProbable(outputDistribution);
        assertEquals(1.0, mostProbable);
        outputDistribution.put(3.0, 15);
        mostProbable = tree.getMostProbable(outputDistribution);
        assertEquals(1.0, mostProbable);
        outputDistribution.put(4.0, 16);
        mostProbable = tree.getMostProbable(outputDistribution);
        assertEquals(4.0, mostProbable);
    }

    @Test
    void getPureClassIfAny() throws Exception
    {
        Map<Double, Integer> outputDistribution = new TreeMap<>();
        outputDistribution.put(0.0, 5);
        outputDistribution.put(1.0, 15);
        outputDistribution.put(2.0, 14);
        DecisionTree tree = new DecisionTree();
        Map.Entry<Double, Integer> pureClassIfAny = tree.getPureClassIfAny(outputDistribution);
        assertEquals(null, pureClassIfAny);

        outputDistribution = new TreeMap<>();
        outputDistribution.put(0.0, 0);
        outputDistribution.put(1.0, 0);
        outputDistribution.put(2.0, 14);
        pureClassIfAny = tree.getPureClassIfAny(outputDistribution);
        assertEquals(2.0, (double) pureClassIfAny.getKey());
    }

    @Test
    void getBestFeature() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
        int bestFeature = tree.getBestFeature(matrix);
        assertEquals(0, bestFeature);
        Matrix meatMatrix = matrix.getRowsWithColumnClass(0, 0.0);
        bestFeature = tree.getBestFeature(meatMatrix);
        assertEquals(1, bestFeature);
        Matrix veggieMatrix = matrix.getRowsWithColumnClass(1, 0);
    }

    @Test
    void calculateOutputInformation() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
        double outputInformation = tree.calculateOutputInformation(matrix);
        assertNumberBetween(outputInformation, 1.530, 1.531);
    }

    @Test
    void calculateFeatureInformation() throws Exception
    {
        Matrix matrix = new Matrix(datasetsLocation + "pizza.arff");
        DecisionTree tree = new DecisionTree();
        double featureInformation = tree.calculateFeatureInformation(matrix, 0);
        assertNumbersEqualWithEpsilon(.983, featureInformation, .001);
        featureInformation = tree.calculateFeatureInformation(matrix, 1);
        assertNumbersEqualWithEpsilon(1.417, featureInformation, .001);
        featureInformation = tree.calculateFeatureInformation(matrix, 2);
        assertNumbersEqualWithEpsilon(1.29, featureInformation, .01);
        //Split on meat
        Matrix meatMatrix = matrix.getRowsWithColumnClass(0, 0.0);
        featureInformation = tree.calculateFeatureInformation(meatMatrix, 0);
        assertNumbersEqualWithEpsilon(.5, featureInformation, .1);
        featureInformation = tree.calculateFeatureInformation(meatMatrix, 1);
        assertNumbersEqualWithEpsilon(0, featureInformation, .1);
    }

    @Test
    void runManager() throws Exception
    {
//        String[] args;
//        MLSystemManager manager = new MLSystemManager();
//        args = ("-L decisiontree -A " + datasetsLocation + "pizza.arff -E training -V").split(" ");
//        manager.run(args);
//
//        args = ("-L decisiontree -A " + datasetsLocation + "tennis.arff -E training -V").split(" ");
//        manager.run(args);
//
//        args = ("-L decisiontree -A " + datasetsLocation + "lenses.arff -E training -V").split(" ");
//        manager.run(args);
//
//        args = ("-L decisiontree -A " + datasetsLocation + "voting.arff -E training -V").split(" ");
//        manager.run(args);

//        args = ("-L decisiontree -A " + datasetsLocation + "cars.arff -E training -V").split(" ");
//        manager.run(args);
//        runCarsTestNoPruning();
//        runVotingTestNoPruning();
//        runCarsTestPruning();
//        runVotingTestPruning();
        runIris();
    }

    private void runIris() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        manager.binRealData(true);
        args = ("-L decisiontree -A " + datasetsLocation + "iris.arff -E training -V").split(" ");
        manager.run(args);
    }

    private void runCarsTestNoPruning() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        assertTrue(new File(datasetsLocation + "carTests/car1.csv").delete());
        for (int i = 0; i < 5; ++i)
        {
            DecisionTree decisionTree = new DecisionTree();
            decisionTree.setOutputFile(datasetsLocation + "carTests/car1.csv");
            manager.setLearner(decisionTree);
            args = ("-L decisiontree -A " + datasetsLocation + "cars.arff -E cross 10 -V").split(" ");
            manager.run(args);
        }
    }

    private void runCarsTestPruning() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        assertTrue(new File(datasetsLocation + "carTests/car2.csv").delete());
        for (int i = 0; i < 5; ++i)
        {
            DecisionTree decisionTree = new DecisionTree();
            decisionTree.setOutputFile(datasetsLocation + "carTests/car2.csv");
            decisionTree.shouldPrune(true);
            manager.setLearner(decisionTree);
            args = ("-L decisiontree -A " + datasetsLocation + "cars.arff -E cross 10 -V").split(" ");
            manager.run(args);
        }
    }

    private void runVotingTestNoPruning() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        assertTrue(new File(datasetsLocation + "votingTests/voting1.csv").delete());
        for (int i = 0; i < 5; ++i)
        {
            DecisionTree decisionTree = new DecisionTree();
            decisionTree.setOutputFile(datasetsLocation + "votingTests/voting1.csv");
            manager.setLearner(decisionTree);
            args = ("-L decisiontree -A " + datasetsLocation + "voting.arff -E cross 10 -V").split(" ");
            manager.run(args);
        }
    }

    private void runVotingTestPruning() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        assertTrue(new File(datasetsLocation + "votingTests/voting2.csv").delete());
        for (int i = 0; i < 5; ++i)
        {
            DecisionTree decisionTree = new DecisionTree();
            decisionTree.setOutputFile(datasetsLocation + "votingTests/voting2.csv");
            decisionTree.shouldPrune(true);
            manager.setLearner(decisionTree);
            args = ("-L decisiontree -A " + datasetsLocation + "voting.arff -E cross 10 -V").split(" ");
            manager.run(args);
        }
    }
}