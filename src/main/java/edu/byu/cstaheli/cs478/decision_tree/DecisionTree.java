package edu.byu.cstaheli.cs478.decision_tree;

import edu.byu.cstaheli.cs478.decision_tree.node.FeatureNode;
import edu.byu.cstaheli.cs478.decision_tree.node.LeafNode;
import edu.byu.cstaheli.cs478.decision_tree.node.Node;
import edu.byu.cstaheli.cs478.toolkit.exception.MatrixException;
import edu.byu.cstaheli.cs478.toolkit.learner.SupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.strategy.LearningStrategy;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Created by cstaheli on 3/1/2017.
 */
public class DecisionTree extends SupervisedLearner
{
    private final static Logger LOGGER = Logger.getLogger(DecisionTree.class.getName());
    private FeatureNode decisionTreeRoot;
    private boolean prune;

    @Override
    public void train(LearningStrategy strategy) throws Exception
    {
        Matrix trainingFeatures = strategy.getTrainingFeatures();
        Matrix trainingLabels = strategy.getTrainingLabels();
        Matrix trainingData = strategy.getTrainingData();
        decisionTreeRoot = (FeatureNode) populateDecisionTree(trainingData);
        if (prune)
        {
            prune(strategy);
        }
        outputFinalStatistics(strategy);
    }

    private void prune(LearningStrategy strategy) throws Exception
    {
        decisionTreeRoot.tryPruning(strategy, this);
    }

    public Node populateDecisionTree(Matrix matrix) throws MatrixException
    {
        Node leafNode = getLeafNodeIfAny(matrix);
        if (leafNode != null)
        {
            return leafNode;
        }

        return getFeatureNode(matrix);
    }

    protected Node getLeafNodeIfAny(Matrix matrix)
    {
        Map<Double, Integer> outputDistribution = matrix.getColumnOccurrences(matrix.cols() - 1);
//        LOGGER.info("Output Class Distributions:");
//        for (Map.Entry<Double, Integer> entry : outputDistribution.entrySet())
//        {
//            LOGGER.info(String.format("Class: %s, Occurrences: %s", entry.getKey(), entry.getValue()));
//        }
        Map.Entry<Double, Integer> pureClassIfAny = getPureClassIfAny(outputDistribution);
        if (pureClassIfAny != null)
        {
            return new LeafNode(pureClassIfAny.getKey());
        }
        else if (matrix.cols() == 2)
        {
            return new LeafNode(getMostProbable(outputDistribution));
        }
        return null;
    }

    protected Map.Entry<Double, Integer> getPureClassIfAny(Map<Double, Integer> occurrences)
    {
        Map.Entry<Double, Integer> pureEntry = null;
        int numberOfZeroOccurrences = 0;
        for (Map.Entry<Double, Integer> entry : occurrences.entrySet())
        {
            int occurrenceCount = entry.getValue();
            if (occurrenceCount == 0)
            {
                ++numberOfZeroOccurrences;
            }
            else
            {
                pureEntry = entry;
            }
        }
        if (numberOfZeroOccurrences == occurrences.size() - 1)
        {
            return pureEntry;
        }
        return null;
    }

    protected double getMostProbable(Map<Double, Integer> outputDistribution)
    {
        Double bestKey = 0.0;
        int maxCount = 0;
        for (Map.Entry<Double, Integer> entry : outputDistribution.entrySet())
        {
            if (entry.getValue() > maxCount)
            {
                maxCount = entry.getValue();
                bestKey = entry.getKey();
            }
        }
        return bestKey;
    }

    protected Node getFeatureNode(Matrix matrix) throws MatrixException
    {
        int bestFeature = getBestFeature(matrix);
        String attributeName = matrix.attrName(bestFeature);
        FeatureNode node = new FeatureNode(bestFeature, attributeName);
        List<Node> children = getNodeChildren(matrix, bestFeature);
        node.setChildren(children);
        return node;
    }

    protected List<Node> getNodeChildren(Matrix matrix, int bestFeature) throws MatrixException
    {
        Map<Double, Integer> featureDistribution = matrix.getColumnOccurrences(bestFeature);
        List<Node> children = new ArrayList<>();
        for (Map.Entry<Double, Integer> entry : featureDistribution.entrySet())
        {
            Matrix filteredMatrix = matrix.getRowsWithColumnClass(bestFeature, entry.getKey());
            Node node = populateDecisionTree(filteredMatrix);
            node.setPrimaryColumnValue(entry.getKey());
            node.setAttributeName(matrix.attrName(bestFeature));
            children.add(node);
        }
        return children;
    }

    public int getBestFeature(Matrix matrix) throws MatrixException
    {
        double outputInformation = calculateOutputInformation(matrix);
        double bestInformationGain = 0;
        int bestFeature = -1;
        for (int i = 0; i < matrix.cols() - 1; ++i)
        {
            double featureInformation = calculateFeatureInformation(matrix, i);
            double informationGain = outputInformation - featureInformation;
            if (informationGain > bestInformationGain)
            {
                bestInformationGain = informationGain;
                bestFeature = i;
            }
        }
        return bestFeature;
    }

    public double calculateOutputInformation(Matrix matrix)
    {
        int lastColumn = matrix.cols() - 1;
        Map<Double, Integer> allOccurrences = matrix.getColumnOccurrences(lastColumn);
        assert matrix.valueCount(lastColumn) > 0;
        assert matrix.valueCount(lastColumn) >= allOccurrences.size();
        int totalOccurrences = matrix.rows();
        double information = 0;
        for (Map.Entry<Double, Integer> entry : allOccurrences.entrySet())
        {
            int occurrences = entry.getValue();
            information += calculateEntropy(calculateProbability(occurrences, totalOccurrences));
        }
        return information;
    }

    public double calculateFeatureInformation(Matrix matrix, int feature) throws MatrixException
    {
        Map<Double, Integer> columnOccurrences = matrix.getColumnOccurrences(feature);
        double information = 0;
        int totalOccurrences = matrix.rows();
        for (Map.Entry<Double, Integer> entry : columnOccurrences.entrySet())
        {
            Matrix relevantMatrix = matrix.getRowsWithColumnClass(feature, entry.getKey());
            information += calculateOutputInformation(relevantMatrix) * calculateProbability(entry.getValue(), totalOccurrences);
        }
        return information;
    }

    private double calculateEntropy(double probability)
    {
        if (probability == 0)
        {
            return 0;
        }
        return -1 * probability * log2(probability);
    }

    private double log2(double value)
    {
        return Math.log(value) / Math.log(2);
    }

    private double calculateProbability(int favorableOutcomes, int totalOutcomes)
    {
        assert totalOutcomes != 0;
        assert favorableOutcomes <= totalOutcomes;
        return ((double) favorableOutcomes / (double) totalOutcomes);
    }

    private void outputFinalStatistics(LearningStrategy strategy)
    {
        if (shouldOutput())
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                double trainingAccuracy = measureAccuracy(strategy.getTrainingFeatures(), strategy.getTrainingLabels(), null);
                double validationAccuracy = measureAccuracy(strategy.getValidationFeatures(), strategy.getValidationLabels(), null);
                double testingAccuracy = measureAccuracy(strategy.getTestingFeatures(), strategy.getTestingLabels(), null);
                int numberOfNodesInTree = getNumberOfNodesInTree();
                int depthOfTree = getDepthOfTree();
                writer.append(String.format("%s, %s, %s, %s, %s\n", trainingAccuracy, validationAccuracy, testingAccuracy, numberOfNodesInTree, depthOfTree));
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception
    {
        labels[0] = decisionTreeRoot.getOutputClass(features);
    }

    public void shouldPrune(boolean prune)
    {
        this.prune = prune;
    }

    public int getNumberOfNodesInTree()
    {
        return 1 + decisionTreeRoot.getNumberOfDescendants();
    }

    public int getDepthOfTree()
    {
        return decisionTreeRoot.getMaxTreeDepth();
    }
}
