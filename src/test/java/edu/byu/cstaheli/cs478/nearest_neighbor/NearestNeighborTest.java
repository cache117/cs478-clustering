package edu.byu.cstaheli.cs478.nearest_neighbor;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.junit.jupiter.api.Test;

import java.io.File;

import static edu.byu.cstaheli.cs478.other.Other.assertNumbersEqualWithEpsilon;
import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.squaredDistance;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 3/14/2017.
 */
class NearestNeighborTest
{
    private static String datasetsLocation = "src/test/resources/datasets/nearest_neighbor/";

    @Test
    void distance()
    {

    }

    @Test
    void valueDistanceMetric()
    {

    }

    @Test
    void predict() throws Exception
    {
        Matrix trainingData = new Matrix(datasetsLocation + "homework.arff");
        NearestNeighbor nearestNeighbor = new NearestNeighbor();
        nearestNeighbor.setTrainingData(trainingData);
        nearestNeighbor.setNumberOfNeighborsToCompareTo(3);
        nearestNeighbor.setUseDistanceWeighting(false);
        double[] newFeatures = {.5, .2};
        double[] label = new double[1];
        nearestNeighbor.predict(newFeatures, label);
        double actual = label[0];
        assertEquals(0.0, actual);
    }

    @Test
    void euclideanDistance() throws Exception
    {

    }

    @Test
    void square() throws Exception
    {
        double first = 0;
        double second = 1;
        double result = squaredDistance(first, second);
        assertNumbersEqualWithEpsilon(1, result, .00001);
    }

    @Test
    void runManager() throws Exception
    {
//        String[] args;
//        MLSystemManager manager = new MLSystemManager();
//        args = ("-L knn -A " + datasetsLocation + "homework.arff -E training -V").split(" ");
//        manager.run(args);
//
//        args = ("-L knn -A " + datasetsLocation + "magicTelescopeTraining.arff -E static " + datasetsLocation + "magicTelescopeTesting.arff -N -V").split(" ");
//        manager.setCalcTrainingAccuracy(false);
//        manager.run(args);
        double startTime = System.currentTimeMillis();
//        testMagicTelescopeKTerms();
//        testHousingPriceKTerms();
//        testNominal();
        testExperiment();
        double elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time to complete (in seconds): " + elapsedTime / 1000.0);
    }

    private void testExperiment() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L knn -A " + datasetsLocation + "magicTelescopeTraining.arff -E static " + datasetsLocation + "magicTelescopeTesting.arff -N -V").split(" ");
        NearestNeighbor nearestNeighbor = new NearestNeighbor();
        nearestNeighbor.setUseDistanceWeighting(true);
        nearestNeighbor.setUseRegression(false);
        nearestNeighbor.setOutputFile(datasetsLocation + "magicTelescope/experiment.csv");
        assertTrue(new File(datasetsLocation + "magicTelescope/experiment.csv").delete());
        System.out.println("Running with K = " + 3);
        nearestNeighbor.setNumberOfNeighborsToCompareTo(3);
        manager.setLearner(nearestNeighbor);
        manager.run(args);
    }

    private void testMagicTelescopeKTerms() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L knn -A " + datasetsLocation + "magicTelescopeTraining.arff -E static " + datasetsLocation + "magicTelescopeTesting.arff -N -V").split(" ");
        NearestNeighbor nearestNeighbor = new NearestNeighbor();
        nearestNeighbor.setUseDistanceWeighting(true);
        nearestNeighbor.setUseRegression(false);
        nearestNeighbor.setOutputFile(datasetsLocation + "magicTelescope/kTest.csv");
        assertTrue(new File(datasetsLocation + "magicTelescope/kTest.csv").delete());
        testKValues(args, manager, nearestNeighbor);
    }

    private void testKValues(String[] args, MLSystemManager manager, NearestNeighbor nearestNeighbor) throws Exception
    {
        for (int i = 1; i < 16; i += 2)
        {
            System.out.println("Running with K = " + i);
            nearestNeighbor.setNumberOfNeighborsToCompareTo(i);
            manager.setLearner(nearestNeighbor);
            manager.run(args);
        }
    }

    private void testHousingPriceKTerms() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L knn -A " + datasetsLocation + "housingTraining.arff -E static " + datasetsLocation + "housingTesting.arff -N").split(" ");
        NearestNeighbor nearestNeighbor = new NearestNeighbor();
        nearestNeighbor.setUseDistanceWeighting(true);
        nearestNeighbor.setUseRegression(true);
        nearestNeighbor.setOutputFile(datasetsLocation + "housingPrice/kTest.csv");
        assertTrue(new File(datasetsLocation + "housingPrice/kTest.csv").delete());
        testKValues(args, manager, nearestNeighbor);
    }

    private void testNominal() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L knn -A " + datasetsLocation + "credit.arff -E random .85 -N -V").split(" ");
        NearestNeighbor nearestNeighbor = new NearestNeighbor();
        nearestNeighbor.setUseDistanceWeighting(true);
        nearestNeighbor.setUseRegression(false);
        nearestNeighbor.setOutputFile(datasetsLocation + "credit/test.csv");
//        assertTrue(new File(datasetsLocation + "credit/test.csv").delete());

        nearestNeighbor.setNumberOfNeighborsToCompareTo(3);
        manager.setLearner(nearestNeighbor);
        manager.run(args);
//        nearestNeighbor.setNumberOfNeighborsToCompareTo(7);
//        manager.setLearner(nearestNeighbor);
//        manager.run(args);
    }
}