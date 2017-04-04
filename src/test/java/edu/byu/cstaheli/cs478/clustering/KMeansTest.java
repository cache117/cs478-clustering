package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 4/2/2017.
 */
class KMeansTest
{
    private static String datasetsLocation = "src/test/resources/datasets/clustering/";

    @Test
    void cluster() throws Exception
    {
//        runBaseLineTest();
        runVerificationTest();
    }

    private void runBaseLineTest() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L kmeans -A " + datasetsLocation + "sponge.arff -E cluster 4 -V").split(" ");
        KMeans kMeans = new KMeans(4, new Random());
        kMeans.setUseLastColumnOfDataset(true);
        kMeans.setUseFirstColumnOfDataset(true);
        Matrix dataset = manager.getLearnerData().getArffData();
        List<Cluster> clusters = new ArrayList<>(4);
        for (int i = 0; i < 4; ++i)
        {
            clusters.add(kMeans.getClusterFromRow(dataset, i));
        }
        kMeans.setClusters(clusters);
        assertTrue(new File(datasetsLocation + "sponge/baseline.csv").delete());
        kMeans.setOutputFile(datasetsLocation + "sponge/baseline.csv");
        manager.setLearner(kMeans);
        manager.run(args);
    }

    private void runVerificationTest() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L kmeans -A " + datasetsLocation + "labor_data.arff -E cluster 5 -V").split(" ");
        manager.run(args);
        KMeans kMeans = new KMeans(5, new Random());
        kMeans.setUseLastColumnOfDataset(false);
        kMeans.setUseFirstColumnOfDataset(false);
        Matrix dataset = manager.getLearnerData().getArffData();
        dataset = new Matrix(dataset, 0, 0, dataset.rows(), dataset.cols() - 1);
        dataset = new Matrix(dataset, 0, 1, dataset.rows(), dataset.cols() - 1);
        List<Cluster> clusters = new ArrayList<>(5);
        for (int i = 0; i < 5; ++i)
        {
            clusters.add(kMeans.getClusterFromRow(dataset, i));
        }
        kMeans.setClusters(clusters);
        assertTrue(new File(datasetsLocation + "labor_data/verification.csv").delete());
        kMeans.setOutputFile(datasetsLocation + "labor_data/verification.csv");
        manager.setLearner(kMeans);
        manager.run(args);
    }

}