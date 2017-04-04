package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by cstaheli on 4/3/2017.
 */
class ClusterTest
{
    private static String datasetsLocation = "src/test/resources/datasets/clustering/";

    @Test
    void calcDistanceFromCentroid()
    {
    }

    @Test
    void calcDistance() throws Exception
    {
        Matrix dataset = new Matrix(datasetsLocation + "labor_data.arff");
        dataset = new Matrix(dataset, 0, 0, dataset.rows(), dataset.cols() - 1);
        dataset = new Matrix(dataset, 0, 1, dataset.rows(), dataset.cols() - 1);
        Cluster cluster = new Cluster(dataset.row(3), dataset);
        double distance3 = cluster.calcDistanceFromCentroid(dataset.row(5));
        cluster = new Cluster(dataset.row(1), dataset);
        double distance1 = cluster.calcDistanceFromCentroid(dataset.row(5));
        assertTrue(distance3 < distance1);

        cluster = new Cluster(dataset.row(3), dataset);
        double distance14to3 = cluster.calcDistanceFromCentroid(dataset.row(14));
        cluster = new Cluster(dataset.row(1), dataset);
        double distance14to1 = cluster.calcDistanceFromCentroid(dataset.row(14));
        assertTrue(distance14to1 < distance14to3);

        cluster = new Cluster(dataset.row(0), dataset);
        double distance7to0 = cluster.calcDistanceFromCentroid(dataset.row(7));
        cluster = new Cluster(dataset.row(4), dataset);
        double distance7to4 = cluster.calcDistanceFromCentroid(dataset.row(7));
        assertTrue(distance7to0 < distance7to4);
    }

    @Test
    void calculateNewCentroid()
    {
    }

    @Test
    void calculateAverageInternalDissimilarity()
    {
    }

    @Test
    void calculateAverageExternalDissimilarity()
    {
    }

    @Test
    void calcSSE()
    {
    }

    @Test
    void calcAverageCentroid() throws Exception
    {
        Matrix dataset = new Matrix(datasetsLocation + "test.arff");
        Cluster cluster = new Cluster(new double[]{0.0, 1.0, 1.0}, dataset);
        for (int i = 0; i < dataset.rows(); ++i)
        {
            double[] row = dataset.row(i);
            cluster.add(row);
        }
        double[] averageCentroid = cluster.calcAverageCentroid();
        assertEquals(0.0, averageCentroid[0]);
        assertEquals(2.0, averageCentroid[1]);
        assertEquals(0.0, averageCentroid[2]);
    }

}