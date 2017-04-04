package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.learner.unsupervised.UnsupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.jetbrains.annotations.NotNull;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.getFormattedDouble;

/**
 * Implements k-means clustering.
 */
public class KMeans extends UnsupervisedLearner
{
    private int k;
    private Random random;
    private List<Cluster> clusters;
    private double bestSilhouetteMetric;
    private int timesSilhouetteDecreased;
    private boolean useLastColumnOfDataset;
    private boolean useFirstColumnOfDataset;

    public KMeans(int k, Random random)
    {
        this.k = k;
        this.random = random;
        bestSilhouetteMetric = 0;
        clusters = new ArrayList<>(0);
        useLastColumnOfDataset = true;
        useFirstColumnOfDataset = true;
    }

    @NotNull
    private List<Cluster> populateInitialCentroids(Matrix dataset)
    {
        List<Cluster> centroids = new ArrayList<>(k);
        for (int i = 0; i < k; ++i)
        {
            Cluster newCluster = getRandomCentroid(dataset);
            centroids.add(newCluster);
        }
        return centroids;
    }

    @NotNull
    private Cluster getRandomCentroid(Matrix dataset)
    {
        int row = getRandomRow(dataset.rows());
        return getClusterFromRow(dataset, row);
    }

    @NotNull
    public Cluster getClusterFromRow(Matrix dataset, int row)
    {
        double[] centroid = dataset.row(row);
        return new Cluster(centroid, dataset);
    }

    @Override
    public void cluster(Matrix dataset)
    {
        if (!useLastColumnOfDataset)
        {
            dataset = new Matrix(dataset, 0, 0, dataset.rows(), dataset.cols() - 1);
        }
        if (!useFirstColumnOfDataset)
        {
            dataset = new Matrix(dataset, 0, 1, dataset.rows(), dataset.cols() - 1);
        }
        if (clusters.size() == 0)
        {
            clusters = populateInitialCentroids(dataset);
        }
        boolean keepTraining;
        int counter = 0;
        do
        {
            ++counter;
            printIterationHeader(counter);
            printCentroids();
            clearClusters();
            addRowsToClusters(dataset);
            fixEmptyClusters(dataset);
            calculateNewCentroids();
            keepTraining = shouldKeepTraining();
            printSilhouetteInfo();
            printSSE();
        } while (keepTraining);
        printFinalStats();
    }

    private void printFinalStats()
    {
        if (shouldOutput())
        {
            try (FileWriter writer = new FileWriter(getOutputFile(), true))
            {
                writer.append(String.format("***\n%s\n", clusters.size()));
                for (Cluster cluster : clusters)
                {
                    writer.append(cluster.getCentroidString(","));
                    writer.append("\n");
                    writer.append(String.valueOf(cluster.size()));
                    writer.append("\n");
                    writer.append(getFormattedDouble(cluster.calcSSE()));
                    writer.append("\n");
                }
                writer.append(getFormattedDouble(calculateTotalSSE()))
                        .append("\n");
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }

    private void printSSE()
    {
        System.out.printf("SSE: %s", getFormattedDouble(calculateTotalSSE()));
    }

    private double calculateTotalSSE()
    {
        double value = 0;
        for (Cluster cluster : clusters)
        {
            value += cluster.calcSSE();
        }
        return value;
    }

    private void fixEmptyClusters(Matrix dataset)
    {
        for (ListIterator<Cluster> iterator = clusters.listIterator(); iterator.hasNext(); )
        {
            Cluster cluster = iterator.next();
            if (cluster.empty())
            {
                iterator.remove();
                Cluster randomCentroid = getRandomCentroid(dataset);
                iterator.add(randomCentroid);
                clearClusters();
                addRowsToClusters(dataset);
                iterator = clusters.listIterator();
            }
        }
    }

    private void printSilhouetteInfo()
    {
        System.out.printf("\nSilhouette : %s\n", bestSilhouetteMetric);
    }

    private void printCentroids()
    {
        int counter = 0;
        printHeader("Printing Centroids");
        for (Cluster cluster : clusters)
        {
            System.out.printf("Centroid %s = ", counter);
            System.out.println(cluster.getCentroidString(",\t"));
            ++counter;
        }
    }

    private void printIterationHeader(int count)
    {
        System.out.printf("\n***************\nIteration %s\n***************\n", count);
    }

    private void clearClusters()
    {
        for (Cluster cluster : clusters)
        {
            cluster.clear();
        }
    }

    private void addRowsToClusters(Matrix dataset)
    {
        printHeader("Making Assignments");
        for (int i = 0; i < dataset.rows(); ++i)
        {
            if (i % 10 == 0)
            {
                System.out.println();
            }
            double[] row = dataset.row(i);
            Cluster bestCluster = getBestClusterForRow(row);
            bestCluster.add(row);
            printAssignment(i, bestCluster);
        }
    }

    private void printHeader(String header)
    {
        System.out.println(header);
    }

    private void printAssignment(int i, Cluster bestCluster)
    {
        System.out.printf("%s=%s ", i, clusters.indexOf(bestCluster));
    }

    private Cluster getBestClusterForRow(double[] row)
    {
        double bestDistance = Double.MAX_VALUE;
        Cluster bestCluster = clusters.get(0);
        for (Cluster cluster : clusters)
        {
            double distance = cluster.calcDistanceFromCentroid(row);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCluster = cluster;
            }
        }
        return bestCluster;
    }

    private void calculateNewCentroids()
    {
        for (Cluster cluster : clusters)
        {
            cluster.calculateNewCentroid();
        }
    }

    private boolean shouldKeepTraining()
    {
        double silhouetteMetric = 0;
        for (Cluster cluster : clusters)
        {
            double internalDissimilarity = cluster.calculateAverageInternalDissimilarity();
            double externalDissimilarity = Double.MAX_VALUE;
            for (Cluster otherCluster : clusters)
            {
                if (!cluster.equals(otherCluster))
                {
                    double dissimilarity = cluster.calculateAverageExternalDissimilarity(otherCluster);
                    if (dissimilarity < externalDissimilarity)
                    {
                        externalDissimilarity = dissimilarity;
                    }
                }
            }
            silhouetteMetric += silhouetteMetric(internalDissimilarity, externalDissimilarity);
        }

        double difference = Math.abs(silhouetteMetric - bestSilhouetteMetric);
        if (silhouetteMetric > bestSilhouetteMetric)
        {
            bestSilhouetteMetric = silhouetteMetric;
        }
        else
        {
            if (++timesSilhouetteDecreased > 2)
            {
                return false;
            }
        }
        return bestSilhouetteMetric == 0 || difference > .00001;
    }

    public int getK()
    {
        return k;
    }

    public void setK(int k)
    {
        this.k = k;
    }

    private int getRandomRow(int rows)
    {
        return random.nextInt(rows);
    }

    private double silhouetteMetric(double clusterDissimilarity, double otherClusterDissimilarity)
    {
        if (Double.compare(clusterDissimilarity, otherClusterDissimilarity) == -1)
        {
            return 1 - (clusterDissimilarity / otherClusterDissimilarity);
        }
        else if (Double.compare(clusterDissimilarity, otherClusterDissimilarity) == 0)
        {
            return 0;
        }
        else
        {
            return (otherClusterDissimilarity / clusterDissimilarity) - 1;
        }
    }

    public void setClusters(List<Cluster> clusters)
    {
        this.clusters = clusters;
    }

    public boolean shouldUseLastColumnOfDataset()
    {
        return useLastColumnOfDataset;
    }

    public void setUseLastColumnOfDataset(boolean useLastColumnOfDataset)
    {
        this.useLastColumnOfDataset = useLastColumnOfDataset;
    }

    public boolean shouldUseFirstColumnOfDataset()
    {
        return useFirstColumnOfDataset;
    }

    public void setUseFirstColumnOfDataset(boolean useFirstColumnOfDataset)
    {
        this.useFirstColumnOfDataset = useFirstColumnOfDataset;
    }
}
