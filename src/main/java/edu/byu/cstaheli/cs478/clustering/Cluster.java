package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.euclideanDistance;

/**
 * Created by cstaheli on 4/1/2017.
 */
public class Cluster
{
    private List<double[]> rows;
    private double[] centroid;
    private List<Boolean> continuousColumn;

    public Cluster(double[] centroid, Matrix dataset)
    {
        this.centroid = centroid;
        rows = new ArrayList<>();
        continuousColumn = populateContinuousColumnIndicators(dataset);
    }

    private List<Boolean> populateContinuousColumnIndicators(Matrix dataset)
    {
        List<Boolean> continuousColumnIndicators = new ArrayList<>(dataset.cols());
        for (int i = 0; i < dataset.cols(); ++i)
        {
            boolean indicator = dataset.valueCount(i) != 0;
            continuousColumnIndicators.add(indicator);
        }
        return continuousColumnIndicators;
    }

    private boolean isColumnContinuous(int column)
    {
        return continuousColumn.get(column);
    }

    public double calcDistance(double[] row)
    {
        assert row.length == continuousColumn.size();
        double distance = 0;
        for (int i = 0; i < row.length; ++i)
        {
            double centroidColumn = centroid[i];
            double rowColumn = row[i];
            if (isColumnContinuous(i))
            {
                distance += Double.compare(centroidColumn, rowColumn) == 0 ? 0 : 1;
            }
            else if (isValueUnknown(rowColumn))
            {
                distance += 1;
            }
            else
            {
                distance += euclideanDistance(centroidColumn, rowColumn);
            }
        }
        return distance;
    }

    private boolean isValueUnknown(double value)
    {
        return Double.compare(value, Matrix.MISSING) == 0;
    }

    public void calculateNewCentroid()
    {
        this.centroid = calcCentroidAverage();
    }

    private double[] calcCentroidAverage()
    {
        double[] centroid = new double[this.centroid.length];
        for (int i = 0; i < centroid.length; ++i)
        {
            centroid[i] = calcColumnAverage(i);
        }
        return centroid;
    }

    private double calcColumnAverage(int column)
    {
        double[] columnValues = getColumnValues(column);
        if (isColumnContinuous(column))
        {
            return calculateArrayAverage(columnValues);
        }
        else
        {
            return calculateArrayMedian(columnValues);
        }
    }

    private double calculateArrayAverage(double[] columnValues)
    {
        double sum = 0;
        int count = 0;
        for (double value : columnValues)
        {
            if (!isValueUnknown(value))
            {
                sum += value;
                ++count;
            }
        }
        return sum / count;
    }

    private double calculateArrayMedian(double[] columnValues)
    {
        Arrays.sort(columnValues);
        double median;
        if (columnValues.length % 2 == 0)
        {
            median = (columnValues[columnValues.length / 2] + columnValues[columnValues.length / 2 - 1]) / 2;
        }
        else
        {
            median = columnValues[columnValues.length / 2];
        }
        return median;
    }

    private double[] getColumnValues(int column)
    {
        double[] columnValues = new double[rows.size()];
        for (int i = 0; i < rows.size(); ++i)
        {
            columnValues[i] = get(i, column);
        }
        return columnValues;
    }

    private double get(int row, int column)
    {
        return rows.get(row)[column];
    }
}
