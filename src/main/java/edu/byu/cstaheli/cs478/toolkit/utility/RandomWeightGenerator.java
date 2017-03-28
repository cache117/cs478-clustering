package edu.byu.cstaheli.cs478.toolkit.utility;

import java.util.Random;

/**
 * Created by cstaheli on 2/18/2017.
 */
public class RandomWeightGenerator
{
    private static RandomWeightGenerator _instance;
    private Random random;

    private RandomWeightGenerator()
    {
        random = new Random();
    }

    public static RandomWeightGenerator getInstance()
    {
        if (_instance == null)
        {
            _instance = new RandomWeightGenerator();
        }
        return _instance;
    }

    public static RandomWeightGenerator getInstance(long seed)
    {
        RandomWeightGenerator random = getInstance();
        random.setRandom(seed);
        return random;
    }

    public static RandomWeightGenerator getInstance(Random random)
    {
        RandomWeightGenerator generator = getInstance();
        generator.setRandom(random);
        return generator;
    }

    public double getRandomWeight()
    {
        return getRandom().nextDouble() - 0.5;
    }

    public int getRandomInt(int bound)
    {
        return getRandom().nextInt(bound);
    }

    public Random getRandom()
    {
        return random;
    }

    public void setRandom(Random random)
    {
        this.random = random;
    }

    public void setRandom(long seed)
    {
        this.random = new Random(seed);
    }
}
