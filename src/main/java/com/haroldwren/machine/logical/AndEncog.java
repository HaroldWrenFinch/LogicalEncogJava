package com.haroldwren.machine.logical;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.text.DecimalFormat;
import java.text.NumberFormat;

public class AndEncog {
    public static double AND_INPUT[][] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
    };

    public static double AND_OUTPUT[][] = {
            {0.0},
            {0.0},
            {0.0},
            {1.0}
    };

    public static void main(String[] args) {
        BasicNetwork basicNetwork = new BasicNetwork();
        basicNetwork.addLayer(new BasicLayer(null, true, 2));
        basicNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), true, 6));
        basicNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        basicNetwork.getStructure().finalizeStructure();
        basicNetwork.reset();

        MLDataSet mlTrainingSet = new BasicMLDataSet(AND_INPUT, AND_OUTPUT);

        MLTrain mlTrain = new ResilientPropagation(basicNetwork, mlTrainingSet);

        int iteration = 1;
        do {
            mlTrain.iteration();
            System.out.println("Iteration: " + iteration + ", Error: " + mlTrain.getError());
            iteration++;
        } while (mlTrain.getError() > 0.001);

        NumberFormat formatter = new DecimalFormat("#0.000");
        System.out.println();
        System.out.println("Results: ");
        for(MLDataPair mlDataPair: mlTrainingSet) {
            final MLData output = basicNetwork.compute(mlDataPair.getInput());
            System.out.print("Input: " + mlDataPair.getInput().getData(0) + ", " + mlDataPair.getInput().getData(1));
            System.out.print(", Actual: " + formatter.format(output.getData(0)));
            System.out.println(", Ideal: " + mlDataPair.getIdeal().getData(0));
        }



    }

}
