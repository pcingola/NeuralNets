package com.pcingola.neunet.kohonen;

import com.pcingola.neunet.Neuron;

/**
 * Kohonen's neuron
 * @author pcingola@sinectis.com
 */
public class KohonenNeuron extends Neuron {

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public KohonenNeuron(int i) {
		super(i);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------
	
	/**
	 * Calculate output based on input
	 * @return neurons' output
	 */
	public double calc() {
		double h = calcActivation();
		output = Math.exp(-h);
		return output;
	}

	/**
	 * Calculate and get activation
	 * @return activation
	 */
	public double calcActivation() {
		double diff, sum = 0;
		for( int i = 0; i < weight.length; i++ ) {
			if( input[i] == null ) throw new RuntimeException("Neuron's input not connected!");
			diff = weight[i] - input[i].getOutput();
			sum += diff * diff;
		}
		activation = Math.sqrt(sum);
		return activation;
	}

	/**
	 * Learning algorithm for a neuron
	 * @param eta: Learning rate
	 */
	public void learn(double eta) {
		for( int i = 0; i < weight.length; i++ ) {
			weight[i] += eta * (input[i].getOutput() - weight[i]);
		}
	}

}