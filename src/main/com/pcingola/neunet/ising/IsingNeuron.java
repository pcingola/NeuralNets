package com.pcingola.neunet.ising;

import com.pcingola.neunet.Neuron;

/**
 * Ising's neuron
 * @author pcingola@sinectis.com
 */
public class IsingNeuron extends Neuron {

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public IsingNeuron(int i) {
		super(i);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	public int calc(double temp, int activation) {
		int out = 0;
		
		double beta = 1 / temp;
		double h = (double) activation;
		double g = 1.0 / (1.0 + Math.exp(-2.0 * beta * h));

		// Random output
		if( Math.random() < g ) out = 1;
		else out = -1;

		return out;
	}

	/**
	 * Calculate output based on input
	 * @return neurons' output
	 */
	public double calc() {
		throw new RuntimeException("Usual 'calc()' method should not be used for this neuron type");
	}
}
