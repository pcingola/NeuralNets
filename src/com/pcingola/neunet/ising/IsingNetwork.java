package com.pcingola.neunet.ising;

import com.pcingola.neunet.Network;

/**
 * A Ising neural network
 * 
 * @author pcingola@sinectis.com
 */
public class IsingNetwork extends Network {

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public IsingNetwork(int sizeX, int sizeY) {
		super(1);
		layer[0] = new IsingLayer(sizeX, sizeY);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	public void calc() {
		layer[0].calc();
	}

	public double field() {
		return ((IsingLayer) layer[0]).field();
	}

	public double getTemp() {
		return ((IsingLayer) layer[0]).getTemp();
	}

	public void learn(double desiredOutputs[][]) {
		throw new RuntimeException("Nothing to learn!");
	}

	public void setTemp(double temp) {
		((IsingLayer) layer[0]).setTemp(temp);
	}

	public String toString() {
		return layer[0].toString();
	}
}
