package com.pcingola.neunet.ising;

import com.pcingola.neunet.Network;

/**
 * A thread that runs in background while calculating Ising's learning algorithm
 * @author pcingola@sinectis.com
 */
public class IsingNetworkThread extends Thread {

	protected IsingNeuNetDemo controller;
	protected IsingNetwork network;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public IsingNetworkThread(IsingNeuNetDemo controller, Network network) {
		super("IsingNetworkThread");
		this.controller = controller;
		this.network = (IsingNetwork) network;
		start();
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/** Run */
	public void run() {
		double tempMax = controller.getTempMax();
		double tempMin = controller.getTempMin();
		double temp = tempMax;
		int numIter = controller.getNumIterations();
		int displayStep = controller.getNumIterations() / 500;
		int updateParamStep = controller.getNumIterations() / 1000;
		int i;

		for( i = 0; i < numIter; i++ ) {
			// Show something every displayStep iterations

			if( (i % displayStep) == 0 ) {
				// Use 4 decimals to print 'temp'
				String tempStr = Double.toString(((int) (network.getTemp() * 1000000) / 1000000.0));
				String fieldStr = Double.toString(((int) (network.field() * 1000000) / 1000000.0));
				controller.setMessage("Iteration: " + i + "  Temp:" + tempStr + "    Field: " + fieldStr + "             ");
				controller.showNetwork(false);
			}

			// Update parameters
			if( (i % updateParamStep) == 0 ) {
				double percent = (double) i / ((double) numIter);
				temp = (tempMin * percent) + (tempMax * (1 - percent));
				network.setTemp(temp);
			}

			// Calculate network
			network.calc();

			// Check if someone stopped the network
			if( !controller.isRunning() ) return;
		}
		controller.setMessage("Finished");
		controller.setRunning(false);
	}
}
