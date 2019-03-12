package com.pcingola.neunet.kohonen;

import com.pcingola.neunet.Network;

/**
 * A thread that runs in background while calculating kohonen's learning algorithm
 * @author pcingola@sinectis.com
 */
public class KohonenNetworkThread extends Thread {

	protected KohonenNeuNetDemo controller;
	protected KohonenNetwork network;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public KohonenNetworkThread(KohonenNeuNetDemo controller, Network network) {
		super("KohonenNetworkThread");
		this.controller = controller;
		this.network = (KohonenNetwork) network;
		start();
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Learn input patterns
	 */
	public void learn() {
		double neighMax = controller.getNeighMax();
		double neighMin = controller.getNeighMin();
		double neigh = neighMax;
		double etaMax = controller.getEtaMax();
		double etaMin = controller.getEtaMin();
		double eta = etaMax;
		int numIter = controller.getNumIterations();
		int displayStep = controller.getNumIterations() / 500;
		int updateParamStep = controller.getNumIterations() / 1000;
		int i;

		for( i = 0; i < numIter; i++ ) {
			// Show something every displayStep iterations
			if( (i % displayStep) == 0 ) {
				// Use 4 decimals to print 'eta'
				String etaStr = Double.toString(((int) (network.getEta() * 1000000) / 1000000.0));
				controller.setMessage("Iteration: " + i + "  " + "Neighbourhood:" + network.getNeighbourhood() + "  " + "Eta:" + etaStr + "          ");
				controller.clear();
				controller.showInputs(false);
				controller.showWeights(false);
			}

			// Update parameters
			if( (i % updateParamStep) == 0 ) {
				double percent = (double) i / ((double) numIter);
				eta = (etaMin * percent) + (etaMax * (1 - percent));
				neigh = (neighMin * percent) + (neighMax * (1 - percent));
				network.setEta(eta);
				network.setNeighbourhood((int) neigh);
			}

			// Get a pattern into 'input' layer
			controller.setInputPattern();

			// Learn that pattern
			network.learn(null);

			// Check if someone stopped the network
			if( !controller.isRunning() ) return;
		}
		controller.setMessage("Finished");
		controller.setRunning(false);
	}

	/** Run */
	public void run() {
		learn();
	}

}