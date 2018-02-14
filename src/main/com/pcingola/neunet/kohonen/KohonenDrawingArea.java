package com.pcingola.neunet.kohonen;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JComponent;

import com.pcingola.neunet.Layer;
import com.pcingola.neunet.Neuron;

/**
 * A drawing area
 * 
 * @author pcingola@sinectis.com
 */
public class KohonenDrawingArea extends JComponent {

	protected KohonenNeuNetDemo controller;
	protected Dimension preferredSize;

	//-------------------------------------------------------------------------
	// Constructor
	//-------------------------------------------------------------------------

	public KohonenDrawingArea(KohonenNeuNetDemo controller) {
		this.controller = controller;
		setBackground(Color.WHITE);
		setOpaque(true);
		preferredSize = new Dimension(600, 400);
	}

	//-------------------------------------------------------------------------
	// Methods
	//-------------------------------------------------------------------------

	/**
	 * Clear screen
	 */
	public void clear() {
		paintComponent(this.getGraphics());
	}

	public Dimension getPreferredSize() {
		return preferredSize;
	}

	protected void paintComponent(Graphics g) {
		//Paint background if we're opaque.
		if( isOpaque() ) {
			g.setColor(getBackground());
			g.fillRect(0, 0, getWidth(), getHeight());
		}
	}

	/**
	 * Run a network
	 */
	protected void runNetwork() {
		new KohonenNetworkThread(controller, controller.getNetwork());
	}

	/**
	 * Show network's input patterns
	 */
	protected void showInputs() {
		double input[][][] = controller.getNetwork().getInput();
		if( input != null ) {
			int i, x, y;
			Graphics g = this.getGraphics();
			for( i = 0; i < input.length; i++ ) {
				x = (int) (input[i][0][0] * getWidth());
				y = (int) (input[i][1][0] * getHeight());
				g.fillRect(x - 3, y - 3, 6, 6);
			}
		}
	}

	/**
	 * Show network's weights patterns
	 */
	protected void showWeights() {
		if( (controller.getNetwork() != null) && (controller.getNetwork().getLayer(1) != null) ) {
			int i, j, x, y;
			Layer lay = controller.getNetwork().getLayer(1);
			int sizeX = lay.getSizeX();
			int sizeY = lay.getSizeY();
			int prevx = 0, prevy = 0, px, py;
			Neuron neu;

			Graphics g = getGraphics();
			g.setColor(getForeground());
			int width = getWidth();
			int heigth = getHeight();

			for( i = 0; i < sizeX; i++ ) {
				neu = lay.getNeuron(i, 0);
				prevx = (int) (neu.getWeight(0) * width);
				prevy = (int) (neu.getWeight(1) * heigth);

				for( j = 0; j < sizeY; j++ ) {
					neu = lay.getNeuron(i, j);
					x = (int) (neu.getWeight(0) * width);
					y = (int) (neu.getWeight(1) * heigth);
					g.drawLine(prevx, prevy, x, y);		// Draw a line between 'this' weight and previous one
					if( (i - 1) >= 0 ) {
						neu = lay.getNeuron(i - 1, j);
						px = (int) (neu.getWeight(0) * width);
						py = (int) (neu.getWeight(1) * heigth);
						g.drawLine(px, py, x, y);	// Draw a line between 'this' weight the neuron to the 'right'
					}
					prevx = x;
					prevy = y;
				}
			}
		}
	}

}

